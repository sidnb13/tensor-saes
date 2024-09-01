import os
import socket
from collections import defaultdict
from functools import partial
from typing import Any, Type, TypeVar, cast

import torch
from accelerate.utils import send_to_device
from torch import Tensor, nn
from torch.distributed._tensor import (
    Replicate,
    Shard,
    distribute_tensor,
    init_device_mesh,
)
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
)
from transformers import PreTrainedModel

from .logger import get_logger

T = TypeVar("T")

logger = get_logger(__name__)


def get_open_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # bind to all interfaces and use an OS provided port
        return s.getsockname()[1]  # return only the port number


def log_parameter_norms(sae, names_str, info):
    with torch.no_grad():
        encoder_norm = torch.norm(sae.encoder.weight).item()
        decoder_norm = torch.norm(sae.W_dec).item()
        bias_norm = sae.b_dec.norm().item()

        info.update(
            {
                f"encoder_norm/{names_str}": encoder_norm,
                f"decoder_norm/{names_str}": decoder_norm,
                f"bias_norm/{names_str}": bias_norm,
            }
        )


def configure_tp_model(model, world_size: int):
    tp_mesh = init_device_mesh("cuda", (world_size,))
    tp_plan = {
        "encoder": ColwiseParallel(),
    }
    model = parallelize_module(model, tp_mesh, tp_plan)

    # Rowwise parallel sharding
    w_dec_shard = nn.Parameter(
        distribute_tensor(model.W_dec.data, tp_mesh, placements=[Shard(0)])
    )
    model.register_parameter("W_dec", w_dec_shard)
    b_dec_repl = nn.Parameter(
        distribute_tensor(model.b_dec.data, tp_mesh, placements=[Replicate()])
    )
    model.register_parameter("b_dec", b_dec_repl)

    from torch.distributed._tensor import DTensor

    print("Encoder weight", DTensor.to_local(model.encoder.weight).shape)
    print("W_dec", DTensor.to_local(model.W_dec).shape)

    model.tp_mesh = tp_mesh

    return model


def assert_type(typ: Type[T], obj: Any) -> T:
    """Assert that an object is of a given type at runtime and return it."""
    if not isinstance(obj, typ):
        raise TypeError(f"Expected {typ.__name__}, got {type(obj).__name__}")

    return cast(typ, obj)


@torch.no_grad()
def geometric_median(points: Tensor, max_iter: int = 100, tol: float = 1e-5):
    """Compute the geometric median `points`. Used for initializing decoder bias."""
    # Initialize our guess as the mean of the points
    guess = points.mean(dim=0)
    prev = torch.zeros_like(guess)

    # Weights for iteratively reweighted least squares
    weights = torch.ones(len(points), device=points.device)

    for _ in range(max_iter):
        prev = guess

        # Compute the weights
        weights = 1 / torch.norm(points - guess, dim=1)

        # Normalize the weights
        weights /= weights.sum()

        # Compute the new geometric median
        guess = (weights.unsqueeze(1) * points).sum(dim=0)

        # Early stopping condition
        if torch.norm(guess - prev) < tol:
            break

    return guess


def get_layer_list(model: PreTrainedModel) -> tuple[str, nn.ModuleList]:
    """Get the list of layers to train SAEs on."""
    N = assert_type(int, model.config.num_hidden_layers)
    candidates = [
        (name, mod)
        for (name, mod) in model.named_modules()
        if isinstance(mod, nn.ModuleList) and len(mod) == N
    ]
    assert len(candidates) == 1, "Could not find the list of layers."

    return candidates[0]


@torch.inference_mode()
def resolve_widths(
    model: PreTrainedModel,
    module_names: list[str],
    dim: int = -1,
) -> dict[str, int]:
    """Find number of output dimensions for the specified modules."""
    module_to_name = {model.get_submodule(name): name for name in module_names}
    shapes: dict[str, int] = {}

    def hook(module, _, output):
        # Unpack tuples if needed
        if isinstance(output, tuple):
            output, *_ = output

        name = module_to_name[module]
        shapes[name] = output.shape[dim]

    handles = [mod.register_forward_hook(hook) for mod in module_to_name]
    dummy = send_to_device(model.dummy_inputs, model.device)
    try:
        model(**dummy)
    finally:
        for handle in handles:
            handle.remove()

    return shapes


@torch.inference_mode()
def resolve_widths_rangewise(
    model: PreTrainedModel,
    module_names_range: list[list[str]],
    dim: int = -1,
) -> dict[str, int]:
    """Find number of output dimensions for the specified modules."""
    module_to_name = [
        {model.get_submodule(name): name for name in module_names}
        for module_names in module_names_range
    ]
    shapes = defaultdict(int)

    def hook(i, module, _, output):
        # Unpack tuples if needed
        if isinstance(output, tuple):
            output, *_ = output

        # accumulate the shapes
        shapes[module_names_range[i]] += output.shape[dim]

    handles = [
        [mod.register_forward_hook(partial(hook, i)) for mod in module_list]
        for i, module_list in enumerate(module_to_name)
    ]
    dummy = send_to_device(model.dummy_inputs, model.device)
    try:
        model(**dummy)
    finally:
        for handle_list in handles:
            for handle in handle_list:
                handle.remove()

    return shapes


# Fallback implementation of SAE decoder
def eager_decode(top_indices: Tensor, top_acts: Tensor, W_dec: Tensor):
    buf = top_acts.new_zeros(top_acts.shape[:-1] + (W_dec.shape[-1],))
    acts = buf.scatter_(dim=-1, index=top_indices, src=top_acts)
    return acts @ W_dec.mT


# Triton implementation of SAE decoder
def triton_decode(top_indices: Tensor, top_acts: Tensor, W_dec: Tensor):
    return TritonDecoder.apply(top_indices, top_acts, W_dec)


try:
    from .kernels import TritonDecoder
except ImportError:
    decoder_impl = eager_decode
    logger.info("Triton not installed, using eager implementation of SAE decoder.")
else:
    if os.environ.get("SAE_DISABLE_TRITON") == "1":
        logger.info("Triton disabled, using eager implementation of SAE decoder.")
        decoder_impl = eager_decode
    else:
        decoder_impl = triton_decode
