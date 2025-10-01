import random
from pathlib import Path

import datasets
import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from src.analysis.causal import run_layer_pair_evaluation, test_linear_approx
from src.analysis.plots import (
    PlotConfig,
    plot_all_features_above_threshold_results,
    plot_binned_features_results,
    plot_feature_activation_rate_histogram,
    plot_layerwise_filtering,
)
from src.analysis.stats import compute_feature_statistics
from src.analysis.utils import (
    bin_features_by_layer,
    chunk_and_tokenize,
    create_random_sae_weights,
    filter_inactive_features,
    load_base_model,
    load_sae_from_ckpt,
)
from src.sae.logger import get_logger

logger = get_logger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@hydra.main(
    version_base=None,
    config_name="analysis",
    config_path=str(Path(__file__).parent.parent.parent / "config"),
)
def main(cfg: DictConfig):
    """Run analysis experiment for an SAE checkpoint."""
    set_seed(cfg.seed)
    checkpoint_dir = Path(cfg.sae.checkpoint_dir)
    plot_dir = checkpoint_dir / "plots"
    plot_dir.mkdir(exist_ok=True, parents=True)
    debug = cfg.debug
    model, config, tokenizer = load_base_model(cfg.sae.model_name, device=cfg.device)
    num_layers = config.num_hidden_layers

    if debug:
        logger.warning("DEBUG MODE ENABLED: Using randomly initialized SAE weights.")
        sae_weights = create_random_sae_weights(
            cfg.sae.num_latents,
            config.hidden_size * config.num_hidden_layers,
            device=cfg.device,
        )
    else:
        logger.info(f"Loading SAE weights and statistics from {checkpoint_dir}")
        sae_weights = load_sae_from_ckpt(str(checkpoint_dir))

    # Load and preprocess the RedPajama dataset as in the notebook
    logger.info("Loading RedPajama dataset and preparing test split...")
    dataset = datasets.load_dataset(
        "togethercomputer/RedPajama-Data-1T-Sample",
        split="train",
    )
    dataset = (
        dataset.train_test_split(test_size=cfg.test_size, seed=cfg.seed)  # type: ignore
        .get("test")
        .select(range(cfg.global_max_ds_size))  # type: ignore
    )
    logger.info(
        f"Tokenizing and chunking dataset with max_seq_len={cfg.sae.seq_len} ..."
    )
    tokenized = chunk_and_tokenize(dataset, tokenizer, max_seq_len=cfg.sae.seq_len)

    if debug:
        logger.info("Running test_linear_approx in debug mode...")
        test_linear_approx(
            model,  # type: ignore
            tokenized,
            sae_weights.feature_encoder_weights,
            sae_weights.feature_encoder_bias,
            sae_weights.feature_decoder_weights,
            j=0,
            k=1,
            lam=1e-2,
            feature_idx=10,
            test_num_batches=2,
        )

    # Compute statistics for the full dataset
    stats = compute_feature_statistics(
        model,
        tokenized,  # full dataset
        sae_weights.feature_encoder_weights,
        sae_weights.feature_encoder_bias,
        seq_len=cfg.sae.seq_len,
        batch_size=cfg.stats_batch_size,
    )

    # Filter inactive features according to config
    sae_weights, stats = filter_inactive_features(
        stats,
        sae_weights,
        strategy=cfg.sae.filtering_strategy,
        acc_features_threshold=cfg.sae.filtering_threshold,
    )

    # Compute common statistics
    encoder_weights = sae_weights.feature_encoder_weights
    encoder_bias = sae_weights.feature_encoder_bias
    decoder_weights = sae_weights.feature_decoder_weights

    # Bin features by layer
    logger.info("Binning features by layer...")
    binned_enc_features, binned_dec_features = bin_features_by_layer(
        encoder_weights, decoder_weights, num_layers
    )

    # Plot feature activation rate histogram
    logger.info("Plotting feature activation rate histogram...")
    plot_cfg_hist = PlotConfig(
        plot_dir=str(plot_dir), plot_name="feature_activation_rates_histogram"
    )
    plot_feature_activation_rate_histogram(
        stats,
        plot_cfg=plot_cfg_hist,
    )

    # Plot layerwise filtering
    logger.info("Plotting layerwise filtering heatmaps...")
    plot_cfg_layerwise = PlotConfig(
        plot_dir=str(plot_dir), plot_name="layerwise_filtering"
    )
    plot_layerwise_filtering(sae_weights, num_layers, stats, plot_cfg_layerwise)

    # Marginalization config flag
    marginalize_across_sequences = cfg.marginalize_across_sequences
    apply_to_all_tokens = cfg.apply_to_all_tokens

    # Run causal analysis: all_features_above_threshold
    logger.info("Running causal analysis: all_features_above_threshold...")
    results_all = run_layer_pair_evaluation(
        model,  # type: ignore
        tokenized,
        feature_encoder_weights=encoder_weights,
        feature_encoder_bias=encoder_bias,
        feature_decoder_weights=decoder_weights,
        num_layers=num_layers,
        marginalization_mode="all_features_above_threshold",
        marginalize_across_sequences=marginalize_across_sequences,
        apply_to_all_tokens=apply_to_all_tokens,
    )
    plot_all_features_above_threshold_results(results_all, num_layers, str(plot_dir))

    # Run causal analysis: binned_features
    logger.info("Running causal analysis: binned_features...")
    results_binned = run_layer_pair_evaluation(
        model,  # type: ignore
        tokenized,
        feature_encoder_weights=encoder_weights,
        feature_encoder_bias=encoder_bias,
        feature_decoder_weights=decoder_weights,
        num_layers=num_layers,
        binned_enc_features=binned_enc_features,
        marginalization_mode="binned_features",
        marginalize_across_sequences=marginalize_across_sequences,
        apply_to_all_tokens=apply_to_all_tokens,
    )
    plot_binned_features_results(results_binned, num_layers, str(plot_dir))

    logger.info("Analysis and plotting complete. Plots saved to %s", plot_dir)


if __name__ == "__main__":
    main()
