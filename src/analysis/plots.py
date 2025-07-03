import os
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from plotnine import (
    aes,
    element_text,
    geom_histogram,
    geom_text,
    geom_tile,
    ggplot,
    theme,
)
from plotnine.ggplot import save_as_pdf_pages
from plotnine.labels import labs
from plotnine.scales import scale_fill_gradient, scale_y_log10
from plotnine.themes import theme_minimal
from transformers import AutoConfig

from src.analysis.stats import GlobalFeatureStatistics
from src.analysis.utils import (
    SaeWeights,
    calculate_layer_norms,
    filter_features_by_layer,
)


@dataclass
class PlotConfig:
    plot_dir: str
    plot_name: str
    activation_threshold: float = 0.0
    norm_threshold: float = 0.0


def plot_feature_activation_rate_histogram(
    stats: GlobalFeatureStatistics,
    plot_cfg: PlotConfig = PlotConfig(
        plot_dir="plots", plot_name="feature_activation_rates_histogram"
    ),
):
    """
    Plots the feature activation rate histogram for the given stats using plotnine (ggplot).
    """

    df = pd.DataFrame({"Activation Rate": stats.feature_activation_rate.cpu().numpy()})
    p = (
        ggplot(df, aes(x="Activation Rate"))
        + geom_histogram(bins=50, fill="#3182bd", color="black")
        + scale_y_log10()
        + labs(
            title="Histogram of Feature Activation Rates",
            x="Activation Rate",
            y="Frequency",
        )
        + theme_minimal()
        + theme(title=element_text(size=14))
    )
    os.makedirs(plot_cfg.plot_dir, exist_ok=True)
    p.save(os.path.join(plot_cfg.plot_dir, f"{plot_cfg.plot_name}.png"), dpi=300, transparent=False)


def plot_feature_cosine_similarity(
    stats: GlobalFeatureStatistics,
    feature_decoder_weights: torch.Tensor,
    model_config: AutoConfig,
    k: int,
    plot_cfg: PlotConfig = PlotConfig(
        plot_dir="plots", plot_name="cosine_similarity_heatmap_top_k_feature"
    ),
):
    """
    Plots the feature cosine similarity heatmap for the given stats using plotnine (ggplot).
    """

    top_feature_index = torch.kthvalue(stats.feature_activation_rate, k=k, dim=0)[1]
    dec_feat_vectors = feature_decoder_weights[top_feature_index].reshape(
        model_config.num_hidden_layers,  # type: ignore
        model_config.hidden_size,  # type: ignore
    )
    cos_sim = torch.nn.functional.cosine_similarity(
        dec_feat_vectors.unsqueeze(1),
        dec_feat_vectors.unsqueeze(0),
        dim=2,
    )
    cos_sim_np = cos_sim.cpu().numpy()
    n = cos_sim_np.shape[0]
    df = pd.DataFrame(
        {
            "Layer1": np.repeat(np.arange(n), n),
            "Layer2": np.tile(np.arange(n), n),
            "Cosine Similarity": cos_sim_np.flatten(),
        }
    )
    p = (
        ggplot(df, aes("Layer1", "Layer2", fill="Cosine Similarity"))
        + geom_tile()
        + scale_fill_gradient(low="#440154", high="#FDE725", limits=(-1, 1))
        + labs(
            title=f"Cosine Similarity of Decoder Vectors (Top {k} Feature)",
            x="Vector Index",
            y="Vector Index",
            fill="Cosine Similarity",
        )
        + theme_minimal()
        + theme(
            figure_size=(10, 8),
            axis_text_x=element_text(rotation=90, hjust=1),
            title=element_text(size=14),
        )
    )
    os.makedirs(plot_cfg.plot_dir, exist_ok=True)
    p.save(os.path.join(plot_cfg.plot_dir, f"{plot_cfg.plot_name}.png"), dpi=300, transparent=False)


def plot_layerwise_filtering(
    sae_weights: SaeWeights,
    num_layers: int,
    stats: GlobalFeatureStatistics,
    plot_cfg: PlotConfig,
):
    """
    For each layer, filter features where that layer has the largest norm (layer-wise filtering),
    then stack the filtered features for all layers horizontally into a single 2D grid plot.
    Do this for both encoder and decoder norms, using plotnine (ggplot).
    Each plot will have y-axis as layer, x-axis as concatenated filtered features (grouped by layer).
    """
    save_folder = plot_cfg.plot_dir
    # Calculate norms: shape (num_layers, num_features)
    encoder_norms = calculate_layer_norms(
        sae_weights.feature_encoder_weights, num_layers
    )
    decoder_norms = calculate_layer_norms(
        sae_weights.feature_decoder_weights, num_layers
    )

    # Sort features by activation rate (descending)
    sorted_indices = torch.argsort(stats.feature_activation_rate, descending=True)
    encoder_norms = encoder_norms[:, sorted_indices]
    decoder_norms = decoder_norms[:, sorted_indices]

    # For each layer, filter features where that layer has the largest norm
    encoder_blocks = []
    decoder_blocks = []
    feature_labels = []
    feature_layer_labels = []
    running_idx = 0

    for i in range(num_layers):
        enc_block, dec_block, layer_mask = filter_features_by_layer(
            encoder_norms, decoder_norms, i
        )
        # Only keep features with nonzero count
        if enc_block.shape[1] == 0:
            continue
        # Optionally, sort by activation rate within this block
        block_activation = stats.feature_activation_rate[layer_mask]
        block_sort_idx = torch.argsort(block_activation, descending=True)
        enc_block = enc_block[:, block_sort_idx]
        dec_block = dec_block[:, block_sort_idx]
        block_activation = block_activation[block_sort_idx]

        encoder_blocks.append(enc_block)
        decoder_blocks.append(dec_block)
        # For x-axis labeling
        feature_labels.extend(
            list(range(running_idx, running_idx + enc_block.shape[1]))
        )
        feature_layer_labels.extend([i] * enc_block.shape[1])
        running_idx += enc_block.shape[1]

    # Stack horizontally: shape (num_layers, total_filtered_features)
    if len(encoder_blocks) == 0 or len(decoder_blocks) == 0:
        print("No features passed the filtering. No plot will be generated.")
        return None, None

    encoder_grid = torch.cat(encoder_blocks, dim=1).cpu().numpy()
    decoder_grid = torch.cat(decoder_blocks, dim=1).cpu().numpy()
    total_features = encoder_grid.shape[1]
    n_layers = encoder_grid.shape[0]

    # Prepare DataFrame for encoder
    encoder_df = pd.DataFrame(
        {
            "Layer": np.repeat(np.arange(n_layers), total_features),
            "Feature": np.tile(np.arange(total_features), n_layers),
            "Norm": encoder_grid.flatten(),
            "FeatureLayer": np.tile(feature_layer_labels, n_layers),
        }
    )

    # Prepare DataFrame for decoder
    decoder_df = pd.DataFrame(
        {
            "Layer": np.repeat(np.arange(n_layers), total_features),
            "Feature": np.tile(np.arange(total_features), n_layers),
            "Norm": decoder_grid.flatten(),
            "FeatureLayer": np.tile(feature_layer_labels, n_layers),
        }
    )

    # Plot encoder norms heatmap
    encoder_plot = (
        ggplot(encoder_df, aes("Feature", "Layer", fill="Norm"))
        + geom_tile()
        + scale_fill_gradient(low="#440154", high="#FDE725")
        + labs(
            title="Encoder Norms (Layer-wise Filtered, Features Stacked by Layer)",
            x="Feature (concatenated by layer)",
            y="Layer",
            fill="Norm",
        )
        + theme_minimal()
        + theme(
            figure_size=(max(16, total_features // 50), 4),
            axis_text_x=element_text(rotation=90, hjust=1, size=6),
            axis_text_y=element_text(size=10),
            title=element_text(size=14),
        )
    )

    # Plot decoder norms heatmap
    decoder_plot = (
        ggplot(decoder_df, aes("Feature", "Layer", fill="Norm"))
        + geom_tile()
        + scale_fill_gradient(low="#440154", high="#FDE725")
        + labs(
            title="Decoder Norms (Layer-wise Filtered, Features Stacked by Layer)",
            x="Feature (concatenated by layer)",
            y="Layer",
            fill="Norm",
        )
        + theme_minimal()
        + theme(
            figure_size=(max(16, total_features // 50), 4),
            axis_text_x=element_text(rotation=90, hjust=1, size=6),
            axis_text_y=element_text(size=10),
            title=element_text(size=14),
        )
    )

    # Save plots
    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
        encoder_plot.save(
            os.path.join(save_folder, "encoder_norms_layerwise_filtered_heatmap.png"),
            dpi=300,
            transparent=False
        )
        decoder_plot.save(
            os.path.join(save_folder, "decoder_norms_layerwise_filtered_heatmap.png"),
            dpi=300,
            transparent=False
        )


def process_layer(
    encoder_weights,
    decoder_weights,
    encoder_bias,
    layer_index,
    num_layers,
):
    """Process a single layer and return filtered weights and statistics."""
    encoder_norms = calculate_layer_norms(encoder_weights, num_layers)
    decoder_norms = calculate_layer_norms(decoder_weights, num_layers)

    filtered_encoder_norms, filtered_decoder_norms, layer_mask = (
        filter_features_by_layer(encoder_norms, decoder_norms, layer_index)
    )

    if filtered_encoder_norms.numel() == 0 or filtered_decoder_norms.numel() == 0:
        print(f"No features left after filtering layer {layer_index}")
        return {}

    num_features = filtered_encoder_norms.shape[1]

    return {
        "filtered_encoder": encoder_weights[layer_mask, :],
        "filtered_encoder_bias": encoder_bias[layer_mask],
        "filtered_decoder": decoder_weights[layer_mask, :],
        "num_features": num_features,
        "active_features": encoder_norms.shape[1],
        "mean_encoder_norm": encoder_norms.mean().item(),
        "mean_decoder_norm": decoder_norms.mean().item(),
    }


def plot_trigger_layer_heatmaps(
    results,
    num_layers,
    rank_k_feature,
    plot_cfg: PlotConfig = PlotConfig(
        plot_dir="plots", plot_name="layer_heatmaps_top_k"
    ),
):
    os.makedirs(plot_cfg.plot_dir, exist_ok=True)

    metric_keys = [
        "causal_cosines",
        "causality",
        "self_cosine_similarity",
    ]
    std_keys = [
        "causal_cosines_std",
        "causality_std",
        "self_cosine_similarity_std",
    ]
    titles = [
        "Causal Cosine",
        "Causal Strength",
        "Self Cosine Similarity",
    ]

    for layer in range(num_layers):
        if not results[layer]:
            print(f"No results for layer {layer}")
            continue

        # Build DataFrames for each metric
        for idx, (metric, std_metric, title) in enumerate(
            zip(metric_keys, std_keys, titles)
        ):
            data = []
            for (i, j), values in results[layer].items():
                if i < j:  # Only upper triangle
                    mean_value = values[metric]
                    std_value = values[std_metric]
                    data.append(
                        {
                            "Layer_i": i,
                            "Layer_j": j,
                            "Value": mean_value,
                            "Std": std_value,
                        }
                    )
            if not data:
                print(f"No data for {metric} in layer {layer}")
                continue
            df = pd.DataFrame(data)

            # Plot with plotnine
            p = (
                ggplot(df, aes(x="Layer_j", y="Layer_i", fill="Value"))
                + geom_tile()
                + scale_fill_gradient(low="#440154", high="#FDE725")
                + labs(
                    title=f"Layer {layer} - {title}",
                    x="Layer j",
                    y="Layer i",
                    fill=metric.replace("_", " ").title(),
                )
                + theme_minimal()
                + theme(
                    figure_size=(10, 8),
                    axis_text_x=element_text(rotation=90, hjust=1, size=10),
                    axis_text_y=element_text(size=10),
                    title=element_text(size=14),
                )
            )
            # Optionally add text annotations for mean/std
            p = p + geom_text(
                aes(
                    label=df.apply(
                        lambda row: f"{row['Value']:.2f}\n({row['Std']:.2f})", axis=1
                    )
                ),
                size=7,
                color="black",
            )

            plot_filename = f"{plot_cfg.plot_name}_layer_{layer}_{metric}_features.png"
            p.save(os.path.join(plot_cfg.plot_dir, plot_filename), dpi=300, transparent=False)


def plot_layer_pair_results_heatmap(
    results,
    num_layers,
    plot_dir,
    plot_name_prefix,
    metrics=("causality", "causal_cosines", "self_cosine_similarity"),
    vmin=None,
    vmax=None,
):
    """
    Generic heatmap plotter for layer-pair results dict.
    Plots all (i, j) pairs for each metric as a heatmap.
    """
    os.makedirs(plot_dir, exist_ok=True)
    for metric in metrics:
        data = np.full((num_layers, num_layers), np.nan)
        std_data = np.full((num_layers, num_layers), np.nan)
        for i in range(num_layers):
            for j in range(num_layers):
                if i < j and (i, j) in results.get(i, {}):
                    data[i, j] = results[i][(i, j)].get(metric, np.nan)
                    std_data[i, j] = results[i][(i, j)].get(f"{metric}_std", np.nan)
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(data, vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_title(f"{plot_name_prefix}: {metric}")
        ax.set_xlabel("Layer j")
        ax.set_ylabel("Layer i")
        ax.set_xticks(np.arange(num_layers))
        ax.set_yticks(np.arange(num_layers))
        fig.colorbar(im, ax=ax, label=metric)
        # Annotate with mean (std)
        for i in range(num_layers):
            for j in range(num_layers):
                if i < j and not np.isnan(data[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{data[i, j]:.2f}\n({std_data[i, j]:.2f})",
                        ha="center",
                        va="center",
                        color="w"
                        if data[i, j] < 0.5 * (np.nanmax(data) + np.nanmin(data))
                        else "black",
                        fontsize=8,
                    )
        plt.tight_layout()
        plt.savefig(
            os.path.join(plot_dir, f"{plot_name_prefix}_{metric}_heatmap.png"), dpi=300, transparent=False, facecolor='white'
        )
        plt.close(fig)


def plot_all_features_above_threshold_results(
    results, num_layers, plot_dir, plot_name_prefix="all_features_above_threshold"
):
    plot_layer_pair_results_heatmap(results, num_layers, plot_dir, plot_name_prefix)


def plot_feature_index_results(
    results, num_layers, plot_dir, plot_name_prefix="feature_index"
):
    plot_layer_pair_results_heatmap(results, num_layers, plot_dir, plot_name_prefix)


def plot_fixed_i_results(results, num_layers, plot_dir, plot_name_prefix="fixed_i"):
    plot_layer_pair_results_heatmap(results, num_layers, plot_dir, plot_name_prefix)


def plot_binned_features_results(
    results, num_layers, plot_dir, plot_name_prefix="binned_features"
):
    plot_layer_pair_results_heatmap(results, num_layers, plot_dir, plot_name_prefix)
