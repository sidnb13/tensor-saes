import os
import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.colors import LogNorm
from transformers import AutoConfig

from src.analysis.stats import GlobalFeatureStatistics


@dataclass
class PlotConfig:
    plot_dir: str
    plot_name: str


def plot_activation_rate_heatmap(
    tokenwise_feature_activation_rate,
    max_features=1000,
    min_rate=1e-6,
    plot_cfg: PlotConfig = PlotConfig(
        plot_dir="plots", plot_name="activation_rate_heatmap_log"
    ),
):
    """
    Plots the activation rate heatmap for the given tokenwise feature activation rate.
    """
    # Convert to numpy and transpose to have features on y-axis and token positions on x-axis
    data = tokenwise_feature_activation_rate.cpu().numpy().T

    # If there are too many features, sample a subset
    if data.shape[0] > max_features:
        indices = np.random.choice(data.shape[0], max_features, replace=False)
        data = data[indices]

    # Set very small values to min_rate to avoid log(0)
    data = np.maximum(data, min_rate)

    # Create the plot
    plt.figure(figsize=(20, 10))

    # Use LogNorm for the color scaling
    sns.heatmap(
        data,
        cmap="viridis",
        norm=LogNorm(vmin=min_rate, vmax=1),
        cbar_kws={"label": "Activation Rate (log scale)"},
    )

    plt.title("Feature Activation Rates by Token Position (Log Scale)")
    plt.xlabel("Token Position")
    plt.ylabel("Feature Index")

    # Save the plot
    plt.savefig(
        os.path.join(plot_cfg.plot_dir, f"{plot_cfg.plot_name}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def plot_sequencewise_activation_rate_heatmap(
    sequencewise_feature_activation_rate,
    max_features=1000,
    min_rate=1e-6,
    max_sequences=1000,
    plot_cfg: PlotConfig = PlotConfig(
        plot_dir="plots", plot_name="sequencewise_activation_rate_heatmap_log"
    ),
):
    """
    Plots the sequencewise activation rate heatmap for the given sequencewise feature activation rate.
    """
    # Convert to numpy and transpose to have features on y-axis and sequences on x-axis
    data = sequencewise_feature_activation_rate.cpu().numpy().T

    # If there are too many features, sample a subset
    if data.shape[0] > max_features:
        indices = np.random.choice(data.shape[0], max_features, replace=False)
        data = data[indices]

    # If there are too many sequences, sample a subset
    if data.shape[1] > max_sequences:
        indices = np.random.choice(data.shape[1], max_sequences, replace=False)
        data = data[:, indices]

    # Sort sequences by their mean activation rate
    sequence_mean_rates = np.mean(data, axis=0)
    sorted_indices = np.argsort(sequence_mean_rates)[::-1]
    data = data[:, sorted_indices]

    # Set very small values to min_rate to avoid log(0)
    data = np.maximum(data, min_rate)

    # Create the plot
    plt.figure(figsize=(20, 10))

    # Use LogNorm for the color scaling
    sns.heatmap(
        data,
        cmap="viridis",
        norm=LogNorm(vmin=min_rate, vmax=1),
        cbar_kws={"label": "Activation Rate (log scale)"},
    )

    plt.title("Feature Activation Rates by Sequence (Log Scale)")
    plt.xlabel("Sequence Index (sorted by mean activation rate)")
    plt.ylabel("Feature Index")

    # Save the plot
    plt.savefig(
        os.path.join(plot_cfg.plot_dir, f"{plot_cfg.plot_name}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def plot_activation_rate_statistics(
    stats,
    K=4,
    plot_cfg: PlotConfig = PlotConfig(
        plot_dir="plots", plot_name="activation_rate_statistics_random_samples"
    ),
):
    """
    Plots the activation rate statistics for the given stats.
    """

    # Create a figure with K rows and 2 columns
    fig, axes = plt.subplots(K, 2, figsize=(20, 5 * K))
    fig.suptitle("Feature Activation Rate Distributions", fontsize=16, y=0.95)

    # Define colors for each type of plot
    sequence_color = "cornflowerblue"
    token_color = "lightcoral"

    # Select K random sequence indices and token positions
    n_sequences = stats.sequencewise_feature_activation_rate.shape[0]
    n_tokens = stats.tokenwise_feature_activation_rate.shape[0]

    random_sequences = torch.randperm(n_sequences)[:K]
    random_positions = torch.randperm(n_tokens)[:K]

    for i in range(K):
        # Sequence-wise activation rate histogram (left column)
        seq_idx = random_sequences[i].item()
        sequence_rates = (
            stats.sequencewise_feature_activation_rate[seq_idx].cpu().numpy()
        )

        axes[i, 0].hist(
            sequence_rates, bins=50, edgecolor="black", color=sequence_color, alpha=0.7
        )
        axes[i, 0].set_title(f"Example {seq_idx} Feature Activation Distribution")
        axes[i, 0].set_xlabel("Activation Rate")
        axes[i, 0].set_ylabel("Frequency")
        axes[i, 0].set_yscale("log")
        axes[i, 0].grid(True, which="both", ls="--", alpha=0.5)

        # Add statistics as text
        stats_text = f"Active: {(sequence_rates > 0).sum()}"
        axes[i, 0].text(
            0.95,
            0.95,
            stats_text,
            transform=axes[i, 0].transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Token-position-wise activation rate histogram (right column)
        pos_idx = random_positions[i].item()
        token_rates = stats.tokenwise_feature_activation_rate[pos_idx].cpu().numpy()

        axes[i, 1].hist(
            token_rates, bins=50, edgecolor="black", color=token_color, alpha=0.7
        )
        axes[i, 1].set_title(
            f"Token Position {pos_idx} Feature Activation Distribution"
        )
        axes[i, 1].set_xlabel("Activation Rate")
        axes[i, 1].set_ylabel("Frequency")
        axes[i, 1].set_yscale("log")
        axes[i, 1].grid(True, which="both", ls="--", alpha=0.5)

        # Add statistics as text
        stats_text = f"Active: {(token_rates > 0).sum()}"
        axes[i, 1].text(
            0.95,
            0.95,
            stats_text,
            transform=axes[i, 1].transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    # Adjust the layout to make room for the title
    plt.subplots_adjust(top=0.93)  # This creates space between the title and the plots
    plt.savefig(
        os.path.join(plot_cfg.plot_dir, f"{plot_cfg.plot_name}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


def plot_feature_activation_rate_histogram(
    stats: GlobalFeatureStatistics,
    plot_cfg: PlotConfig = PlotConfig(
        plot_dir="plots", plot_name="feature_activation_rates_histogram"
    ),
):
    """
    Plots the feature activation rate histogram for the given stats.
    """
    plt.figure(figsize=(12, 6))
    plt.hist(stats.feature_activation_rate.cpu().numpy(), bins=50, edgecolor="black")
    plt.title("Histogram of Feature Activation Rates")
    plt.xlabel("Activation Rate")
    plt.ylabel("Frequency")
    plt.yscale("log")  # Use log scale for y-axis to better visualize the distribution
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(
        os.path.join(plot_cfg.plot_dir, f"{plot_cfg.plot_name}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()


def plot_token_position_activation_histograms(
    tokenwise_feature_activation_rate,
    N=9,
    seq_len=64,
    exclude_first_token_position=0,
    plot_cfg: PlotConfig = PlotConfig(
        plot_dir="plots", plot_name="token_position_activation_histograms_N9"
    ),
):
    # Randomly sample N token positions
    token_positions = random.sample(range(seq_len - exclude_first_token_position), N)
    token_positions.sort()  # Sort for better visualization

    # Calculate number of rows and columns for the grid
    n_rows = int(np.ceil(np.sqrt(N)))
    n_cols = int(np.ceil(N / n_rows))

    # Create the main figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    fig.suptitle(
        "Feature Activation Rate Distributions for Random Token Positions", fontsize=16
    )

    # Flatten axes array for easy iteration
    axes = axes.flatten() if N > 1 else [axes]

    for i, position in enumerate(token_positions):
        ax = axes[i]

        # Get activation rates for this token position
        activation_rates = tokenwise_feature_activation_rate[position].cpu().numpy()

        # Create histogram
        ax.hist(activation_rates, bins=30, edgecolor="black")

        ax.set_title(f"Token Position: {position}")
        ax.set_xlabel("Activation Rate")
        ax.set_ylabel("Frequency")
        ax.set_yscale("log")  # Use log scale for y-axis

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()

    # Save the plot to PLOT_DIR
    plt.savefig(
        os.path.join(plot_cfg.plot_dir, f"{plot_cfg.plot_name}.png"),
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()


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
    Plots the feature cosine similarity heatmap for the given stats.
    """
    # Select a top feature
    top_feature_index = torch.kthvalue(stats.feature_activation_rate, k=k, dim=0)[1]
    dec_feat_vectors = feature_decoder_weights[top_feature_index].reshape(
        model_config.num_hidden_layers, model_config.hidden_size  # type: ignore
    )

    # Compute cosine similarity between layers
    cos_sim = torch.nn.functional.cosine_similarity(
        dec_feat_vectors.unsqueeze(1),  # Shape: [num_layers, 1, hidden_size]
        dec_feat_vectors.unsqueeze(0),  # Shape: [1, num_layers, hidden_size]
        dim=2,  # Compute similarity along the hidden_size dimension
    )
    # \text{cos_sim} shape: [num_layers, num_layers]

    # Create a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cos_sim.cpu().numpy(), cmap="coolwarm", vmin=-1, vmax=1, center=0)
    plt.title(f"Cosine Similarity of Decoder Vectors (Top {k} Feature)")
    plt.xlabel("Vector Index")
    plt.ylabel("Vector Index")
    plt.tight_layout()
    plt.savefig(
        os.path.join(plot_cfg.plot_dir, f"{plot_cfg.plot_name}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    # Print some statistics about the cosine similarities
    print(f"Min cosine similarity: {cos_sim.min().item():.4f}")
    print(f"Max cosine similarity: {cos_sim.max().item():.4f}")
    print(f"Mean cosine similarity: {cos_sim.mean().item():.4f}")
    print(f"Median cosine similarity: {cos_sim.median().item():.4f}")
