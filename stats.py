"""Stats and charts for the Kelly Criterion Wealth Simulation.

Reads simulation results from .npy files and produces:
- Top 50 wealthiest individuals table
- 8 bar charts (4 by stupidity, 4 by belief bonus)
"""

import numpy as np
import matplotlib.pyplot as plt


def load_results():
    """Load simulation results as memory-mapped arrays."""
    wealth = np.load('results_wealth.npy', mmap_mode='r')
    stupidity = np.load('results_stupidity.npy', mmap_mode='r')
    belief_bonus = np.load('results_belief_bonus.npy', mmap_mode='r')
    print(f"Loaded {len(wealth):,} individuals.")
    return wealth, stupidity, belief_bonus


def print_top_50(wealth, stupidity, belief_bonus):
    """Print top 50 wealthiest individuals."""
    top_indices = np.argpartition(wealth, -50)[-50:]
    top_indices = top_indices[np.argsort(wealth[top_indices])[::-1]]

    print("\n" + "=" * 72)
    print("TOP 50 WEALTHIEST INDIVIDUALS")
    print("=" * 72)
    print(f"{'Rank':>4}  {'Final Wealth':>20}  {'Stupidity':>10}  {'Belief Bonus':>13}")
    print("-" * 72)
    for i, idx in enumerate(top_indices, 1):
        w = wealth[idx]
        s = stupidity[idx]
        b = belief_bonus[idx]
        print(f"{i:>4}  ${w:>19,.2f}  {s:>9.2%}  {b:>+12.2%}")
    print("=" * 72)


def plot_charts(wealth, stupidity, belief_bonus):
    """Generate 8 bar charts: 4 for stupidity bins, 4 for belief bonus bins."""
    # Stupidity: 20 bins of 5% width
    stupidity_edges = np.linspace(0.0, 1.0, 21)
    _plot_binned_charts(
        trait_values=stupidity,
        wealth_values=wealth,
        bin_edges=stupidity_edges,
        trait_name="Stupidity",
        filename="stupidity_charts.png",
    )

    # Belief bonus: 8 bins of 5% width
    belief_edges = np.linspace(-0.20, 0.20, 9)
    _plot_binned_charts(
        trait_values=belief_bonus,
        wealth_values=wealth,
        bin_edges=belief_edges,
        trait_name="Risk Inclination (Belief Bonus)",
        filename="risk_inclination_charts.png",
    )


def _plot_binned_charts(trait_values, wealth_values, bin_edges, trait_name, filename):
    """Produce a 2x2 figure with 4 bar charts for one binning dimension."""
    num_bins = len(bin_edges) - 1
    bin_indices = np.digitize(trait_values, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    # Build bin labels
    labels = []
    for i in range(num_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        labels.append(f"{lo:.0%}-{hi:.0%}")

    # Compute stats per bin
    means = []
    medians = []
    mean_logs = []
    median_logs = []
    bankrupt_total = 0

    for b in range(num_bins):
        print(f"  Computing stats for {trait_name} bin {b + 1}/{num_bins}...")
        mask = bin_indices == b
        bin_wealth = wealth_values[mask]

        if len(bin_wealth) == 0:
            means.append(0)
            medians.append(0)
            mean_logs.append(0)
            median_logs.append(0)
            continue

        means.append(float(np.mean(bin_wealth)))
        medians.append(float(np.median(bin_wealth)))

        positive = bin_wealth[bin_wealth > 0]
        bankrupt_total += len(bin_wealth) - len(positive)
        if len(positive) > 0:
            log_wealth = np.log10(positive)
            mean_logs.append(float(np.mean(log_wealth)))
            median_logs.append(float(np.median(log_wealth)))
        else:
            mean_logs.append(0)
            median_logs.append(0)

    x = np.arange(num_bins)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Wealth Distribution by {trait_name}", fontsize=14, fontweight='bold')

    chart_data = [
        (axes[0, 0], means, f"Mean Final Wealth by {trait_name}"),
        (axes[0, 1], medians, f"Median Final Wealth by {trait_name}"),
        (axes[1, 0], mean_logs, f"Mean log\u2081\u2080(Wealth) by {trait_name}"),
        (axes[1, 1], median_logs, f"Median log\u2081\u2080(Wealth) by {trait_name}"),
    ]

    for ax, data, title in chart_data:
        ax.bar(x, data, color='steelblue', edgecolor='white', linewidth=0.5)
        ax.set_title(title)
        ax.set_xlabel(trait_name)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        if "log" in title.lower():
            ax.set_ylabel("log\u2081\u2080(Wealth)")
        else:
            ax.set_ylabel("Wealth ($)")
            ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved {filename}")

    if bankrupt_total > 0:
        print(f"  Note: {bankrupt_total:,} bankrupt individuals excluded from log-wealth charts.")


def main():
    wealth, stupidity, belief_bonus = load_results()
    print_top_50(wealth, stupidity, belief_bonus)
    print("\nGenerating charts...")
    plot_charts(wealth, stupidity, belief_bonus)
    print("Done.")


if __name__ == "__main__":
    main()
