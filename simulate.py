"""Kelly Criterion Wealth Simulation.

Simulates a population of individuals making randomized gambling decisions
using the Kelly Criterion, with varying levels of stupidity and risk inclination.
"""

from dataclasses import dataclass
import math

import numpy as np
from numpy.random import Generator
import matplotlib.pyplot as plt


@dataclass
class SimConfig:
    """All simulation parameters."""
    population: int = 1_000_000
    starting_wealth: float = 100_000.0
    num_gambles: int = 20
    min_stupidity: float = 0.0
    max_stupidity: float = 1.0
    min_belief_bonus: float = -0.20
    max_belief_bonus: float = 0.20
    min_ev: float = -0.5
    max_ev: float = 1.5
    min_success: float = 0.10
    max_success: float = 1.00
    seed: int = 42


class Gamble:
    """A single gambling opportunity with binary outcome."""
    __slots__ = ('ev', 'true_success_chance', 'gross_payout', 'net_odds')

    def __init__(self, ev: float, true_success_chance: float):
        self.ev = ev
        self.true_success_chance = true_success_chance
        self.gross_payout = (1 + ev) / true_success_chance
        self.net_odds = self.gross_payout - 1


class Individual:
    """A person with persistent traits who makes Kelly-optimal bets."""
    __slots__ = ('stupidity', 'belief_bonus', 'wealth')

    def __init__(self, stupidity: float, belief_bonus: float, starting_wealth: float):
        self.stupidity = stupidity
        self.belief_bonus = belief_bonus
        self.wealth = starting_wealth

    def perceive_probability(self, gamble: Gamble, rng: Generator) -> float:
        """Compute perceived success chance with stupidity noise + belief bonus."""
        noise = rng.uniform(-self.stupidity / 2, self.stupidity / 2)
        perceived = gamble.true_success_chance + noise + self.belief_bonus
        return max(0.0, min(1.0, perceived))

    def kelly_fraction(self, perceived_prob: float, net_odds: float) -> float:
        """Compute Kelly bet fraction. Returns 0 if bet is unfavorable."""
        if net_odds <= 0:
            return 0.0
        q = 1.0 - perceived_prob
        f_star = (net_odds * perceived_prob - q) / net_odds
        return max(0.0, min(1.0, f_star))

    def place_bet(self, gamble: Gamble, rng: Generator):
        """Evaluate a gamble: compute bet via Kelly, resolve win/loss, update wealth."""
        perceived_prob = self.perceive_probability(gamble, rng)
        f_star = self.kelly_fraction(perceived_prob, gamble.net_odds)
        bet = f_star * self.wealth
        if bet <= 0:
            return
        won = rng.random() < gamble.true_success_chance
        if won:
            self.wealth += bet * gamble.net_odds
        else:
            self.wealth -= bet


class Simulation:
    """Orchestrates population creation and the simulation loop."""

    def __init__(self, config: SimConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.individuals = self._create_population()

    def _create_population(self) -> list:
        """Create N individuals with random traits."""
        cfg = self.config
        stupidities = self.rng.uniform(cfg.min_stupidity, cfg.max_stupidity, cfg.population)
        bonuses = self.rng.uniform(cfg.min_belief_bonus, cfg.max_belief_bonus, cfg.population)
        return [
            Individual(stupidities[i], bonuses[i], cfg.starting_wealth)
            for i in range(cfg.population)
        ]

    def _create_gamble(self) -> Gamble:
        """Roll a single random gamble."""
        cfg = self.config
        ev = self.rng.uniform(cfg.min_ev, cfg.max_ev)
        success = self.rng.uniform(cfg.min_success, cfg.max_success)
        return Gamble(ev, success)

    def run(self):
        """Run all gamble rounds."""
        cfg = self.config
        for round_num in range(cfg.num_gambles):
            print(f"  Round {round_num + 1}/{cfg.num_gambles}...")
            for individual in self.individuals:
                gamble = self._create_gamble()
                individual.place_bet(gamble, self.rng)


class Reporter:
    """Handles output: top-50 table and binned bar charts."""

    def __init__(self, individuals: list, config: SimConfig):
        self.individuals = individuals
        self.config = config

    def print_top_50(self):
        """Print top 50 wealthiest individuals."""
        sorted_inds = sorted(self.individuals, key=lambda ind: ind.wealth, reverse=True)
        top = sorted_inds[:50]

        print("\n" + "=" * 72)
        print("TOP 50 WEALTHIEST INDIVIDUALS")
        print("=" * 72)
        print(f"{'Rank':>4}  {'Final Wealth':>20}  {'Stupidity':>10}  {'Belief Bonus':>13}")
        print("-" * 72)
        for i, ind in enumerate(top, 1):
            print(f"{i:>4}  ${ind.wealth:>19,.2f}  {ind.stupidity:>9.2%}  {ind.belief_bonus:>+12.2%}")
        print("=" * 72)

    def plot_charts(self):
        """Generate 8 bar charts: 4 for stupidity bins, 4 for belief bonus bins."""
        wealths = np.array([ind.wealth for ind in self.individuals])
        stupidities = np.array([ind.stupidity for ind in self.individuals])
        belief_bonuses = np.array([ind.belief_bonus for ind in self.individuals])

        # Stupidity: 20 bins of 5% width
        stupidity_edges = np.linspace(0.0, 1.0, 21)
        self._plot_binned_charts(
            trait_values=stupidities,
            wealth_values=wealths,
            bin_edges=stupidity_edges,
            trait_name="Stupidity",
            format_pct=True,
            filename="stupidity_charts.png",
        )

        # Belief bonus: 8 bins of 5% width
        belief_edges = np.linspace(-0.20, 0.20, 9)
        self._plot_binned_charts(
            trait_values=belief_bonuses,
            wealth_values=wealths,
            bin_edges=belief_edges,
            trait_name="Risk Inclination (Belief Bonus)",
            format_pct=True,
            filename="risk_inclination_charts.png",
        )

    def _plot_binned_charts(self, trait_values, wealth_values, bin_edges,
                            trait_name, format_pct, filename):
        """Produce a 2x2 figure with 4 bar charts for one binning dimension."""
        bin_indices = np.digitize(trait_values, bin_edges) - 1
        num_bins = len(bin_edges) - 1
        # Clamp edge cases
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)

        # Build bin labels
        labels = []
        for i in range(num_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            if format_pct:
                labels.append(f"{lo:.0%}-{hi:.0%}")
            else:
                labels.append(f"{lo:.2f}-{hi:.2f}")

        # Compute stats per bin
        means = []
        medians = []
        mean_logs = []
        median_logs = []
        bankrupt_counts = []

        for b in range(num_bins):
            mask = bin_indices == b
            bin_wealth = wealth_values[mask]

            if len(bin_wealth) == 0:
                means.append(0)
                medians.append(0)
                mean_logs.append(0)
                median_logs.append(0)
                bankrupt_counts.append(0)
                continue

            means.append(np.mean(bin_wealth))
            medians.append(np.median(bin_wealth))

            positive = bin_wealth[bin_wealth > 0]
            bankrupt_counts.append(len(bin_wealth) - len(positive))
            if len(positive) > 0:
                log_wealth = np.log10(positive)
                mean_logs.append(np.mean(log_wealth))
                median_logs.append(np.median(log_wealth))
            else:
                mean_logs.append(0)
                median_logs.append(0)

        x = np.arange(num_bins)

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f"Wealth Distribution by {trait_name}", fontsize=14, fontweight='bold')

        chart_data = [
            (axes[0, 0], means, f"Mean Final Wealth by {trait_name}"),
            (axes[0, 1], medians, f"Median Final Wealth by {trait_name}"),
            (axes[1, 0], mean_logs, f"Mean log₁₀(Wealth) by {trait_name}"),
            (axes[1, 1], median_logs, f"Median log₁₀(Wealth) by {trait_name}"),
        ]

        for ax, data, title in chart_data:
            ax.bar(x, data, color='steelblue', edgecolor='white', linewidth=0.5)
            ax.set_title(title)
            ax.set_xlabel(trait_name)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
            if "log" in title.lower():
                ax.set_ylabel("log₁₀(Wealth)")
            else:
                ax.set_ylabel("Wealth ($)")
                ax.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  Saved {filename}")

        total_bankrupt = sum(bankrupt_counts)
        if total_bankrupt > 0:
            print(f"  Note: {total_bankrupt:,} bankrupt individuals excluded from log-wealth charts.")


def main():
    config = SimConfig()

    print("Starting Kelly Criterion Wealth Simulation")
    print(f"  Population: {config.population:,}")
    print(f"  Gambles per individual: {config.num_gambles}")
    print(f"  Starting wealth: ${config.starting_wealth:,.0f}")
    print()

    print("Creating population and running simulation...")
    sim = Simulation(config)
    sim.run()
    print("Simulation complete.\n")

    reporter = Reporter(sim.individuals, config)
    reporter.print_top_50()

    print("\nGenerating charts...")
    reporter.plot_charts()
    print("Done.")


if __name__ == "__main__":
    main()
