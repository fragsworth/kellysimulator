"""Kelly Criterion Wealth Simulation.

Simulates a population of individuals making randomized gambling decisions
using the Kelly Criterion, with varying levels of stupidity.
Results are written to disk as .npy files for separate analysis by stats.py.
"""

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator


@dataclass
class SimConfig:
    """All simulation parameters."""
    population: int = 1_000_000
    starting_wealth: float = 100_000.0
    num_gambles: int = 20
    batch_size: int = 1_000_000
    min_stupidity: float = 0.0
    max_stupidity: float = 1.0
    min_ev: float = -0.5
    max_ev: float = 1.5
    min_success: float = 0.10
    max_success: float = 1.00
    seed: int = 42


class Gamble:
    """A single gambling opportunity with binary outcome.

    Each gamble has an EV, true success chance, and derived payout.
    In the vectorized simulation, arrays of these values are used instead
    of individual Gamble objects.
    """
    __slots__ = ('ev', 'true_success_chance', 'gross_payout', 'net_odds')

    def __init__(self, ev: float, true_success_chance: float):
        self.ev = ev
        self.true_success_chance = true_success_chance
        self.gross_payout = (1 + ev) / true_success_chance
        self.net_odds = self.gross_payout - 1


class Individual:
    """A person with a persistent stupidity trait who makes Kelly-optimal bets.

    Each individual has a stupidity value that persists across all gambles.
    In the vectorized simulation, arrays of these values are used instead
    of individual objects.
    """
    __slots__ = ('stupidity', 'wealth')

    def __init__(self, stupidity: float, starting_wealth: float):
        self.stupidity = stupidity
        self.wealth = starting_wealth

    def perceive_probability(self, gamble: Gamble, rng: Generator) -> float:
        """Compute perceived success chance with stupidity noise."""
        noise = rng.uniform(-self.stupidity / 2, self.stupidity / 2)
        perceived = gamble.true_success_chance + noise
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
    """Runs the simulation in batches, writing results to memory-mapped files."""

    def __init__(self, config: SimConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

    def run(self):
        """Run the full simulation, writing results to .npy files."""
        cfg = self.config
        N = cfg.population

        # Pre-allocate memory-mapped output files
        stupidity_mmap = np.lib.format.open_memmap(
            'results_stupidity.npy', mode='w+', dtype=np.float64, shape=(N,))
        wealth_mmap = np.lib.format.open_memmap(
            'results_wealth.npy', mode='w+', dtype=np.float64, shape=(N,))

        num_batches = (N + cfg.batch_size - 1) // cfg.batch_size

        for batch_idx in range(num_batches):
            start = batch_idx * cfg.batch_size
            end = min(start + cfg.batch_size, N)
            batch_n = end - start

            print(f"  Batch {batch_idx + 1}/{num_batches} ({batch_n:,} individuals)...", flush=True)

            # Generate persistent traits for this batch
            stupidity = self.rng.uniform(cfg.min_stupidity, cfg.max_stupidity, batch_n)
            wealth = np.full(batch_n, cfg.starting_wealth)

            # Run all gamble rounds for this batch
            for round_num in range(cfg.num_gambles):
                self._run_round(wealth, stupidity, batch_n)

            # Write results to memory-mapped files
            stupidity_mmap[start:end] = stupidity
            wealth_mmap[start:end] = wealth

        # Flush to disk
        del stupidity_mmap, wealth_mmap
        print("  Results written to disk.")

    def _run_round(self, wealth, stupidity, n):
        """Execute one round of gambling for all individuals in a batch (vectorized)."""
        cfg = self.config
        rng = self.rng

        # Roll gamble parameters
        ev = rng.uniform(cfg.min_ev, cfg.max_ev, n)
        true_prob = rng.uniform(cfg.min_success, cfg.max_success, n)
        gross_payout = (1 + ev) / true_prob
        net_odds = gross_payout - 1  # b in Kelly formula

        # Compute perceived probability
        noise = rng.uniform(-stupidity / 2, stupidity / 2)
        perceived_prob = np.clip(true_prob + noise, 0.0, 1.0)

        # Kelly fraction: f* = (b*p - q) / b
        q = 1.0 - perceived_prob
        f_star = np.where(
            net_odds > 0,
            (net_odds * perceived_prob - q) / net_odds,
            0.0,
        )
        f_star = np.clip(f_star, 0.0, 1.0)

        # Bet and resolve
        bet = f_star * wealth
        wins = rng.random(n) < true_prob
        wealth[:] = np.where(wins, wealth + bet * net_odds, wealth - bet)


def main():
    config = SimConfig()

    print("Starting Kelly Criterion Wealth Simulation")
    print(f"  Population: {config.population:,}")
    print(f"  Gambles per individual: {config.num_gambles}")
    print(f"  Starting wealth: ${config.starting_wealth:,.0f}")
    print(f"  Batch size: {config.batch_size:,}")
    print()

    sim = Simulation(config)
    sim.run()
    print("Simulation complete.")


if __name__ == "__main__":
    main()
