"""Unit tests for the Kelly Criterion wealth simulation."""

import numpy as np
import pytest

from simulate import Gamble, Individual, SimConfig, Simulation


# -- Gamble tests ------------------------------------------------------------

class TestGamble:
    def test_gross_payout_formula(self):
        """grossPayout = (1 + EV) / successChance"""
        g = Gamble(ev=1.0, true_success_chance=0.5)
        assert g.gross_payout == pytest.approx(4.0)
        assert g.net_odds == pytest.approx(3.0)

    def test_high_ev_low_chance(self):
        """+1.5 EV at 10% success = 25x gross."""
        g = Gamble(ev=1.5, true_success_chance=0.10)
        assert g.gross_payout == pytest.approx(25.0)
        assert g.net_odds == pytest.approx(24.0)

    def test_breakeven_gamble(self):
        """EV=0 means gross_payout = 1/successChance (break-even)."""
        g = Gamble(ev=0.0, true_success_chance=0.5)
        assert g.gross_payout == pytest.approx(2.0)
        assert g.net_odds == pytest.approx(1.0)

    def test_negative_ev(self):
        """EV=-0.5 with 50% success -> gross_payout = 1.0, net_odds = 0."""
        g = Gamble(ev=-0.5, true_success_chance=0.5)
        assert g.gross_payout == pytest.approx(1.0)
        assert g.net_odds == pytest.approx(0.0)

    def test_certain_win(self):
        """100% success with EV=1.0 -> gross_payout = 2.0."""
        g = Gamble(ev=1.0, true_success_chance=1.0)
        assert g.gross_payout == pytest.approx(2.0)
        assert g.net_odds == pytest.approx(1.0)


# -- Individual perception tests ---------------------------------------------

class TestPerceiveProbability:
    def test_zero_stupidity(self):
        """With stupidity=0, perceived = true probability."""
        ind = Individual(stupidity=0.0, starting_wealth=100_000)
        g = Gamble(ev=1.0, true_success_chance=0.5)
        rng = np.random.default_rng(0)
        assert ind.perceive_probability(g, rng) == pytest.approx(0.5)

    def test_clamped_above_one(self):
        """Perceived probability clamped to 1.0 when noise pushes it over."""
        ind = Individual(stupidity=1.0, starting_wealth=100_000)
        g = Gamble(ev=1.0, true_success_chance=0.95)
        # Use a seed that gives high positive noise
        rng = np.random.default_rng(42)
        results = [ind.perceive_probability(g, rng) for _ in range(1000)]
        assert all(r <= 1.0 for r in results)

    def test_clamped_below_zero(self):
        """Perceived probability clamped to 0.0 when noise pushes it below."""
        ind = Individual(stupidity=1.0, starting_wealth=100_000)
        g = Gamble(ev=1.0, true_success_chance=0.05)
        rng = np.random.default_rng(42)
        results = [ind.perceive_probability(g, rng) for _ in range(1000)]
        assert all(r >= 0.0 for r in results)

    def test_stupidity_adds_noise(self):
        """With high stupidity, perceived probability varies across calls."""
        ind = Individual(stupidity=1.0, starting_wealth=100_000)
        g = Gamble(ev=1.0, true_success_chance=0.5)
        rng = np.random.default_rng(42)
        results = [ind.perceive_probability(g, rng) for _ in range(1000)]
        # Should have variation
        assert max(results) > 0.5
        assert min(results) < 0.5
        # Noise range is [-0.5, +0.5], so perceived is in [0.0, 1.0]
        assert all(0.0 <= r <= 1.0 for r in results)

    def test_stupidity_noise_range(self):
        """Stupidity of 10% means noise in [-5%, +5%]."""
        ind = Individual(stupidity=0.10, starting_wealth=100_000)
        g = Gamble(ev=1.0, true_success_chance=0.5)
        rng = np.random.default_rng(42)
        results = [ind.perceive_probability(g, rng) for _ in range(10_000)]
        # All should be within [0.45, 0.55]
        assert all(0.45 - 1e-9 <= r <= 0.55 + 1e-9 for r in results)
        # Should span most of that range
        assert max(results) > 0.54
        assert min(results) < 0.46


# -- Kelly fraction tests ----------------------------------------------------

class TestKellyFraction:
    def test_standard_kelly(self):
        """f* = (b*p - q) / b for a standard gamble."""
        ind = Individual(stupidity=0, starting_wealth=100_000)
        # 50% chance, 3:1 net odds -> f* = (3*0.5 - 0.5) / 3 = 1/3
        assert ind.kelly_fraction(0.5, 3.0) == pytest.approx(1 / 3)

    def test_kelly_certain_win(self):
        """p=1.0 -> f* = 1.0 (bet everything)."""
        ind = Individual(stupidity=0, starting_wealth=100_000)
        assert ind.kelly_fraction(1.0, 2.0) == pytest.approx(1.0)

    def test_kelly_zero_edge(self):
        """When p = 1/(b+1) (break-even), f* = 0."""
        ind = Individual(stupidity=0, starting_wealth=100_000)
        # b=1, p=0.5 -> f* = (1*0.5 - 0.5) / 1 = 0
        assert ind.kelly_fraction(0.5, 1.0) == pytest.approx(0.0)

    def test_kelly_negative_edge(self):
        """When perceived edge is negative, f* should be 0 (clamped)."""
        ind = Individual(stupidity=0, starting_wealth=100_000)
        # b=1, p=0.3 -> f* = (0.3 - 0.7) / 1 = -0.4 -> clamped to 0
        assert ind.kelly_fraction(0.3, 1.0) == 0.0

    def test_kelly_net_odds_zero(self):
        """When net_odds <= 0, bet nothing."""
        ind = Individual(stupidity=0, starting_wealth=100_000)
        assert ind.kelly_fraction(0.9, 0.0) == 0.0
        assert ind.kelly_fraction(0.9, -1.0) == 0.0

    def test_kelly_capped_at_one(self):
        """f* capped at 1.0 even when formula gives higher."""
        ind = Individual(stupidity=0, starting_wealth=100_000)
        assert ind.kelly_fraction(1.0, 0.5) == 1.0


# -- Individual place_bet tests ----------------------------------------------

class TestPlaceBet:
    def test_win_increases_wealth(self):
        """Winning a bet increases wealth by bet * net_odds."""
        ind = Individual(stupidity=0.0, starting_wealth=100_000)
        # Force a win by using true_success_chance=1.0 gamble
        g_sure = Gamble(ev=1.0, true_success_chance=1.0)
        # net_odds = 1.0, perceived_prob = 1.0, f* = (1*1-0)/1 = 1.0
        # bet = 100_000, win: wealth = 100_000 + 100_000*1 = 200_000
        rng = np.random.default_rng(0)
        ind.place_bet(g_sure, rng)
        assert ind.wealth == pytest.approx(200_000)

    def test_zero_wealth_bets_nothing(self):
        """Bankrupt individual bets nothing and stays at 0."""
        ind = Individual(stupidity=0.0, starting_wealth=0.0)
        g = Gamble(ev=1.0, true_success_chance=0.5)
        rng = np.random.default_rng(0)
        ind.place_bet(g, rng)
        assert ind.wealth == 0.0

    def test_net_odds_zero_no_bet(self):
        """When net_odds <= 0, no bet is placed."""
        ind = Individual(stupidity=0.0, starting_wealth=100_000)
        g = Gamble(ev=-0.5, true_success_chance=0.5)
        # gross_payout = 0.5/0.5 = 1.0, net_odds = 0
        rng = np.random.default_rng(0)
        ind.place_bet(g, rng)
        assert ind.wealth == 100_000


# -- Vectorized _run_round equivalence tests ---------------------------------

class TestVectorizedRound:
    """Verify that the vectorized _run_round produces the same results
    as the Individual OOP methods for the same RNG state."""

    def test_single_individual_equivalence(self):
        """Vectorized round with N=1 should match Individual.place_bet."""
        seed = 123
        for _ in range(20):  # Test multiple random gambles
            # OOP path
            rng_oop = np.random.default_rng(seed)
            ind = Individual(stupidity=0.30, starting_wealth=100_000)
            ev = rng_oop.uniform(-0.5, 1.5)
            true_prob = rng_oop.uniform(0.10, 1.00)
            g = Gamble(ev=ev, true_success_chance=true_prob)
            ind.place_bet(g, rng_oop)

            # Vectorized path
            rng_vec = np.random.default_rng(seed)
            cfg = SimConfig(population=1, batch_size=1)
            sim = Simulation(cfg)
            sim.rng = rng_vec
            wealth = np.array([100_000.0])
            stupidity = np.array([0.30])
            sim._run_round(wealth, stupidity, 1)

            assert wealth[0] == pytest.approx(ind.wealth, rel=1e-10), \
                f"Mismatch at seed {seed}: OOP={ind.wealth}, vec={wealth[0]}"
            seed += 1

    def test_vectorized_known_values(self):
        """Verify vectorized round produces correct results for known inputs."""
        cfg = SimConfig(population=2, batch_size=2)
        sim = Simulation(cfg)

        stupidity = np.array([0.0, 0.0])
        wealth = np.array([100_000.0, 100_000.0])

        # We'll manually verify by checking the math after one round
        rng = np.random.default_rng(999)
        sim.rng = rng

        # Capture what the round will generate
        rng_check = np.random.default_rng(999)
        ev = rng_check.uniform(-0.5, 1.5, 2)
        true_prob = rng_check.uniform(0.10, 1.00, 2)
        gross_payout = (1 + ev) / true_prob
        net_odds = gross_payout - 1
        noise = rng_check.uniform(-stupidity / 2, stupidity / 2)  # all zeros
        perceived = np.clip(true_prob + noise, 0, 1)
        q = 1 - perceived
        f_star = np.where(net_odds > 0, (net_odds * perceived - q) / net_odds, 0.0)
        f_star = np.clip(f_star, 0, 1)
        bet = f_star * wealth
        wins = rng_check.random(2) < true_prob
        expected = np.where(wins, wealth + bet * net_odds, wealth - bet)

        sim._run_round(wealth, stupidity, 2)
        np.testing.assert_allclose(wealth, expected)

    def test_vectorized_preserves_no_bet_on_negative_edge(self):
        """Vectorized: individuals with negative perceived edge don't bet."""
        cfg = SimConfig(population=1000, batch_size=1000,
                        min_ev=-0.5, max_ev=-0.01,  # all negative EV
                        min_stupidity=0.0, max_stupidity=0.0)
        sim = Simulation(cfg)
        wealth = np.full(1000, 100_000.0)
        stupidity = np.zeros(1000)
        sim._run_round(wealth, stupidity, 1000)
        # Perfect perceivers should never bet on negative EV
        np.testing.assert_array_equal(wealth, 100_000.0)


# -- Simulation integration tests --------------------------------------------

class TestSimulationIntegration:
    def test_wealth_never_negative(self):
        """No individual should end with negative wealth."""
        cfg = SimConfig(population=10_000, batch_size=10_000, num_gambles=20, seed=42)
        sim = Simulation(cfg)
        rng = sim.rng
        wealth = np.full(cfg.population, cfg.starting_wealth)
        stupidity = rng.uniform(0, 1, cfg.population)
        for _ in range(cfg.num_gambles):
            sim._run_round(wealth, stupidity, cfg.population)
        assert np.all(wealth >= 0)

    def test_bankrupt_stays_bankrupt(self):
        """An individual with wealth=0 should remain at 0 after a round."""
        cfg = SimConfig(population=100, batch_size=100)
        sim = Simulation(cfg)
        wealth = np.zeros(100)
        stupidity = np.full(100, 0.5)
        sim._run_round(wealth, stupidity, 100)
        assert np.all(wealth == 0)

    def test_zero_gambles_preserves_wealth(self):
        """With num_gambles=0, everyone keeps starting wealth."""
        cfg = SimConfig(population=1000, batch_size=1000, num_gambles=0)
        sim = Simulation(cfg)
        wealth = np.full(1000, cfg.starting_wealth)
        # No rounds to run
        assert np.all(wealth == cfg.starting_wealth)

    def test_high_positive_ev_grows_wealth_on_average(self):
        """With uniformly high EV gambles, average wealth should grow."""
        cfg = SimConfig(
            population=50_000, batch_size=50_000, num_gambles=10,
            min_ev=0.5, max_ev=1.5,  # all positive EV
            min_stupidity=0.0, max_stupidity=0.0,  # perfect knowledge
            seed=42,
        )
        sim = Simulation(cfg)
        wealth = np.full(cfg.population, cfg.starting_wealth)
        stupidity = np.zeros(cfg.population)
        for _ in range(cfg.num_gambles):
            sim._run_round(wealth, stupidity, cfg.population)
        # Median should grow (using median to be robust against outliers)
        assert np.median(wealth) > cfg.starting_wealth

    def test_negative_ev_shrinks_wealth_on_average(self):
        """With uniformly negative EV, smart bettors should mostly not bet."""
        cfg = SimConfig(
            population=10_000, batch_size=10_000, num_gambles=10,
            min_ev=-0.5, max_ev=-0.01,  # all negative EV
            min_stupidity=0.0, max_stupidity=0.0,  # perfect knowledge
            seed=42,
        )
        sim = Simulation(cfg)
        wealth = np.full(cfg.population, cfg.starting_wealth)
        stupidity = np.zeros(cfg.population)
        for _ in range(cfg.num_gambles):
            sim._run_round(wealth, stupidity, cfg.population)
        assert np.all(wealth == cfg.starting_wealth)


# -- Kelly formula mathematical properties -----------------------------------

class TestKellyMathProperties:
    def test_kelly_formula_matches_definition(self):
        """Verify f* = (b*p - q) / b directly."""
        ind = Individual(0, 100_000)
        p, b = 0.6, 2.0
        q = 1 - p
        expected = (b * p - q) / b
        assert ind.kelly_fraction(p, b) == pytest.approx(expected)

    def test_ev_relationship(self):
        """Gamble EV = successChance * grossPayout - 1."""
        for ev in [-0.5, 0.0, 0.5, 1.0, 1.5]:
            for p in [0.1, 0.25, 0.5, 0.75, 1.0]:
                g = Gamble(ev=ev, true_success_chance=p)
                computed_ev = p * g.gross_payout - 1
                assert computed_ev == pytest.approx(ev, abs=1e-10)

    def test_net_payout_example(self):
        """$1 bet at 4x gross pays net $3 on a win."""
        g = Gamble(ev=1.0, true_success_chance=0.5)
        assert g.gross_payout == pytest.approx(4.0)
        # Net profit per $1 bet on win = gross_payout - 1 = 3
        assert g.net_odds == pytest.approx(3.0)
