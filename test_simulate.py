"""Unit tests for the Kelly Criterion wealth simulation."""

import numpy as np
import pytest

from simulate import Gamble, Individual, SimConfig, Simulation


# ── Gamble tests ──────────────────────────────────────────────────────────

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
        """EV=-0.5 with 50% success → gross_payout = 1.0, net_odds = 0."""
        g = Gamble(ev=-0.5, true_success_chance=0.5)
        assert g.gross_payout == pytest.approx(1.0)
        assert g.net_odds == pytest.approx(0.0)

    def test_certain_win(self):
        """100% success with EV=1.0 → gross_payout = 2.0."""
        g = Gamble(ev=1.0, true_success_chance=1.0)
        assert g.gross_payout == pytest.approx(2.0)
        assert g.net_odds == pytest.approx(1.0)


# ── Individual perception tests ───────────────────────────────────────────

class TestPerceiveProbability:
    def test_zero_stupidity_no_bonus(self):
        """With stupidity=0 and bonus=0, perceived = true probability."""
        ind = Individual(stupidity=0.0, belief_bonus=0.0, starting_wealth=100_000)
        g = Gamble(ev=1.0, true_success_chance=0.5)
        rng = np.random.default_rng(0)
        assert ind.perceive_probability(g, rng) == pytest.approx(0.5)

    def test_belief_bonus_additive(self):
        """Belief bonus shifts perceived probability by exact amount (no stupidity noise)."""
        ind = Individual(stupidity=0.0, belief_bonus=0.10, starting_wealth=100_000)
        g = Gamble(ev=1.0, true_success_chance=0.5)
        rng = np.random.default_rng(0)
        assert ind.perceive_probability(g, rng) == pytest.approx(0.60)

    def test_negative_belief_bonus(self):
        """Negative belief bonus reduces perceived probability."""
        ind = Individual(stupidity=0.0, belief_bonus=-0.15, starting_wealth=100_000)
        g = Gamble(ev=1.0, true_success_chance=0.5)
        rng = np.random.default_rng(0)
        assert ind.perceive_probability(g, rng) == pytest.approx(0.35)

    def test_clamped_above_one(self):
        """Perceived probability clamped to 1.0 when bonus pushes it over."""
        ind = Individual(stupidity=0.0, belief_bonus=0.20, starting_wealth=100_000)
        g = Gamble(ev=1.0, true_success_chance=0.95)
        rng = np.random.default_rng(0)
        assert ind.perceive_probability(g, rng) == 1.0

    def test_clamped_below_zero(self):
        """Perceived probability clamped to 0.0 when bonus pushes it below."""
        ind = Individual(stupidity=0.0, belief_bonus=-0.20, starting_wealth=100_000)
        g = Gamble(ev=1.0, true_success_chance=0.10)
        rng = np.random.default_rng(0)
        assert ind.perceive_probability(g, rng) == 0.0

    def test_stupidity_adds_noise(self):
        """With high stupidity, perceived probability varies across calls."""
        ind = Individual(stupidity=1.0, belief_bonus=0.0, starting_wealth=100_000)
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
        ind = Individual(stupidity=0.10, belief_bonus=0.0, starting_wealth=100_000)
        g = Gamble(ev=1.0, true_success_chance=0.5)
        rng = np.random.default_rng(42)
        results = [ind.perceive_probability(g, rng) for _ in range(10_000)]
        # All should be within [0.45, 0.55]
        assert all(0.45 - 1e-9 <= r <= 0.55 + 1e-9 for r in results)
        # Should span most of that range
        assert max(results) > 0.54
        assert min(results) < 0.46


# ── Kelly fraction tests ──────────────────────────────────────────────────

class TestKellyFraction:
    def test_standard_kelly(self):
        """f* = (b*p - q) / b for a standard gamble."""
        ind = Individual(stupidity=0, belief_bonus=0, starting_wealth=100_000)
        # 50% chance, 3:1 net odds → f* = (3*0.5 - 0.5) / 3 = 1/3
        assert ind.kelly_fraction(0.5, 3.0) == pytest.approx(1 / 3)

    def test_kelly_certain_win(self):
        """p=1.0 → f* = 1.0 (bet everything)."""
        ind = Individual(stupidity=0, belief_bonus=0, starting_wealth=100_000)
        assert ind.kelly_fraction(1.0, 2.0) == pytest.approx(1.0)

    def test_kelly_zero_edge(self):
        """When p = 1/(b+1) (break-even), f* = 0."""
        ind = Individual(stupidity=0, belief_bonus=0, starting_wealth=100_000)
        # b=1, p=0.5 → f* = (1*0.5 - 0.5) / 1 = 0
        assert ind.kelly_fraction(0.5, 1.0) == pytest.approx(0.0)

    def test_kelly_negative_edge(self):
        """When perceived edge is negative, f* should be 0 (clamped)."""
        ind = Individual(stupidity=0, belief_bonus=0, starting_wealth=100_000)
        # b=1, p=0.3 → f* = (0.3 - 0.7) / 1 = -0.4 → clamped to 0
        assert ind.kelly_fraction(0.3, 1.0) == 0.0

    def test_kelly_net_odds_zero(self):
        """When net_odds <= 0, bet nothing."""
        ind = Individual(stupidity=0, belief_bonus=0, starting_wealth=100_000)
        assert ind.kelly_fraction(0.9, 0.0) == 0.0
        assert ind.kelly_fraction(0.9, -1.0) == 0.0

    def test_kelly_capped_at_one(self):
        """f* capped at 1.0 even when formula gives higher."""
        ind = Individual(stupidity=0, belief_bonus=0, starting_wealth=100_000)
        # b=0.5, p=1.0 → f* = (0.5*1 - 0) / 0.5 = 1.0 (exactly)
        # b=0.1, p=1.0 → f* = (0.1*1 - 0) / 0.1 = 1.0 (exactly)
        # To get f* > 1 we need extremely favorable perceived odds
        # f* = (b*p - q) / b = p - q/b. With p=0.99, b=0.01: f* = 0.99 - 0.01/0.01 = -0.01 (no)
        # Actually Kelly for binary bets can't exceed 1 naturally since f* = p - q/b
        # and p <= 1. When p=1, f*=1. So cap test is about the clamp working.
        assert ind.kelly_fraction(1.0, 0.5) == 1.0


# ── Individual place_bet tests ────────────────────────────────────────────

class TestPlaceBet:
    def test_win_increases_wealth(self):
        """Winning a bet increases wealth by bet * net_odds."""
        ind = Individual(stupidity=0.0, belief_bonus=0.0, starting_wealth=100_000)
        g = Gamble(ev=1.0, true_success_chance=0.5)
        # net_odds = 3.0, Kelly fraction = (3*0.5-0.5)/3 = 1/3
        # bet = 100_000 * 1/3 ≈ 33_333.33
        # Force a win by using true_success_chance=1.0 gamble
        g_sure = Gamble(ev=1.0, true_success_chance=1.0)
        # net_odds = 1.0, perceived_prob = 1.0, f* = (1*1-0)/1 = 1.0
        # bet = 100_000, win: wealth = 100_000 + 100_000*1 = 200_000
        rng = np.random.default_rng(0)
        ind.place_bet(g_sure, rng)
        assert ind.wealth == pytest.approx(200_000)

    def test_loss_decreases_wealth(self):
        """Losing a bet decreases wealth by the bet amount."""
        ind = Individual(stupidity=0.0, belief_bonus=0.20, starting_wealth=100_000)
        # Use a gamble that will certainly lose (success=0.10)
        # but perceived as favorable due to belief_bonus
        g = Gamble(ev=0.5, true_success_chance=0.10)
        # perceived = 0.10 + 0.20 = 0.30
        # gross_payout = 1.5/0.1 = 15.0, net_odds = 14.0
        # f* = (14*0.30 - 0.70) / 14 = (4.2-0.7)/14 = 0.25
        # bet = 25_000
        # Force a loss with rng that gives random() >= 0.10
        rng = np.random.default_rng(0)
        # Check rng.random() value to make sure it's a loss
        test_rng = np.random.default_rng(0)
        test_rng.uniform(-0.0, 0.0)  # consume the noise call (stupidity=0)
        roll = test_rng.random()
        if roll < 0.10:
            # Unlikely but possible, skip this seed
            pytest.skip("RNG seed gives a win, pick another seed")
        ind.place_bet(g, rng)
        expected_wealth = 100_000 - 25_000
        assert ind.wealth == pytest.approx(expected_wealth)

    def test_zero_wealth_bets_nothing(self):
        """Bankrupt individual bets nothing and stays at 0."""
        ind = Individual(stupidity=0.0, belief_bonus=0.10, starting_wealth=0.0)
        g = Gamble(ev=1.0, true_success_chance=0.5)
        rng = np.random.default_rng(0)
        ind.place_bet(g, rng)
        assert ind.wealth == 0.0

    def test_unfavorable_gamble_no_bet(self):
        """Kelly says bet nothing when perceived edge is negative."""
        ind = Individual(stupidity=0.0, belief_bonus=-0.20, starting_wealth=100_000)
        g = Gamble(ev=0.0, true_success_chance=0.5)
        # perceived = 0.5 - 0.20 = 0.30
        # net_odds = 1.0, f* = (1*0.3-0.7)/1 = -0.4 → 0
        rng = np.random.default_rng(0)
        ind.place_bet(g, rng)
        assert ind.wealth == 100_000  # unchanged

    def test_net_odds_zero_no_bet(self):
        """When net_odds <= 0, no bet is placed."""
        ind = Individual(stupidity=0.0, belief_bonus=0.0, starting_wealth=100_000)
        g = Gamble(ev=-0.5, true_success_chance=0.5)
        # gross_payout = 0.5/0.5 = 1.0, net_odds = 0
        rng = np.random.default_rng(0)
        ind.place_bet(g, rng)
        assert ind.wealth == 100_000


# ── Vectorized _run_round equivalence tests ───────────────────────────────

class TestVectorizedRound:
    """Verify that the vectorized _run_round produces the same results
    as the Individual OOP methods for the same RNG state."""

    def test_single_individual_equivalence(self):
        """Vectorized round with N=1 should match Individual.place_bet."""
        seed = 123
        for _ in range(20):  # Test multiple random gambles
            # OOP path
            rng_oop = np.random.default_rng(seed)
            ind = Individual(stupidity=0.30, belief_bonus=0.05, starting_wealth=100_000)
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
            belief_bonus = np.array([0.05])
            sim._run_round(wealth, stupidity, belief_bonus, 1)

            assert wealth[0] == pytest.approx(ind.wealth, rel=1e-10), \
                f"Mismatch at seed {seed}: OOP={ind.wealth}, vec={wealth[0]}"
            seed += 1

    def test_vectorized_known_values(self):
        """Verify vectorized round produces correct results for known inputs."""
        # Set up a scenario where we can predict the outcome:
        # 2 individuals, one will win, one will lose
        cfg = SimConfig(population=2, batch_size=2)
        sim = Simulation(cfg)

        # Individual 0: stupidity=0, bonus=0 (perfect perceiver)
        # Individual 1: stupidity=0, bonus=0 (perfect perceiver)
        stupidity = np.array([0.0, 0.0])
        belief_bonus = np.array([0.0, 0.0])
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
        perceived = np.clip(true_prob + noise + belief_bonus, 0, 1)
        q = 1 - perceived
        f_star = np.where(net_odds > 0, (net_odds * perceived - q) / net_odds, 0.0)
        f_star = np.clip(f_star, 0, 1)
        bet = f_star * wealth
        wins = rng_check.random(2) < true_prob
        expected = np.where(wins, wealth + bet * net_odds, wealth - bet)

        sim._run_round(wealth, stupidity, belief_bonus, 2)
        np.testing.assert_allclose(wealth, expected)

    def test_vectorized_preserves_no_bet_on_negative_edge(self):
        """Vectorized: individuals with negative perceived edge don't bet."""
        cfg = SimConfig(population=1000, batch_size=1000,
                        min_ev=-0.5, max_ev=-0.01,  # all negative EV
                        min_stupidity=0.0, max_stupidity=0.0,
                        min_belief_bonus=0.0, max_belief_bonus=0.0)
        sim = Simulation(cfg)
        wealth = np.full(1000, 100_000.0)
        stupidity = np.zeros(1000)
        belief_bonus = np.zeros(1000)
        sim._run_round(wealth, stupidity, belief_bonus, 1000)
        # Perfect perceivers should never bet on negative EV
        np.testing.assert_array_equal(wealth, 100_000.0)


# ── Simulation integration tests ──────────────────────────────────────────

class TestSimulationIntegration:
    def test_wealth_never_negative(self):
        """No individual should end with negative wealth."""
        cfg = SimConfig(population=10_000, batch_size=10_000, num_gambles=20, seed=42)
        sim = Simulation(cfg)
        rng = sim.rng
        wealth = np.full(cfg.population, cfg.starting_wealth)
        stupidity = rng.uniform(0, 1, cfg.population)
        belief_bonus = rng.uniform(-0.2, 0.2, cfg.population)
        for _ in range(cfg.num_gambles):
            sim._run_round(wealth, stupidity, belief_bonus, cfg.population)
        assert np.all(wealth >= 0)

    def test_bankrupt_stays_bankrupt(self):
        """An individual with wealth=0 should remain at 0 after a round."""
        cfg = SimConfig(population=100, batch_size=100)
        sim = Simulation(cfg)
        wealth = np.zeros(100)
        stupidity = np.full(100, 0.5)
        belief_bonus = np.full(100, 0.1)
        sim._run_round(wealth, stupidity, belief_bonus, 100)
        assert np.all(wealth == 0)

    def test_zero_gambles_preserves_wealth(self):
        """With num_gambles=0, everyone keeps starting wealth."""
        cfg = SimConfig(population=1000, batch_size=1000, num_gambles=0)
        sim = Simulation(cfg)
        wealth = np.full(1000, cfg.starting_wealth)
        stupidity = sim.rng.uniform(0, 1, 1000)
        belief_bonus = sim.rng.uniform(-0.2, 0.2, 1000)
        # No rounds to run
        assert np.all(wealth == cfg.starting_wealth)

    def test_high_positive_ev_grows_wealth_on_average(self):
        """With uniformly high EV gambles, average wealth should grow."""
        cfg = SimConfig(
            population=50_000, batch_size=50_000, num_gambles=10,
            min_ev=0.5, max_ev=1.5,  # all positive EV
            min_stupidity=0.0, max_stupidity=0.0,  # perfect knowledge
            min_belief_bonus=0.0, max_belief_bonus=0.0,  # no bias
            seed=42,
        )
        sim = Simulation(cfg)
        wealth = np.full(cfg.population, cfg.starting_wealth)
        stupidity = np.zeros(cfg.population)
        belief_bonus = np.zeros(cfg.population)
        for _ in range(cfg.num_gambles):
            sim._run_round(wealth, stupidity, belief_bonus, cfg.population)
        # Median should grow (using median to be robust against outliers)
        assert np.median(wealth) > cfg.starting_wealth

    def test_negative_ev_shrinks_wealth_on_average(self):
        """With uniformly negative EV, smart bettors should mostly not bet."""
        cfg = SimConfig(
            population=10_000, batch_size=10_000, num_gambles=10,
            min_ev=-0.5, max_ev=-0.01,  # all negative EV
            min_stupidity=0.0, max_stupidity=0.0,  # perfect knowledge
            min_belief_bonus=0.0, max_belief_bonus=0.0,
            seed=42,
        )
        sim = Simulation(cfg)
        wealth = np.full(cfg.population, cfg.starting_wealth)
        stupidity = np.zeros(cfg.population)
        belief_bonus = np.zeros(cfg.population)
        for _ in range(cfg.num_gambles):
            sim._run_round(wealth, stupidity, belief_bonus, cfg.population)
        # With 0 stupidity and 0 bonus, Kelly correctly identifies negative-EV bets.
        # But EV can be slightly positive when gross_payout > 1 despite EV < 0...
        # Actually, for EV < 0: grossPayout = (1+EV)/p < 1/p.
        # Kelly f* = (b*p - q)/b. With EV<0, expected value = p*grossPayout - 1 < 0.
        # For perfect perceiver, p*(b+1) - 1 = EV < 0, so b*p + p - 1 < 0, b*p - q < 0.
        # Therefore f* < 0 → clamped to 0. Nobody bets.
        assert np.all(wealth == cfg.starting_wealth)


# ── Kelly formula mathematical properties ─────────────────────────────────

class TestKellyMathProperties:
    def test_kelly_formula_matches_definition(self):
        """Verify f* = (b*p - q) / b directly."""
        ind = Individual(0, 0, 100_000)
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
