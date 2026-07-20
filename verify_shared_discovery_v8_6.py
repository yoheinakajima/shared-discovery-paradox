#!/usr/bin/env python3
"""Verification and figure generation for the merged Shared Discovery paper (v8.6).

New in v8.6:
  * closed-form (fast) common-cue likelihood coefficients, cross-checked
    against the direct enumeration;
  * the symmetric-market cell of the proportional sparse-evidence limit:
    Poisson water-filling equation, high-precision solution, and a batched
    finite-M Monte Carlo convergence diagnostic at M in {16, 64, 256, 1024};
  * a comparative-statics grid scan verifying that the market-private gap
    under copying changes sign at most once on every tested (M, N, p)
    combination (grid_crossings.csv).


The script reproduces the independent sixteen-box benchmark, the anonymous
symmetric equal-split equilibrium, the exact 2-1/N price-of-anarchy theorem and
tight family, the sole-rescue implementation result, and the common-cue
copying comparison.  Report-count classes and the latent-cue likelihood are
enumerated exactly; one-dimensional equilibrium equations are solved to high
floating-point precision.

New in v8.1:
  * the blind coordinated control G_blind = N/M = 0.50 and the uncoordinated
    blind value 1-(1-1/M)^N = 0.403281;
  * the average-action-quality column Q for all four canonical protocols and
    the redundancy identity N*Q - G = E[(K-1)^+], verified class by class;
  * the proportional sparse-evidence limits (alpha=1/2, r=3.75) with a seeded
    Monte Carlo convergence diagnostic at larger M.

Run:
    python verify_shared_discovery_v8_6.py
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from math import comb, e, factorial
from pathlib import Path
from typing import Iterator, Sequence

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Equal-split congestion game
# ---------------------------------------------------------------------------

def phi(s: float, n: int) -> float:
    """Expected reciprocal share E[1/(K+1)], K ~ Binomial(n-1,s)."""
    if s <= 1e-15:
        return 1.0
    return (1.0 - (1.0 - s) ** n) / (n * s)


def _phi_array(s: np.ndarray, n: int) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    out = np.empty_like(s)
    near_zero = s <= 1e-12
    out[near_zero] = 1.0
    z = ~near_zero
    out[z] = (1.0 - (1.0 - s[z]) ** n) / (n * s[z])
    return out


def _dphi_array(s: np.ndarray, n: int) -> np.ndarray:
    """Stable derivative using phi(s)=N^{-1} sum_{j=0}^{N-1}(1-s)^j."""
    s = np.asarray(s, dtype=float)
    r = 1.0 - s
    out = np.zeros_like(s)
    for j in range(1, n):
        out -= (j / n) * r ** (j - 1)
    return out


class _PhiInverse:
    """Fast high-accuracy inverse using interpolation plus safeguarded Newton."""

    def __init__(self, n: int, grid_size: int = 8193) -> None:
        self.n = n
        self.s_grid = np.linspace(0.0, 1.0, grid_size)
        self.y_grid = _phi_array(self.s_grid, n)

    def __call__(self, y: np.ndarray | float) -> np.ndarray:
        arr = np.asarray(y, dtype=float)
        out = np.empty_like(arr)
        hi = arr >= 1.0
        lo = arr <= 1.0 / self.n
        mid = ~(hi | lo)
        out[hi] = 0.0
        out[lo] = 1.0
        if np.any(mid):
            target = arr[mid]
            # np.interp requires an increasing abscissa.
            s = np.interp(target, self.y_grid[::-1], self.s_grid[::-1])
            for _ in range(4):
                f = _phi_array(s, self.n) - target
                d = _dphi_array(s, self.n)
                step = np.divide(f, d, out=np.zeros_like(f), where=np.abs(d) > 1e-15)
                s = np.clip(s - step, 0.0, 1.0)
            out[mid] = s
        return out


@lru_cache(maxsize=None)
def _phi_inverse_cache(n: int) -> _PhiInverse:
    return _PhiInverse(n)


def phi_inv(y: float, n: int) -> float:
    return float(_phi_inverse_cache(n)(np.asarray(y)))


def symmetric_equal_split_equilibrium(
    posterior: Sequence[float], n: int, tol: float = 1e-14
) -> tuple[np.ndarray, float]:
    """Anonymous symmetric equilibrium and its common payoff lambda.

    A dominant-posterior corner must be treated explicitly.  If
    pi_(1)/n >= pi_(2), all players choose the top box and lambda=pi_(1)/n.
    """
    pi = np.asarray(posterior, dtype=float)
    if pi.ndim != 1 or len(pi) == 0 or np.any(pi < -tol):
        raise ValueError("posterior must be a nonnegative one-dimensional vector")
    if float(pi.sum()) <= 0:
        raise ValueError("posterior must have positive mass")
    pi = pi / float(pi.sum())

    order = np.argsort(-pi)
    top = int(order[0])
    second = float(pi[order[1]]) if len(pi) > 1 else 0.0
    if n == 1 or float(pi[top]) / n >= second - tol:
        s = np.zeros_like(pi)
        s[top] = 1.0
        return s, float(pi[top]) / n

    inv = _phi_inverse_cache(n)
    lo, hi = float(pi[top]) / n, float(pi[top])
    for _ in range(75):
        lam = (lo + hi) / 2.0
        active = pi > lam
        s = np.zeros_like(pi)
        s[active] = inv(lam / pi[active])
        if float(s.sum()) > 1.0:
            lo = lam
        else:
            hi = lam
    lam = (lo + hi) / 2.0
    active = pi > lam
    s = np.zeros_like(pi)
    s[active] = inv(lam / pi[active])
    if abs(float(s.sum()) - 1.0) > 3e-10:
        raise ArithmeticError(f"equilibrium probabilities sum to {s.sum()}")
    return s, lam


def discovery(posterior: Sequence[float], strategy: Sequence[float], n: int) -> float:
    pi = np.asarray(posterior, dtype=float)
    s = np.asarray(strategy, dtype=float)
    return float(np.sum(pi * (1.0 - (1.0 - s) ** n)))


def planner_value(posterior: Sequence[float], n: int) -> float:
    pi = np.sort(np.asarray(posterior, dtype=float))[::-1]
    return float(pi[: min(n, len(pi))].sum())


def expected_distinct(strategy: Sequence[float], n: int) -> float:
    s = np.asarray(strategy, dtype=float)
    return float(np.sum(1.0 - (1.0 - s) ** n))


# ---------------------------------------------------------------------------
# Exact report-count classes and latent common-cue likelihood coefficients
# ---------------------------------------------------------------------------

def integer_partitions(total: int, max_parts: int, max_value: int | None = None) -> Iterator[tuple[int, ...]]:
    """Yield nonincreasing positive partitions."""
    if total == 0:
        yield ()
        return
    if max_parts == 0:
        return
    if max_value is None:
        max_value = total
    for first in range(min(total, max_value), 0, -1):
        for rest in integer_partitions(total - first, max_parts - 1, first):
            yield (first,) + rest


@dataclass(frozen=True)
class CountClass:
    counts: tuple[int, ...]
    multiplicity: int


def count_classes(m: int, n: int) -> list[CountClass]:
    """Classes under permutations of the M-1 labels other than true state 0."""
    out: list[CountClass] = []
    for x0 in range(n + 1):
        for parts in integer_partitions(n - x0, m - 1):
            padded = tuple(parts) + (0,) * (m - 1 - len(parts))
            multiplicity = factorial(m - 1)
            for count in Counter(padded).values():
                multiplicity //= factorial(count)
            out.append(CountClass((x0,) + padded, multiplicity))
    return out


@dataclass
class LikelihoodClass:
    counts: np.ndarray
    multiplicity: int
    # coeff[target,k] so likelihood(c)=sum_k coeff[target,k] c^k(1-c)^(N-k)
    coeff: np.ndarray


def _prob_power(prob: float, exponent: int) -> float:
    return 1.0 if exponent == 0 else prob**exponent


def precompute_likelihood_classes(m: int, n: int, p: float) -> list[LikelihoodClass]:
    q = (1.0 - p) / (m - 1)
    result: list[LikelihoodClass] = []
    for cls in count_classes(m, n):
        x = np.asarray(cls.counts, dtype=int)
        coeff = np.zeros((m, n + 1), dtype=float)
        for target in range(m):
            probs = np.full(m, q, dtype=float)
            probs[target] = p
            for cue in range(m):
                cue_prob = float(probs[cue])
                for k in range(int(x[cue]) + 1):
                    rem = x.copy()
                    rem[cue] -= k
                    multinomial = float(factorial(n - k))
                    for r in range(m):
                        multinomial *= _prob_power(float(probs[r]), int(rem[r])) / factorial(int(rem[r]))
                    coeff[target, k] += cue_prob * comb(n, k) * multinomial
        result.append(LikelihoodClass(x, cls.multiplicity, coeff))
    return result


def likelihood_basis(c: float, n: int) -> np.ndarray:
    return np.array([_prob_power(c, k) * _prob_power(1.0 - c, n - k) for k in range(n + 1)])


def cutoff_inclusion(values: Sequence[float], target: int, budget: int, tol: float = 1e-11) -> float:
    """Uniform tie-breaking probability that target belongs to a top-budget set."""
    v = np.asarray(values, dtype=float)
    t = float(v[target])
    greater = int(np.sum(v > t + tol))
    equal = int(np.sum(np.abs(v - t) <= tol))
    if greater >= budget:
        return 0.0
    if greater + equal <= budget:
        return 1.0
    return float(budget - greater) / equal


@dataclass
class ExAnteResult:
    consensus: float
    market: float
    private: float
    planner: float
    market_distinct: float
    private_distinct: float
    frontier: tuple[float, ...]
    class_probability_sum: float
    market_quality: float = 0.0


def canonical_ex_ante(
    c: float,
    precomputed: Sequence[LikelihoodClass],
    m: int = 16,
    n: int = 8,
) -> ExAnteResult:
    """Enumerate ex-ante outcomes conditional on target 0 (symmetry makes this unconditional)."""
    basis = likelihood_basis(c, n)
    consensus = market = private = planner = 0.0
    market_distinct = private_distinct = 0.0
    market_quality = 0.0
    frontier = np.zeros(n, dtype=float)
    prob_sum = 0.0

    for cls in precomputed:
        likes = cls.coeff @ basis
        class_prob = cls.multiplicity * float(likes[0])
        if class_prob <= 1e-20:
            continue
        normalizer = float(likes.sum())
        if normalizer <= 0.0:
            continue
        posterior = likes / normalizer
        prob_sum += class_prob

        consensus += class_prob * cutoff_inclusion(posterior, 0, 1)
        s, lam = symmetric_equal_split_equilibrium(posterior, n)
        market += class_prob * (1.0 - (1.0 - float(s[0])) ** n)
        market_distinct += class_prob * expected_distinct(s, n)
        market_quality += class_prob * float(s[0])
        private += class_prob * float(cls.counts[0] >= 1)
        private_distinct += class_prob * float(np.count_nonzero(cls.counts))
        planner += class_prob * cutoff_inclusion(posterior, 0, n)
        for budget in range(1, n + 1):
            frontier[budget - 1] += class_prob * cutoff_inclusion(posterior, 0, budget)

        assert abs(discovery(posterior, s, n) - n * lam) < 3e-8

    return ExAnteResult(
        consensus,
        market,
        private,
        planner,
        market_distinct,
        private_distinct,
        tuple(float(v) for v in frontier),
        prob_sum,
        market_quality=market_quality,
    )


def private_discovery_closed_form(c: float, n: int, p: float) -> float:
    return 1.0 - p * ((1.0 - c) * (1.0 - p)) ** n - (1.0 - p) * (1.0 - p * (1.0 - c)) ** n


def find_market_private_crossover(
    precomputed: Sequence[LikelihoodClass], n: int = 8, p: float = 0.20
) -> tuple[float, float]:
    lo, hi = 0.70, 0.90
    for _ in range(38):
        mid = (lo + hi) / 2.0
        r = canonical_ex_ante(mid, precomputed, n=n)
        gap = r.market - private_discovery_closed_form(mid, n, p)
        if gap > 0.0:
            hi = mid
        else:
            lo = mid
    c_star = (lo + hi) / 2.0
    r = canonical_ex_ante(c_star, precomputed, n=n)
    return c_star, (r.market + r.private) / 2.0


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_reciprocal_share() -> None:
    for n in [2, 8, 32]:
        for s in [0.05, 0.2, 0.5, 0.9]:
            direct = sum(comb(n - 1, j) * s**j * (1 - s) ** (n - 1 - j) / (j + 1) for j in range(n))
            assert abs(direct - phi(s, n)) < 1e-12
    print("[ok] reciprocal-share identity")


def check_equilibrium_and_welfare() -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        m = int(rng.integers(2, 30))
        n = int(rng.integers(2, 12))
        pi = rng.random(m)
        if rng.random() < 0.25:
            pi[0] += 20.0  # force pure-corner tests
        pi /= pi.sum()
        s, lam = symmetric_equal_split_equilibrium(pi, n)
        assert abs(float(s.sum()) - 1.0) < 3e-10
        payoffs = np.array([pi[b] * phi(float(s[b]), n) for b in range(m)])
        support = s > 1e-9
        assert np.max(np.abs(payoffs[support] - lam)) < 3e-8
        if np.any(~support):
            assert np.max(payoffs[~support]) <= lam + 3e-8
        assert abs(discovery(pi, s, n) - n * lam) < 3e-8
    print("[ok] symmetric equilibrium, pure corner, and G=N lambda (200 posteriors)")


def check_uniform() -> None:
    for n in [2, 8, 32]:
        for t in [2, n, 2 * n, 60]:
            pi = np.full(t, 1.0 / t)
            s, _ = symmetric_equal_split_equilibrium(pi, n)
            assert np.max(np.abs(s - 1.0 / t)) < 2e-9
            assert abs(discovery(pi, s, n) - (1.0 - (1.0 - 1.0 / t) ** n)) < 2e-9
    ratio = 1.0 / (1.0 - (1.0 - 1.0 / 8.0) ** 8)
    print(f"[ok] uniform symmetric-mixing ratio: N=8 -> {ratio:.6f}; limit e/(e-1)={e/(e-1):.6f}")


def check_exact_poa() -> None:
    for n in [2, 8, 16, 32]:
        welfare = 0.5
        optimum = 0.5 + (n - 1) / (2.0 * n)
        ratio = optimum / welfare
        assert abs(ratio - (2.0 - 1.0 / n)) < 1e-12
        print(f"     N={n:2d}: tight PoA={ratio:.6f} = 2-1/N")
    print("[ok] exact mixed price of anarchy is 2-1/N (tight family)")


def check_sole_rescue() -> None:
    rng = np.random.default_rng(3)
    for _ in range(300):
        pi = rng.random(16) + 1e-6
        pi /= pi.sum()
        top = set(np.argsort(-pi)[:8].tolist())
        assert max(pi[b] for b in range(16) if b not in top) <= min(pi[b] for b in top) + 1e-12
    print("[ok] sole-rescue fully implements the top-N portfolio in pure Nash equilibrium")


# ---------------------------------------------------------------------------
# v8.1 additions: blind control, average action quality, scaling limits
# ---------------------------------------------------------------------------

def check_blind_control(consensus: float, m: int = 16, n: int = 8) -> None:
    """Coordinated and uncoordinated no-clue benchmarks."""
    blind_coordinated = n / m
    blind_uncoordinated = 1.0 - (1.0 - 1.0 / m) ** n
    assert abs(blind_coordinated - 0.50) < 1e-15
    assert abs(blind_uncoordinated - 0.403281) < 5e-7
    assert blind_coordinated > consensus
    assert blind_uncoordinated > consensus
    print(f"[ok] blind coordinated portfolio N/M = {blind_coordinated:.6f} > consensus {consensus:.6f}")
    print(f"     blind uncoordinated 1-(1-1/M)^N = {blind_uncoordinated:.12f} (footnote value)")
    # Hybrid family with k blind sentinels and N-k clue-followers:
    # G_k = 1 - (1 - k/M)(1-p)^{N-k}, maximized at k=0 canonically.
    p = 0.20
    hybrid = [1.0 - (1.0 - k / m) * (1.0 - p) ** (n - k) for k in range(n + 1)]
    assert max(range(n + 1), key=lambda k: hybrid[k]) == 0
    assert all(hybrid[k] > hybrid[k + 1] for k in range(n))
    print("     blind-sentinel hybrid G_k maximized at k=0 (extensions claim)")


def check_quality_and_redundancy(
    precomputed: Sequence[LikelihoodClass],
    independent: ExAnteResult,
    m: int = 16,
    n: int = 8,
    p: float = 0.20,
) -> dict[str, float]:
    """Average standalone action quality Q per protocol, plus N*Q - G = E[(K-1)^+].

    K is the number of actions selecting the target.  The identity
    N*Q - G = E[(K-1)^+] (expected redundant hits) is verified class by class
    for the equal-split market, and in closed form for the other protocols.
    """
    basis = likelihood_basis(0.0, n)
    q_market = 0.0
    redundant_market_direct = 0.0
    for cls in precomputed:
        likes = cls.coeff @ basis
        class_prob = cls.multiplicity * float(likes[0])
        if class_prob <= 1e-20:
            continue
        posterior = likes / float(likes.sum())
        s, _ = symmetric_equal_split_equilibrium(posterior, n)
        s0 = float(s[0])
        q_market += class_prob * s0
        # E[(K-1)^+] with K ~ Binomial(n, s0), computed directly and in closed form.
        direct = sum(max(k - 1, 0) * comb(n, k) * s0**k * (1.0 - s0) ** (n - k) for k in range(n + 1))
        closed = n * s0 - (1.0 - (1.0 - s0) ** n)
        assert abs(direct - closed) < 1e-12
        redundant_market_direct += class_prob * direct

    q = {
        "consensus": independent.consensus,          # all N actions repeat the mode
        "market": q_market,
        "private": p,                                 # each clue is correct w.p. p
        "planner": independent.planner / n,           # collision-free portfolio
    }
    assert abs(q_market - independent.market_quality) < 1e-12

    # Redundancy identity N*Q - G = E[(K-1)^+] for every protocol.
    checks = {
        "consensus": (n * q["consensus"] - independent.consensus, (n - 1) * independent.consensus),
        "market": (n * q["market"] - independent.market, redundant_market_direct),
        "private": (n * p - independent.private, n * p - (1.0 - (1.0 - p) ** n)),
        "planner": (n * q["planner"] - independent.planner, 0.0),
    }
    for name, (lhs, rhs) in checks.items():
        assert abs(lhs - rhs) < 3e-9, name
    # Exact inversion: Q decreasing while G increasing across the ordering.
    assert q["consensus"] > q["market"] > q["private"] > q["planner"]
    assert independent.consensus < independent.market < independent.private < independent.planner

    print("[ok] average action quality and redundancy identity N*Q - G = E[(K-1)^+]")
    print("     protocol            Q               G               redundant hits")
    for name, g in [
        ("consensus", independent.consensus),
        ("market", independent.market),
        ("private", independent.private),
        ("planner", independent.planner),
    ]:
        print(f"     {name:<18}{q[name]:.12f}  {g:.12f}  {n * q[name] - g:.12f}")
    return q


def check_scaling_limits(seed: int = 11, samples: int = 60_000) -> None:
    """Proportional sparse-evidence limits: N/M -> alpha with fixed r = p/q.

    Verifies the limiting formulas at the canonical scaling alpha=1/2, r=3.75
    and runs a seeded Monte Carlo convergence diagnostic at growing M.
    """
    alpha, r = 0.5, 3.75
    g_blind = alpha
    g_private_limit = 1.0 - np.exp(-alpha * r)
    g_portfolio_limit = 1.0 - (1.0 - alpha) * np.exp(-alpha * (r - 1.0))
    tau = 1.0 - (1.0 - alpha) * np.exp(alpha)
    # Fill-identity form equals the closed form.
    assert abs(g_private_limit + np.exp(-alpha * r) * tau - g_portfolio_limit) < 1e-15
    assert abs(g_private_limit - 0.846645) < 5e-7
    assert abs(g_portfolio_limit - 0.873580) < 5e-7
    assert 0.0 < tau < 1.0
    print(f"[ok] scaling limits (alpha=0.5, r=3.75): blind={g_blind:.6f}, "
          f"private={g_private_limit:.12f}, portfolio={g_portfolio_limit:.12f}")

    rng = np.random.default_rng(seed)
    consensus_path = []
    for m in [16, 64, 256]:
        n = m // 2
        p_m = r / (m - 1 + r)
        probs = np.full(m, (1.0 - p_m) / (m - 1))
        probs[0] = p_m
        counts = rng.multinomial(n, probs, size=samples)
        c0 = counts[:, 0]
        false = counts[:, 1:]
        max_false = false.max(axis=1)
        ties = (false == c0[:, None]).sum(axis=1) + 1
        consensus_mc = float(np.mean(np.where(c0 > max_false, 1.0,
                                     np.where(c0 == max_false, 1.0 / ties, 0.0))))
        named = c0 >= 1
        d_false = (false > 0).sum(axis=1)
        portfolio_mc = float(np.mean(np.where(named, 1.0, (n - d_false) / (m - d_false))))
        private_exact = 1.0 - (1.0 - p_m) ** n
        consensus_path.append(consensus_mc)
        print(f"     M={m:4d}, N={n:3d}: consensus~{consensus_mc:.4f}, "
              f"private={private_exact:.6f}, portfolio~{portfolio_mc:.6f}")
        if m == 256:
            assert abs(private_exact - g_private_limit) < 5e-3
            assert abs(portfolio_mc - g_portfolio_limit) < 8e-3
    assert consensus_path[0] > consensus_path[1] > consensus_path[2]
    print("[ok] Monte Carlo convergence diagnostic: consensus declines toward 0, "
          "private and portfolio approach their limits")


# ---------------------------------------------------------------------------
# v8.6 additions: fast likelihood coefficients, market scaling limit,
# batched finite-M market Monte Carlo, comparative-statics grid scan
# ---------------------------------------------------------------------------

def _falling(x: int, k: int) -> float:
    """Falling factorial x!/(x-k)!, zero when x < k."""
    if x < k:
        return 0.0
    out = 1.0
    for i in range(k):
        out *= x - i
    return out


def fast_precompute_likelihood_classes(m: int, n: int, p: float) -> list[LikelihoodClass]:
    """Closed-form common-cue likelihood coefficients.

    For count vector x with A = prod_r x_r! and target count T = x[target],

      coeff[target, k] = C(n,k) (n-k)!/A * [ p^{T-k+1} q^{n-T} fall(T,k)
                          + p^T q^{n-T-k+1} (F(k) - fall(T,k)) ],

    where fall(x,k)=x!/(x-k)! (zero if x<k) and F(k)=sum_b fall(x_b,k).
    The first bracket term is the cue-at-target contribution, the second
    collects all other cue positions.  Algebraically identical to the direct
    enumeration in precompute_likelihood_classes but O(m n) per class.
    """
    q = (1.0 - p) / (m - 1)
    result: list[LikelihoodClass] = []
    for cls in count_classes(m, n):
        x = np.asarray(cls.counts, dtype=int)
        inv_a = 1.0
        for xr in x:
            inv_a /= factorial(int(xr))
        fall = np.array([[_falling(int(xb), k) for k in range(n + 1)] for xb in x])
        f_sum = fall.sum(axis=0)  # F(k)
        coeff = np.zeros((m, n + 1), dtype=float)
        for target in range(m):
            t = int(x[target])
            for k in range(n + 1):
                pref = comb(n, k) * factorial(n - k) * inv_a
                cue_at_target = (
                    _prob_power(p, t - k + 1) * _prob_power(q, n - t) * fall[target, k]
                    if t >= k
                    else 0.0
                )
                cue_elsewhere = (
                    _prob_power(p, t)
                    * _prob_power(q, n - t - k + 1)
                    * (f_sum[k] - fall[target, k])
                )
                coeff[target, k] = pref * (cue_at_target + cue_elsewhere)
        result.append(LikelihoodClass(x, cls.multiplicity, coeff))
    return result


def check_fast_likelihood() -> None:
    """Cross-check the closed-form coefficients against direct enumeration."""
    for m, n, p in [(16, 8, 0.20), (7, 5, 0.30), (9, 4, 0.15)]:
        slow = precompute_likelihood_classes(m, n, p)
        fast = fast_precompute_likelihood_classes(m, n, p)
        assert len(slow) == len(fast)
        worst = 0.0
        for a, b in zip(slow, fast):
            assert np.array_equal(a.counts, b.counts) and a.multiplicity == b.multiplicity
            scale = max(np.max(np.abs(a.coeff)), 1e-300)
            worst = max(worst, float(np.max(np.abs(a.coeff - b.coeff))) / scale)
        assert worst < 1e-12, worst
    print("[ok] closed-form likelihood coefficients match direct enumeration "
          "on (16,8,0.20), (7,5,0.30), (9,4,0.15)")


# --- batched anonymous-equilibrium solver (vectorized across samples) ------

def _phi_batch(s: np.ndarray, n: int) -> np.ndarray:
    """phi(s) = (1-(1-s)^n)/(n s), numerically stable, vectorized."""
    out = np.ones_like(s)
    z = s > 1e-14
    sz = np.clip(s[z], None, 1.0 - 1e-16)
    out[z] = -np.expm1(n * np.log1p(-sz)) / (n * sz)
    return out


def _phi_inv_batch(y: np.ndarray, n: int, iters: int = 60) -> np.ndarray:
    """Solve phi(s)=y elementwise by bisection on s in [0,1]."""
    lo = np.zeros_like(y)
    hi = np.ones_like(y)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        too_big = _phi_batch(mid, n) > y  # phi decreasing: value too large -> s too small
        lo = np.where(too_big, mid, lo)
        hi = np.where(too_big, hi, mid)
    s = 0.5 * (lo + hi)
    s[y >= 1.0] = 0.0
    s[y <= 1.0 / n] = 1.0
    return s


def batched_market_values(mult: np.ndarray, weights: np.ndarray, n: int,
                          iters: int = 80, return_details: bool = False):
    """Discovery G = n*lambda for the anonymous equilibrium, per sample.

    mult:    (S, J) integer multiplicities of posterior classes (>=0 allowed)
    weights: (J,) or (S, J) unnormalized posterior weight of one box per class
    Reproduces the semantics of symmetric_equal_split_equilibrium, including
    the dominant-posterior corner.  For n <= 32 the inner inverse reuses the
    interpolation-Newton machinery; for larger n it falls back to bisection.
    """
    s_count, j_count = mult.shape
    w = np.asarray(weights, dtype=float)
    if w.ndim == 1:
        w = np.broadcast_to(w, (s_count, j_count))
    z = (mult * w).sum(axis=1)
    v = w / z[:, None]  # per-box posterior by class
    occupied = mult > 0
    # top and second-largest distinct posterior values present in each sample
    big = np.where(occupied, v, -np.inf)
    top = big.max(axis=1)
    top_j = big.argmax(axis=1)
    second_candidates = big.copy()
    rows = np.arange(s_count)
    # second value: next distinct class below the top, or the top itself if
    # the top class holds two or more boxes (two boxes share the same value)
    second_candidates[rows, top_j] = np.where(
        mult[rows, top_j] >= 2, big[rows, top_j], -np.inf
    )
    second = second_candidates.max(axis=1)
    second = np.where(np.isfinite(second), second, 0.0)
    pure = top / n >= second - 1e-14
    lam_pure = top / n

    if n <= 32:
        inv = _phi_inverse_cache(n)

        def phi_inv_fn(y: np.ndarray) -> np.ndarray:
            return inv(y)
    else:
        def phi_inv_fn(y: np.ndarray) -> np.ndarray:
            return _phi_inv_batch(y, n)

    lo = top / n
    hi = top.copy()
    with np.errstate(divide="ignore"):
        for _ in range(iters):
            lam = 0.5 * (lo + hi)
            ratio = lam[:, None] / v
            s = np.where(v > lam[:, None], phi_inv_fn(ratio), 0.0)
            budget = (mult * s).sum(axis=1)
            lo = np.where(budget > 1.0, lam, lo)
            hi = np.where(budget > 1.0, hi, lam)
    lam = 0.5 * (lo + hi)
    lam = np.where(pure, lam_pure, lam)
    if return_details:
        return n * lam, lam, pure, v, top_j
    return n * lam


def check_batched_solver(seed: int = 5, trials: int = 60) -> None:
    """Batched solver agrees with the scalar water-filling solver."""
    rng = np.random.default_rng(seed)
    for _ in range(trials):
        m = int(rng.integers(3, 12))
        n = int(rng.integers(2, 10))
        j_count = int(rng.integers(2, 5))
        weights = np.sort(rng.uniform(0.05, 3.0, size=j_count))
        mult = rng.integers(0, 4, size=(1, j_count))
        if mult.sum() < 2:
            mult[0, -1] += 2
        posterior = np.repeat(weights, mult[0])
        posterior = posterior / posterior.sum()
        _, lam = symmetric_equal_split_equilibrium(posterior, n)
        g_scalar = n * lam
        g_batch = float(batched_market_values(mult, weights, n)[0])
        assert abs(g_scalar - g_batch) < 2e-9, (g_scalar, g_batch)
    print("[ok] batched equilibrium solver matches the scalar solver "
          f"on {trials} random class posteriors")


# --- proportional sparse-evidence limit: the market cell -------------------

def _h(x: float) -> float:
    return 1.0 if x <= 1e-14 else -math.expm1(-x) / x


def _h_inv(y: float) -> float:
    """Solve h(x) = (1-exp(-x))/x = y for y in (0,1)."""
    if y >= 1.0:
        return 0.0
    lo, hi = 1e-15, max(4.0, 2.0 / y)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _h(mid) > y:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def market_scaling_limit(alpha: float, r: float, jmax: int = 160) -> dict:
    """Limit of the anonymous symmetric market under proportional scaling.

    G_market -> alpha * Lambda, where Lambda uniquely solves

        sum_j f_j h^{-1}(Lambda/psi_j) 1{psi_j > Lambda} = alpha,

    f_j = Poisson(alpha) mass, psi_j = r^j exp(-alpha(r-1)),
    h(x) = (1-e^{-x})/x.  The tilted form g_j = f_j psi_j = Poisson(alpha r)
    mass gives the equivalent expression G = sum_j g_j (1 - e^{-x_j}).
    """
    log_alpha, log_r = math.log(alpha), math.log(r)
    f = [math.exp(-alpha + j * log_alpha - math.lgamma(j + 1)) for j in range(jmax)]
    g = [math.exp(-alpha * r + j * (log_alpha + log_r) - math.lgamma(j + 1))
         for j in range(jmax)]
    log_psi = [j * log_r - alpha * (r - 1.0) for j in range(jmax)]

    def budget_and_cover(lam: float) -> tuple[float, float, float]:
        budget = cover = leak = 0.0
        log_lam = math.log(lam)
        for j in range(jmax):
            log_ratio = log_lam - log_psi[j]
            if log_ratio >= 0.0:
                continue  # psi_j <= lam: inactive
            if log_ratio < -34.0:  # ratio < ~1.7e-15: x_j = psi_j/lam exactly enough
                budget += g[j] / lam
                cover += g[j]
                continue
            x_j = _h_inv(math.exp(log_ratio))
            budget += f[j] * x_j
            cover += g[j] * -math.expm1(-x_j)
            leak += g[j] * math.exp(-x_j)
        return budget, cover, leak

    lo, hi = 1e-12, 1.0 / alpha
    assert budget_and_cover(lo)[0] > alpha > budget_and_cover(hi)[0]
    for _ in range(200):
        lam = 0.5 * (lo + hi)
        if budget_and_cover(lam)[0] > alpha:
            lo = lam
        else:
            hi = lam
    lam = 0.5 * (lo + hi)
    budget, cover, leak = budget_and_cover(lam)
    active_min = next(j for j in range(jmax) if log_psi[j] > math.log(lam))
    abandoned = sum(g[j] for j in range(jmax) if log_psi[j] <= math.log(lam))
    return {
        "Lambda": lam,
        "G": alpha * lam,
        "G_tilted": cover,
        "budget_residual": budget - alpha,
        "active_min_count": active_min,
        "g0": g[0],
        "g1": g[1],
        "abandoned_mass": abandoned,
        "collision_leak": leak,
        "private_limit": -math.expm1(-alpha * r),
    }


def simulate_market_scaling(alpha: float, r: float, m: int, samples: int,
                            rng: np.random.Generator,
                            chunk: int = 2000) -> tuple[float, float]:
    """Monte Carlo mean and standard error of finite-M market discovery."""
    n = int(round(alpha * m))
    p_m = r / (m - 1 + r)
    uniform_false = np.full(m - 1, 1.0 / (m - 1))
    values = np.empty(samples)
    done = 0
    while done < samples:
        s_now = min(chunk, samples - done)
        t = rng.binomial(n, p_m, size=s_now)
        per_sample: list[tuple[int, np.ndarray]] = [None] * s_now  # type: ignore
        for rem in np.unique(t):
            rows = np.nonzero(t == rem)[0]
            draws = rng.multinomial(int(n - rem), uniform_false, size=len(rows))
            for i, row in enumerate(rows):
                per_sample[row] = (int(rem), draws[i])
        j_top = 1 + max(max(rem, int(d.max())) for rem, d in per_sample)
        mult = np.zeros((s_now, j_top), dtype=np.int64)
        for row, (rem, d) in enumerate(per_sample):
            mult[row] = np.bincount(d, minlength=j_top)
            mult[row, rem] += 1  # the target box joins its count class
        weights = r ** np.arange(j_top, dtype=float)
        values[done:done + s_now] = batched_market_values(mult, weights, n)
        done += s_now
    mean = float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(samples))
    return mean, se


def check_market_scaling_limit(seed: int = 17) -> dict:
    alpha, r = 0.5, 3.75
    limit = market_scaling_limit(alpha, r)
    assert abs(limit["budget_residual"]) < 1e-11
    assert abs(limit["G"] - limit["G_tilted"]) < 1e-11
    assert limit["active_min_count"] == 2  # limit market abandons 0- and 1-report boxes
    assert abs(limit["Lambda"] - 1.094021073740) < 5e-11
    assert abs(limit["G"] - 0.547010536870) < 5e-11
    # gap decomposition: private - market = P(exactly one report) + collision leak
    gap = limit["private_limit"] - limit["G"]
    assert abs(gap - (limit["g1"] + limit["collision_leak"])) < 1e-11
    assert abs(limit["g1"] - 0.287540562834) < 5e-11
    assert abs(limit["collision_leak"] - 0.012093933451) < 5e-11
    print(f"[ok] gap decomposition: private-market = {gap:.12f} = "
          f"g1 {limit['g1']:.12f} + collision leak {limit['collision_leak']:.12f}")
    print(f"[ok] market scaling limit: Lambda={limit['Lambda']:.12f}, "
          f"G={limit['G']:.12f} (tilted form matches, residual "
          f"{limit['budget_residual']:.1e}, active counts >= "
          f"{limit['active_min_count']})")

    rng = np.random.default_rng(seed)
    plan = [(16, 40_000), (64, 30_000), (256, 20_000), (1024, 8_000)]
    means = []
    for m, samples in plan:
        mean, se = simulate_market_scaling(alpha, r, m, samples, rng)
        means.append(mean)
        print(f"     M={m:5d}, N={m // 2:4d}: market~{mean:.6f} (se {se:.6f})")
        if m == 16:
            assert abs(mean - 0.599099252439) < 4.0 * se + 1e-9
    assert means[0] > means[1] > means[2] > means[3] > limit["G"] - 5e-3
    assert abs(means[-1] - limit["G"]) < 0.02
    print("[ok] finite-M market values decline monotonically toward the limit")
    return limit


def check_partial_conformity(precomputed: Sequence[LikelihoodClass],
                             m: int = 16, n: int = 8) -> None:
    """Mechanical adoption of the plurality winner is monotonically harmful.

    With a adopters (any exchangeable subset) replacing their own clue by the
    pooled plurality recommendation, discovery is
      G(a) = sum_class P(class) [1 - (1 - tau1) rho(t, a)],
    where tau1 is the tie-broken probability that consensus hits the target
    and rho(t, a) = C(N-t, a-t)/C(N, a) is the probability that all t
    target-clue holders are among the a adopters.  Endpoints are exactly
    G(0) = G_private and G(N) = G_consensus; the curve is strictly
    decreasing (the winner is always an already-named box, so adoption
    deletes coverage and never adds it).
    """
    basis = likelihood_basis(0.0, n)
    curve = np.zeros(n + 1)
    for cls in precomputed:
        likes = cls.coeff @ basis
        class_prob = cls.multiplicity * float(likes[0])
        if class_prob <= 1e-20:
            continue
        posterior = likes / likes.sum()
        tau1 = cutoff_inclusion(posterior, 0, 1)
        t = int(cls.counts[0])
        for a in range(n + 1):
            rho = comb(n - t, a - t) / comb(n, a) if a >= t else 0.0
            curve[a] += class_prob * (1.0 - (1.0 - tau1) * rho)
    assert abs(curve[0] - 0.832227840000) < 1e-9
    assert abs(curve[n] - 0.383468709731) < 1e-9
    assert all(curve[a] > curve[a + 1] for a in range(n))
    assert abs(curve[1] - 0.791279999598) < 1e-9  # first adopter already costs 4.1pp
    values = ", ".join(f"{g:.4f}" for g in curve)
    print(f"[ok] partial conformity: G(a) strictly decreasing, a=0..{n}: {values}")


def check_abundant_evidence(seed: int = 23, samples: int = 60_000) -> None:
    """Fixed p (abundant evidence): consensus is consistent, the paradox vanishes.

    With p fixed and N_M/M -> alpha, the target's expected count N p grows
    linearly while false counts stay bounded, so the plurality winner is the
    target with probability tending to one.
    """
    alpha, p = 0.5, 0.20
    rng = np.random.default_rng(seed)
    path = []
    for m in [16, 64, 256]:
        n = m // 2
        probs = np.full(m, (1.0 - p) / (m - 1))
        probs[0] = p
        counts = rng.multinomial(n, probs, size=samples)
        c0 = counts[:, 0]
        false = counts[:, 1:]
        max_false = false.max(axis=1)
        ties = (false == c0[:, None]).sum(axis=1) + 1
        consensus_mc = float(np.mean(np.where(c0 > max_false, 1.0,
                                     np.where(c0 == max_false, 1.0 / ties, 0.0))))
        path.append(consensus_mc)
        print(f"     M={m:4d}, N={n:3d}, p={p}: consensus~{consensus_mc:.4f}")
    assert path[0] < path[1] < path[2]
    assert path[2] > 0.995
    print("[ok] abundant-evidence regime: fixed-p consensus rises toward one "
          "(the paradox is a sparse-evidence phenomenon)")


# --- comparative-statics grid: single crossing of market vs private --------
def market_ex_ante(c: float, precomputed: Sequence[LikelihoodClass],
                   n: int) -> float:
    """Ex-ante anonymous market discovery only (batched across count classes).

    Boxes with equal counts have bit-identical posterior entries under the
    closed-form coefficients, so grouping by np.unique is exact.
    """
    basis = likelihood_basis(c, n)
    class_probs: list[float] = []
    rows_w: list[np.ndarray] = []
    rows_m: list[np.ndarray] = []
    t_slots: list[int] = []
    for cls in precomputed:
        likes = cls.coeff @ basis
        class_prob = cls.multiplicity * float(likes[0])
        if class_prob <= 1e-20:
            continue
        normalizer = float(likes.sum())
        if normalizer <= 0.0:
            continue
        posterior = likes / normalizer
        vals, inverse, cnts = np.unique(posterior, return_inverse=True,
                                        return_counts=True)
        class_probs.append(class_prob)
        rows_w.append(vals)
        rows_m.append(cnts)
        t_slots.append(int(inverse[0]))
    s_count = len(class_probs)
    j_count = max(len(w) for w in rows_w)
    w = np.zeros((s_count, j_count))
    mult = np.zeros((s_count, j_count), dtype=np.int64)
    for i, (vals, cnts) in enumerate(zip(rows_w, rows_m)):
        w[i, : len(vals)] = vals
        mult[i, : len(cnts)] = cnts
    _, lam, pure, v, top_j = batched_market_values(mult, w, n, return_details=True)
    t = np.asarray(t_slots)
    rows = np.arange(s_count)
    v_t = v[rows, t]
    inv = _phi_inverse_cache(n)
    with np.errstate(divide="ignore"):
        s_t = np.where(v_t > lam, inv(np.divide(lam, v_t,
                       out=np.full_like(lam, np.inf), where=v_t > 0)), 0.0)
    s_t = np.where(pure, (t == top_j).astype(float), s_t)
    coverage = 1.0 - (1.0 - s_t) ** n
    return float(np.dot(np.asarray(class_probs), coverage))


def check_grid_single_crossing(out_dir: Path) -> None:
    """Market-private sign changes across a comparative-statics grid."""
    pre16 = fast_precompute_likelihood_classes(16, 8, 0.20)
    for c_check in (0.0, 0.40, 0.80):
        lean = market_ex_ante(c_check, pre16, 8)
        full = canonical_ex_ante(c_check, pre16, 16, 8).market
        assert abs(lean - full) < 1e-10, (c_check, lean, full)

    m_values = [8, 12, 16, 24, 32]
    n_values = [4, 6, 8]
    p_values = [0.10, 0.15, 0.20, 0.25, 0.30]
    c_grid = np.round(np.arange(0.0, 0.981, 0.02), 4)
    rows = []
    counts = {0: 0, 1: 0}
    for m in m_values:
        for n in n_values:
            for p in p_values:
                if p <= 1.0 / m:  # require r = p/q > 1
                    continue
                pre = fast_precompute_likelihood_classes(m, n, p)
                gaps = np.array([
                    market_ex_ante(c, pre, n) - private_discovery_closed_form(c, n, p)
                    for c in c_grid
                ])
                signs = np.sign(np.where(np.abs(gaps) < 1e-12, 0.0, gaps))
                nz = signs[signs != 0.0]
                crossings = int(np.sum(nz[1:] != nz[:-1]))
                c_star = float("nan")
                if crossings == 1:
                    idx = int(np.nonzero(np.diff(nz) != 0)[0][0])
                    keep = np.nonzero(signs != 0.0)[0]
                    lo, hi = float(c_grid[keep[idx]]), float(c_grid[keep[idx + 1]])
                    for _ in range(30):
                        mid = 0.5 * (lo + hi)
                        gap = market_ex_ante(mid, pre, n) - \
                            private_discovery_closed_form(mid, n, p)
                        if gap > 0.0:
                            hi = mid
                        else:
                            lo = mid
                    c_star = 0.5 * (lo + hi)
                counts[crossings] = counts.get(crossings, 0) + 1
                rows.append((m, n, p, float(gaps[0]), crossings, c_star))
                assert crossings <= 1, (m, n, p, crossings)

    path = out_dir / "grid_crossings.csv"
    with path.open("w", encoding="utf-8") as fh:
        fh.write("M,N,p,gap_at_c0,sign_changes,c_star\n")
        for m, n, p, gap0, crossings, c_star in rows:
            star = "" if math.isnan(c_star) else f"{c_star:.6f}"
            fh.write(f"{m},{n},{p:.2f},{gap0:.9f},{crossings},{star}\n")
    total = len(rows)
    always_market = sum(1 for row in rows if row[4] == 0 and row[3] > 0)
    always_private = sum(1 for row in rows if row[4] == 0 and row[3] < 0)
    print(f"[ok] grid scan: {total} (M,N,p) combinations, sign changes at most "
          f"once everywhere ({counts.get(1, 0)} single crossings, "
          f"{always_market} market-dominant, {always_private} private-dominant); "
          f"details in {path.name}")


# ---------------------------------------------------------------------------
# Figures and outputs
# ---------------------------------------------------------------------------

def generate_figures(
    out_dir: Path,
    independent: ExAnteResult,
    grid: list[float],
    results: list[ExAnteResult],
    c_star: float,
    crossing_value: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = ["Consensus", "Symmetric market", "Private clues", "Planner portfolio"]
    values = [independent.consensus, independent.market, independent.private, independent.planner]
    distinct = [1.0, independent.market_distinct, independent.private_distinct, 8.0]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    bars = ax.bar(labels, values)
    ax.set_ylim(0.0, 0.95)
    ax.set_ylabel("Discovery probability")
    ax.set_title("Same search problem, different action protocols")
    ax.axhline(0.5, linestyle="--", color="0.35", linewidth=1.1, zorder=0)
    ax.text(0.995, 0.512, "blind coordinated portfolio $= 8/16 = 0.50$",
            transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=8, color="0.25")
    for bar, value, d in zip(bars, values, distinct):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.018, f"{value:.3f}", ha="center", va="bottom")
        ax.text(bar.get_x() + bar.get_width() / 2, 0.03, rf"$E[D]={d:.2f}$", ha="center", va="bottom", rotation=90, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "canonical_protocols.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "canonical_protocols.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    budgets = np.arange(1, 9)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(budgets, independent.frontier, marker="o", label="Pooled portfolio frontier")
    ax.axhline(independent.private, linestyle="--", label="Private clue-following")
    ax.scatter([7], [independent.frontier[6]], zorder=3)
    ax.annotate(r"recovery budget $L^*=7$", xy=(7, independent.frontier[6]), xytext=(4.7, 0.875), arrowprops={"arrowstyle": "->"})
    ax.set_xlim(0.8, 8.2)
    ax.set_ylim(0.35, 0.90)
    ax.set_xlabel("Coordinated action budget L")
    ax.set_ylabel("Discovery probability")
    ax.set_title("Discovery is a budget-indexed frontier")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "pooled_frontier.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "pooled_frontier.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(grid, [r.consensus for r in results], marker="o", label="Consensus")
    ax.plot(grid, [r.market for r in results], marker="o", label="Symmetric equal-split market")
    ax.plot(grid, [r.private for r in results], marker="o", label="Private report-following")
    ax.plot(grid, [r.planner for r in results], marker="o", label="Bayes-optimal portfolio")
    ax.scatter([c_star], [crossing_value], zorder=4)
    ax.annotate(f"market/private crossover\n$c^*={c_star:.3f}$", xy=(c_star, crossing_value), xytext=(0.53, 0.34), arrowprops={"arrowstyle": "->"})
    ax.set_xlim(-0.02, 0.97)
    ax.set_ylim(0.15, 0.90)
    ax.set_xlabel("Common-cue copying probability c")
    ax.set_ylabel("Discovery probability")
    ax.set_title("Correlation changes which decentralized protocol performs better")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "copying_protocols.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "copying_protocols.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(grid, [r.planner - r.private for r in results], marker="o")
    ax.set_xlim(-0.02, 0.97)
    ax.set_ylim(0.0, 0.36)
    ax.set_xlabel("Common-cue copying probability c")
    ax.set_ylabel("Planner gain over private search")
    ax.set_title("Duplicated capacity becomes more valuable to reallocate")
    fig.tight_layout()
    fig.savefig(out_dir / "planner_gain.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "planner_gain.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_csvs(out_dir: Path, grid: list[float], results: list[ExAnteResult]) -> None:
    with (out_dir / "copying_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["c", "consensus", "symmetric_market", "private", "planner", "market_distinct", "private_distinct"])
        for c, r in zip(grid, results):
            w.writerow([f"{c:.8f}", *[f"{v:.12f}" for v in (r.consensus, r.market, r.private, r.planner, r.market_distinct, r.private_distinct)]])

    with (out_dir / "independent_frontier.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["budget", "pooled_discovery"])
        for budget, value in enumerate(results[0].frontier, start=1):
            w.writerow([budget, f"{value:.12f}"])


def write_quality_csv(out_dir: Path, independent: ExAnteResult, q: dict[str, float], n: int = 8) -> None:
    rows = [
        ("consensus", q["consensus"], independent.consensus, 1.0),
        ("symmetric_market", q["market"], independent.market, independent.market_distinct),
        ("private_clues", q["private"], independent.private, independent.private_distinct),
        ("planner_portfolio", q["planner"], independent.planner, 8.0),
    ]
    with (out_dir / "canonical_quality.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["protocol", "avg_action_quality_Q", "discovery_G", "expected_distinct", "redundant_hits"])
        for name, quality, g, distinct in rows:
            w.writerow([name, f"{quality:.12f}", f"{g:.12f}", f"{distinct:.6f}", f"{n * quality - g:.12f}"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    out_dir = args.out_dir

    m, n, p = 16, 8, 0.20
    check_reciprocal_share()
    check_equilibrium_and_welfare()
    check_uniform()
    check_exact_poa()
    check_sole_rescue()
    check_fast_likelihood()
    check_batched_solver()

    precomputed = fast_precompute_likelihood_classes(m, n, p)
    assert len(precomputed) == 67
    grid = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    results = [canonical_ex_ante(c, precomputed, m, n) for c in grid]
    for c, r in zip(grid, results):
        assert abs(r.class_probability_sum - 1.0) < 3e-10
        assert abs(r.private - private_discovery_closed_form(c, n, p)) < 3e-10

    distinct_path = [r.private_distinct for r in results]
    assert all(a > b for a, b in zip(distinct_path, distinct_path[1:])), \
        "expected strictly decreasing distinct proposals across the copying grid"
    print("[ok] expected distinct private proposals strictly decreasing in c "
          f"({distinct_path[0]:.6f} at c=0 to {distinct_path[-1]:.6f} at c=0.95)")

    independent = results[0]
    target_frontier = np.array([
        0.383468709731,
        0.527532262620,
        0.604527804729,
        0.670580744436,
        0.735604205526,
        0.794435313543,
        0.836010397698,
        0.859421246199,
    ])
    assert np.max(np.abs(np.asarray(independent.frontier) - target_frontier)) < 5e-10
    assert abs(independent.market - 0.599099252439) < 5e-10
    assert abs(independent.market_quality - 0.344136160273) < 5e-10
    assert independent.consensus < independent.market < independent.private < independent.planner
    assert independent.frontier[5] < independent.private < independent.frontier[6]

    check_blind_control(independent.consensus, m, n)
    check_partial_conformity(precomputed, m, n)
    quality = check_quality_and_redundancy(precomputed, independent, m, n, p)
    check_scaling_limits()
    market_limit = check_market_scaling_limit()
    check_abundant_evidence()
    check_grid_single_crossing(out_dir)

    c_star, crossing_value = find_market_private_crossover(precomputed, n, p)
    assert abs(c_star - 0.7884616566) < 3e-8

    print("\nCanonical independent values")
    print(f"  consensus              {independent.consensus:.12f}")
    print(f"  symmetric market       {independent.market:.12f}")
    print(f"  private clue-following {independent.private:.12f}")
    print(f"  planner portfolio      {independent.planner:.12f}")
    print(f"  E distinct (market)    {independent.market_distinct:.12f}")
    print(f"  E distinct (private)   {independent.private_distinct:.12f}")
    print(f"  crossover c*           {c_star:.12f} at G={crossing_value:.12f}")
    print(f"  market scaling limit   Lambda={market_limit['Lambda']:.12f}, "
          f"G={market_limit['G']:.12f}")

    print("\nCommon-cue table")
    print(" c     consensus    market       private      planner")
    for c, r in zip(grid, results):
        print(f" {c:0.2f}  {r.consensus:0.9f}  {r.market:0.9f}  {r.private:0.9f}  {r.planner:0.9f}")

    generate_figures(out_dir / "figures", independent, grid, results, c_star, crossing_value)
    write_csvs(out_dir, grid, results)
    write_quality_csv(out_dir, independent, quality, n)
    print(f"\n[ok] figures and CSVs written to {out_dir}")
    print("All v8.6 checks passed.")


if __name__ == "__main__":
    main()
