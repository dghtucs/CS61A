"""Kelly criterion implementations.

Functions:
 - `kelly_fraction_binary(p, b)`: standard Kelly for a bet paying b:1 with win probability p.
 - `kelly_from_returns(returns, probs)`: numeric solver maximizing expected log growth for discrete outcomes.
 - `kelly_general(outcomes)`: convenience wrapper accepting list of (r, p) where r is return per unit bet.

Examples are provided in the `__main__` block.
"""
from typing import List, Sequence, Tuple
import math


def kelly_fraction_binary(p: float, b: float) -> float:
    """Return Kelly fraction for a binary bet paying b:1 with win probability p.

    f* = (b*p - (1-p)) / b

    Args:
        p: probability of winning (0..1)
        b: net odds (profit per unit bet on win). Example: if you win +1 on a $1 bet, b=1.

    Returns:
        Fraction of capital to bet (can be negative if Kelly says short).
    """
    q = 1.0 - p
    return (b * p - q) / b


def _expected_log_growth(f: float, outcomes: Sequence[Tuple[float, float]]) -> float:
    # outcomes: sequence of (r, p) where r is return multiplier minus 1 per unit bet.
    # If you bet fraction f of capital, new capital multiplier for outcome r is (1 + f * r).
    s = 0.0
    for r, p in outcomes:
        val = 1.0 + f * r
        if val <= 0:
            return float('-inf')
        s += p * math.log(val)
    return s


def kelly_from_returns(returns: Sequence[float], probs: Sequence[float], bracket: Tuple[float, float]=None) -> float:
    """Numerically find Kelly fraction maximizing expected log growth for discrete returns.

    returns: sequence of r values (net return per unit bet). Example: +1 means win 1, -1 means lose 1.
    probs: matching probabilities (must sum to ~1).
    bracket: (min_f, max_f) search interval for fraction f.

    Returns best f (float).
    """
    outcomes = list(zip(returns, probs))

    # compute feasible interval where 1 + f * r > 0 for all outcomes
    lo_feasible = -1e9
    hi_feasible = 1e9
    for r in returns:
        if r > 0:
            lo_feasible = max(lo_feasible, -1.0 / r + 1e-12)
        elif r < 0:
            hi_feasible = min(hi_feasible, 1.0 / (-r) - 1e-12)
    if lo_feasible >= hi_feasible:
        raise ValueError('No feasible fraction f where 1 + f*r > 0 for all outcomes')

    if bracket is None:
        lo, hi = lo_feasible, min(hi_feasible, 10.0)
    else:
        lo, hi = bracket
        # intersect with feasible
        lo = max(lo, lo_feasible)
        hi = min(hi, hi_feasible)
    # golden-section search on unimodal function (expected log growth)
    gr = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = _expected_log_growth(c, outcomes)
    fd = _expected_log_growth(d, outcomes)
    for _ in range(80):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = _expected_log_growth(c, outcomes)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = _expected_log_growth(d, outcomes)
        if abs(b - a) < 1e-9:
            break
    # return midpoint of best interval
    return (a + b) / 2


def kelly_general(outcomes: Sequence[Tuple[float, float]]) -> float:
    """Wrapper accepting list of (r, p) where r is net return per unit bet.

    Example: a fair coin that pays +1 with p=0.6 and -1 with p=0.4:
        kelly_general([(1.0, 0.6), (-1.0, 0.4)])
    """
    # basic validation
    s = sum(p for _, p in outcomes)
    if abs(s - 1.0) > 1e-6:
        raise ValueError('Probabilities must sum to 1')
    # choose search bracket: allow shorting up to -0.999 and leveraging up to 10x by default
    return kelly_from_returns([r for r, _ in outcomes], [p for _, p in outcomes], bracket=(-0.999, 10.0))


if __name__ == '__main__':
    # Examples
    print('Kelly binary example: p=0.6, b=1 ->', kelly_fraction_binary(0.6, 1.0))
    # Coin that pays +1 w.p.0.6, -1 w.p.0.4
    print('Kelly general (coin):', kelly_general([(1.0, 0.6), (-1.0, 0.4)]))
    # Example with three outcomes
    outcomes = [(2.0, 0.2), (1.0, 0.5), (-1.0, 0.3)]
    print('Kelly general (3-outcomes):', kelly_general(outcomes))
