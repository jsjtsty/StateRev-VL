"""Candidate tracking algorithms for the shell game (pre-registered family).

Each function maps (n_cups, init position, swap list) -> predicted final
position. `ground_truth` is the correct composition; the others are
shortcut hypotheses about what a model might compute instead.
"""
from __future__ import annotations


def _apply(s: int, a: int, b: int) -> int:
    return b if s == a else a if s == b else s


def ground_truth(n, init, swaps):
    s = init
    for a, b in swaps:
        s = _apply(s, a, b)
    return s


def last_touch_init(n, init, swaps):
    """PRIMARY shortcut: anchor on the starting POSITION; answer the other end
    of the most recent swap involving that position (init if none)."""
    s = init
    for a, b in swaps:
        if init in (a, b):
            s = b if a == init else a
    return s


def stale(n, init, swaps):
    return init


def first_swap_only(n, init, swaps):
    return _apply(init, *swaps[0]) if swaps else init


def last_swap_on_init(n, init, swaps):
    return _apply(init, *swaps[-1]) if swaps else init


def last_swap_endpoint(n, init, swaps):
    """Destination of the most recent motion, ignoring the ball entirely:
    the position the front-moving cup (first element) arrived at."""
    return swaps[-1][1] if swaps else init


def stop_after(j):
    def f(n, init, swaps):
        return ground_truth(n, init, swaps[:j])
    f.__name__ = f'stop_after_{j}'
    return f


FAMILY = {
    'ground_truth': ground_truth,
    'last_touch_init': last_touch_init,
    'stale': stale,
    'first_swap_only': first_swap_only,
    'last_swap_on_init': last_swap_on_init,
    'last_swap_endpoint': last_swap_endpoint,
    **{f'stop_after_{j}': stop_after(j) for j in (1, 2, 3)},
}
