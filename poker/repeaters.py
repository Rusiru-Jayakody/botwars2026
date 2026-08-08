"""
No-Limit Texas Hold'em (No Blinds) - Adaptive Shark v3
=====================================================
Two new mechanisms beyond v1/v2's equity+opponent-stats framework:

1. RANGE-AWARE EQUITY SIMULATION
   v1/v2 simulated every opponent as holding two fully random cards. That
   is wrong the moment an opponent has bet or raised -- their real range
   is stronger than random, so treating it as random systematically
   overvalues our calling/raising hands against aggression (and
   undervalues folding). We precompute a preflop hand-strength percentile
   table once (all 1,326 starting hands, ranked by a lightweight scoring
   function), and when simulating an opponent who has bet/raised on
   multiple streets *this hand*, we rejection-sample their hole cards
   from the top of that table instead of the full deck. Passive
   opponents (called/checked only, or haven't acted yet) stay fully
   random -- no made-up information there.

2. CROSS-STREET HAND MEMORY
   The engine's PlayerView only exposes the CURRENT street's action list
   -- by the turn, you can no longer see what happened on the flop or
   preflop of the same hand. We keep a small process-local log (the
   player module stays loaded for the whole tournament, so module-level
   state persists call to call) that stitches together every street of
   the hand in progress as we see it. This lets us:
     - detect multi-barrel aggression (feeds the range-narrowing above)
     - know if WE had the betting initiative on the previous street, and
       continuation-bet wider / bluff more when we did (standard range-
       advantage theory), the same way we already widen when the table
       has checked around to us.

Everything else -- the exact-legal raise-to reconstruction, pot-odds
calling, aggression-conditioned risk premium facing raises, threshold-
caller detection for value-bet sizing -- is carried over from v2.
"""

import itertools
import random
from collections import Counter

SUITS = ["H", "D", "C", "S"]
RANKS = list(range(2, 15))
FULL_DECK = [(s, r) for s in SUITS for r in RANKS]
STREET_ORDER = ("preflop", "flop", "turn", "river")

_rng = random.Random()

# ---------------------------------------------------------------------------
# Hand evaluation (5-card, board+hole)
# ---------------------------------------------------------------------------
def _straight_high(ranks):
    unique = sorted(set(ranks), reverse=True)
    if 14 in unique:
        unique.append(1)
    unique = sorted(set(unique), reverse=True)
    for i in range(len(unique) - 4):
        window = unique[i:i + 5]
        if window[0] - window[4] == 4:
            return window[0]
    return None


def _evaluate_five(cards):
    ranks = sorted((c[1] for c in cards), reverse=True)
    suits = [c[0] for c in cards]
    counts = Counter(ranks)
    by_freq = sorted(counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
    is_flush = len(set(suits)) == 1
    straight_high = _straight_high(ranks)

    if is_flush and straight_high:
        return (8, straight_high)
    if by_freq[0][1] == 4:
        return (6, by_freq[0][0], by_freq[1][0])
    if by_freq[0][1] == 3 and by_freq[1][1] == 2:
        return (5, by_freq[0][0], by_freq[1][0])
    if is_flush:
        return (4, *ranks)
    if straight_high:
        return (3, straight_high)
    if by_freq[0][1] == 3:
        kickers = [r for r in ranks if r != by_freq[0][0]]
        return (2, by_freq[0][0], *kickers)
    if by_freq[0][1] == 2 and by_freq[1][1] == 2:
        hi, lo = max(by_freq[0][0], by_freq[1][0]), min(by_freq[0][0], by_freq[1][0])
        kicker = [r for r in ranks if r not in (hi, lo)][0]
        return (1, hi, lo, kicker)
    if by_freq[0][1] == 2:
        kickers = [r for r in ranks if r != by_freq[0][0]]
        return (0, by_freq[0][0], *kickers)
    return (-1, *ranks)


def evaluate_best_hand(hole, board):
    all_cards = list(hole) + list(board)
    if len(all_cards) < 5:
        return None
    best = None
    for combo in itertools.combinations(all_cards, 5):
        score = _evaluate_five(combo)
        if best is None or score > best:
            best = score
    return best


# ---------------------------------------------------------------------------
# Preflop strength percentile table (for range narrowing, not decisions)
# ---------------------------------------------------------------------------
def _preflop_score(c1, c2):
    r1, r2 = c1[1], c2[1]
    hi, lo = max(r1, r2), min(r1, r2)
    if r1 == r2:
        return hi * 2 + 4
    score = hi + lo * 0.5
    if c1[0] == c2[0]:
        score += 2
    gap = hi - lo
    if gap == 1:
        score += 1
    elif gap == 2:
        score += 0.5
    return score


_ALL_SCORES = sorted(_preflop_score(a, b) for a, b in itertools.combinations(FULL_DECK, 2))


def _score_threshold(top_fraction):
    top_fraction = max(0.01, min(1.0, top_fraction))
    idx = min(int(len(_ALL_SCORES) * (1 - top_fraction)), len(_ALL_SCORES) - 1)
    return _ALL_SCORES[idx]


_STRONG_THRESHOLD = _score_threshold(0.15)   # top ~15% of starting hands
_MEDIUM_THRESHOLD = _score_threshold(0.33)   # top ~33%


def _draw_opponent_hole(pool, min_score, max_attempts=5):
    """Rejection-sample 2 cards from pool meeting a strength floor, mutating pool."""
    best_pair = None
    for _ in range(max_attempts):
        pair = _rng.sample(pool, 2)
        if _preflop_score(pair[0], pair[1]) >= min_score:
            best_pair = pair
            break
        best_pair = pair
    pool.remove(best_pair[0])
    pool.remove(best_pair[1])
    return best_pair


# ---------------------------------------------------------------------------
# Monte Carlo equity, range-aware
# ---------------------------------------------------------------------------
def _choose_iterations(n_opponents, any_narrowed):
    budget = 5200 if any_narrowed else 6500
    per_iter_cost = (n_opponents + 1) * 21
    return max(25, min(220, budget // max(1, per_iter_cost)))


def estimate_equity(hole, board, opponent_thresholds, iterations):
    """opponent_thresholds: list, one per opponent; None = fully random,
    otherwise a minimum preflop-score floor to rejection-sample toward."""
    n_opponents = len(opponent_thresholds)
    if n_opponents <= 0:
        return 1.0
    known = set(hole) | set(board)
    base_deck = [c for c in FULL_DECK if c not in known]
    need_board = 5 - len(board)
    if 2 * n_opponents + need_board > len(base_deck):
        return 0.5

    wins = 0.0
    for _ in range(iterations):
        pool = base_deck[:]
        opp_holes = []
        for thresh in opponent_thresholds:
            if thresh is None:
                pair = _rng.sample(pool, 2)
                pool.remove(pair[0])
                pool.remove(pair[1])
            else:
                pair = _draw_opponent_hole(pool, thresh)
            opp_holes.append(pair)
        full_board = board + _rng.sample(pool, need_board)

        my_score = evaluate_best_hand(hole, full_board)
        opp_scores = [evaluate_best_hand(oh, full_board) for oh in opp_holes]
        best_opp = max(opp_scores)

        if my_score > best_opp:
            wins += 1.0
        elif my_score == best_opp:
            tied = 1 + sum(1 for s in opp_scores if s == my_score)
            wins += 1.0 / tied

    return wins / iterations


# ---------------------------------------------------------------------------
# Cross-hand opponent modeling (persists across the whole tournament)
# ---------------------------------------------------------------------------
def analyze_opponents(hand_history, my_name):
    stats = {}

    def get(pname):
        return stats.setdefault(pname, {
            "bets": 0, "raises": 0, "calls": 0, "checks": 0, "folds": 0,
            "call_amounts": [], "fold_amounts": [],
            "faced_bet": 0, "faced_raise": 0,
            "folded_vs_bet": 0, "folded_vs_raise": 0,
            "vpip_hands": set(), "hands_seen": set(),
        })

    for h in hand_history:
        hand_num = h.get("hand_number")
        seat_order = h.get("seat_order", [])
        actions_by_street = h.get("actions", {})
        for street in STREET_ORDER:
            street_actions = actions_by_street.get(street, [])
            current_bet_level = 0
            last_agg_kind = None
            wager = {p: 0 for p in seat_order}
            for player, action in street_actions:
                kind = action[0]
                if player == my_name:
                    if kind in ("bet", "raise"):
                        current_bet_level = action[1]
                        wager[player] = action[1]
                        last_agg_kind = kind
                    elif kind == "call":
                        wager[player] = current_bet_level
                    continue

                to_call_faced = current_bet_level - wager.get(player, 0)
                st = get(player)
                st["hands_seen"].add(hand_num)

                if to_call_faced > 0 and last_agg_kind == "bet":
                    st["faced_bet"] += 1
                elif to_call_faced > 0 and last_agg_kind == "raise":
                    st["faced_raise"] += 1

                if kind == "bet":
                    st["bets"] += 1
                    wager[player] = action[1]
                    current_bet_level = action[1]
                    last_agg_kind = "bet"
                    if street == "preflop":
                        st["vpip_hands"].add(hand_num)
                elif kind == "raise":
                    st["raises"] += 1
                    wager[player] = action[1]
                    current_bet_level = action[1]
                    last_agg_kind = "raise"
                    if street == "preflop":
                        st["vpip_hands"].add(hand_num)
                elif kind == "call":
                    st["calls"] += 1
                    if to_call_faced > 0:
                        st["call_amounts"].append(to_call_faced)
                    wager[player] = current_bet_level
                    if street == "preflop":
                        st["vpip_hands"].add(hand_num)
                elif kind == "check":
                    st["checks"] += 1
                elif kind == "fold":
                    st["folds"] += 1
                    if to_call_faced > 0:
                        st["fold_amounts"].append(to_call_faced)
                        if last_agg_kind == "bet":
                            st["folded_vs_bet"] += 1
                        elif last_agg_kind == "raise":
                            st["folded_vs_raise"] += 1

    for pname, st in stats.items():
        n_hands = max(1, len(st["hands_seen"]))
        total_actions = st["bets"] + st["raises"] + st["calls"] + st["checks"] + st["folds"]
        st["vpip"] = len(st["vpip_hands"]) / n_hands
        st["aggression"] = (st["bets"] + st["raises"]) / max(1, st["calls"] + st["bets"] + st["raises"])
        st["fold_freq"] = st["folds"] / max(1, total_actions)
        amounts = st["call_amounts"]
        if amounts:
            lo, hi = min(amounts), max(amounts)
            st["call_ceiling"] = hi
            st["is_threshold_caller"] = len(amounts) >= 2 and (hi - lo) <= 0.25 * hi
        else:
            st["call_ceiling"] = None
            st["is_threshold_caller"] = False

    return stats


# ---------------------------------------------------------------------------
# Within-hand, cross-street memory (module-level: survives across nextMove
# calls for the life of the tournament process, reset lazily per hand).
# ---------------------------------------------------------------------------
_hand_log = {}  # hand_number -> {street: [(player, action), ...]}


def _sync_hand_log(view):
    log = _hand_log.setdefault(view.hand_number, {})
    log[view.street] = list(view.action_history)
    if len(_hand_log) > 6:
        for k in list(_hand_log.keys()):
            if k < view.hand_number - 3:
                del _hand_log[k]
    return log


def _record_own_action(view, action):
    log = _hand_log.setdefault(view.hand_number, {})
    street_log = log.setdefault(view.street, [])
    street_log.append((view.your_name, action))


def _last_aggressor(action_list):
    level, kind, agg = 0, None, None
    for player, action in action_list:
        if action[0] in ("bet", "raise"):
            level, kind, agg = action[1], action[0], player
    return level, kind, agg


def _streets_aggressed(log, upto_street, player):
    count = 0
    for street in STREET_ORDER:
        if street == upto_street:
            break
        _, kind, agg = _last_aggressor(log.get(street, []))
        if agg == player and kind in ("bet", "raise"):
            count += 1
    return count


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------
def _value_bet_size(pot, stack, threshold_ceilings, strong):
    if threshold_ceilings:
        base = int(min(threshold_ceilings) * 0.95)
    else:
        base = max(3000, int(pot * (0.9 if strong else 0.65)))
    return max(1, min(stack, base))


def _raise_target(min_raise_to, max_raise_to, my_wager, to_call, pot, strong):
    if strong:
        want = my_wager + to_call + int(pot * 0.9)
    else:
        want = min_raise_to
    return max(min_raise_to, min(max_raise_to, want))


# ---------------------------------------------------------------------------
# Main decision
# ---------------------------------------------------------------------------
def _decide(view):
    hole = list(view.your_hole_cards)
    board = list(view.community_cards)
    to_call = view.amount_to_call
    stack = view.your_stack
    pot = view.pot
    name = view.your_name
    street = view.street

    opponents = [p for p in view.seat_order
                 if p != name and view.player_status.get(p) != "folded"]
    n_opp = len(opponents)
    if n_opp == 0:
        return ("check",) if to_call == 0 else ("call",)

    hand_log = _sync_hand_log(view)

    # Build per-opponent range-narrowing thresholds from THIS hand's
    # multi-street aggression pattern.
    opponent_thresholds = []
    for opp in opponents:
        barrels = _streets_aggressed(hand_log, street, opp)
        if street != "preflop":
            # also count aggression already shown on the current street
            _, kind, agg = _last_aggressor(hand_log.get(street, []))
            if agg == opp and kind in ("bet", "raise"):
                barrels += 1
        if barrels >= 2:
            opponent_thresholds.append(_STRONG_THRESHOLD)
        elif barrels == 1:
            opponent_thresholds.append(_MEDIUM_THRESHOLD)
        else:
            opponent_thresholds.append(None)

    any_narrowed = any(t is not None for t in opponent_thresholds)
    iterations = _choose_iterations(n_opp, any_narrowed)
    equity = estimate_equity(hole, board, opponent_thresholds, iterations)
    baseline = 1.0 / (n_opp + 1)
    strength_ratio = equity / baseline if baseline > 0 else equity

    stats = analyze_opponents(view.hand_history, name)
    live_stats = [stats[p] for p in opponents if p in stats]
    avg_fold_freq = (sum(s["fold_freq"] for s in live_stats) / len(live_stats)
                      if live_stats else 0.35)
    avg_aggression = (sum(s["aggression"] for s in live_stats) / len(live_stats)
                       if live_stats else 0.3)
    threshold_ceilings = [s["call_ceiling"] for s in live_stats if s["is_threshold_caller"]]

    cbl, facing_kind, _facing_agg = _last_aggressor(view.action_history)
    my_wager = cbl - to_call
    max_raise_to = my_wager + stack

    checked_around = (len(view.action_history) > 0 and to_call == 0
                       and all(a[0] == "check" for _, a in view.action_history))

    we_have_initiative = False
    if street != "preflop":
        prev_street = STREET_ORDER[STREET_ORDER.index(street) - 1]
        _, pkind, pagg = _last_aggressor(hand_log.get(prev_street, []))
        we_have_initiative = (pagg == name and pkind in ("bet", "raise"))

    # ---------------- Facing no bet: check or open ----------------
    if to_call == 0:
        value_th_std = 1.7
        bluff_prob = 0.18
        eq_floor_std = 0.50

        if checked_around:
            value_th_std = 1.3
            bluff_prob += 0.15
            eq_floor_std = 0.42
        if we_have_initiative:
            bluff_prob += 0.10
            eq_floor_std = min(eq_floor_std, 0.45)
        bluff_prob = min(0.5, bluff_prob)

        action = None
        if strength_ratio >= 3.0 or equity >= 0.68:
            action = ("bet", _value_bet_size(pot, stack, threshold_ceilings, strong=True))
        elif strength_ratio >= value_th_std or equity >= eq_floor_std:
            action = ("bet", _value_bet_size(pot, stack, threshold_ceilings, strong=False))
        elif n_opp <= 3 and avg_fold_freq >= 0.5 and _rng.random() < bluff_prob:
            action = ("bet", max(1, min(stack, max(2500, int(pot * 0.55)))))
        else:
            action = ("check",)

        _record_own_action(view, action)
        return action

    # ---------------- Facing a bet/raise ----------------
    pot_after_call = pot + to_call
    required_equity = to_call / pot_after_call if pot_after_call > 0 else 0.0
    cheap_call = to_call <= max(1500, int(stack * 0.03))

    risk_premium = 0.03
    if facing_kind == "raise":
        risk_premium = 0.06
        if avg_aggression < 0.28:
            risk_premium = 0.11

    can_raise = view.min_raise_to is not None and stack > to_call
    action = None
    if can_raise and (strength_ratio >= 3.3 or equity >= 0.76):
        action = ("raise", _raise_target(view.min_raise_to, max_raise_to, my_wager,
                                          to_call, pot, strong=True))
    elif equity >= required_equity + risk_premium or cheap_call:
        action = ("call",)
    elif (can_raise and n_opp <= 3 and avg_fold_freq >= 0.5
            and equity >= 0.20 and _rng.random() < 0.20):
        action = ("raise", _raise_target(view.min_raise_to, max_raise_to, my_wager,
                                          to_call, pot, strong=False))
    else:
        action = ("fold",)

    _record_own_action(view, action)
    return action


def nextMove(gameState):
    try:
        return _decide(gameState)
    except Exception:
        try:
            return ("check",) if gameState.amount_to_call == 0 else ("fold",)
        except Exception:
            return ("fold",)