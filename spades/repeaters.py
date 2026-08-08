"""
MyBot -- 2-player Spades bot.

Design principles (this variant has a 24-card unseen KITTY, unlike normal
4-player spades where every card is always in someone's hand):

1. PROBABILISTIC "BOSS" EVALUATION.
   Both opponent bots (team_name.py, apexbot.py) treat a card as safe to
   cash only if it is a *strict* boss (no higher card of that suit is
   provably dead yet). But only 26 of the 50 cards are ever dealt -- the
   other 24 sit dead in the kitty. So an "unaccounted for" higher card is
   usually NOT in the opponent's hand, it's dead in the kitty. We compute
   the exact hypergeometric probability that the opponent holds none of
   the outstanding higher cards (given how many unseen cards remain and
   how big the opponent's hand currently is) and use that continuous
   probability to drive aggression, instead of a strict boolean. This
   should let us cash tricks the other bots consider "unsafe" and pass
   up tricks they wrongly consider necessary to protect.

2. MONTE CARLO BIDDING (double-dummy style).
   Both opponents use closed-form bid formulas that are miscalibrated in
   various hands (team_name double counts spade honors; apexbot uses a
   single global linear regression with no adjustment for shape). We
   instead sample many plausible opponent hands from the 37 unseen cards
   (13 to opponent, 24 dead) and run a quick double-dummy-ish greedy
   playout for each sample, averaging the tricks we win. This adapts
   automatically to voids, spade concentration, and hand shape.

3. DENIAL PLAY.
   Like apexbot, once our own bid is safely made we don't go fully passive
   -- we keep contesting tricks the opponent still needs, as long as
   contesting doesn't risk our own bag count. team_name never does this
   at all (pure "EVADER" once its own bid is hit), which is exploitable:
   we can often grab their needed tricks for free late in the hand.

4. VOID-AWARE PLAY.
   We track opponent void suits from discards and use them to (a) attack
   with low cards to bleed spades / steal free tricks when contesting,
   and (b) avoid leading into voids while trying to stay safe for nil.
"""

import random
from itertools import combinations

SUITS = ["H", "D", "C", "S"]
SPADES = "S"
ACE = 14
CARDS_PER_HAND = 13
TRICKS_PER_ROUND = 13

FULL_DECK = [(s, r) for s in SUITS for r in range(2, 15) if not (s in ("C", "D") and r == 2)]


# ---------------------------------------------------------------------------
# Shared rules helpers (kept local / dependency-free from engine.py)
# ---------------------------------------------------------------------------
def _legal_cards(hand, lead_card, spades_broken):
    if lead_card is None:
        non_spades = [c for c in hand if c[0] != SPADES]
        if not non_spades:
            return list(hand)
        if spades_broken:
            return list(hand)
        return non_spades
    lead_suit = lead_card[0]
    same_suit = [c for c in hand if c[0] == lead_suit]
    return same_suit if same_suit else list(hand)


def _beats(card, other):
    """True if `card` beats `other` where `other` was played first (i.e. other
    is effectively the lead for comparison purposes)."""
    cs, cr = card
    os_, or_ = other
    if cs == SPADES and os_ != SPADES:
        return True
    if os_ == SPADES and cs != SPADES:
        return False
    if cs != os_:
        return False
    return cr > or_


def _by_suit(hand):
    d = {s: [] for s in SUITS}
    for suit, rank in hand:
        d[suit].append(rank)
    for s in d:
        d[s].sort(reverse=True)
    return d


def _ncr(n, r):
    if r < 0 or r > n:
        return 0
    if r == 0 or r == n:
        return 1
    num = 1
    den = 1
    r = min(r, n - r)
    for i in range(r):
        num *= (n - i)
        den *= (i + 1)
    return num // den


# ---------------------------------------------------------------------------
# Monte Carlo double-dummy-ish playout, used only for BIDDING estimates.
# Both simulated hands are fully known within a sample (the other 24 cards
# not in either hand are the kitty for that sample), so "boss" checks here
# are exact, not probabilistic -- the uncertainty comes from resampling many
# different opponent hands.
# ---------------------------------------------------------------------------
def _sim_lead(hand, other_hand, spades_broken):
    legal = _legal_cards(hand, None, spades_broken)
    by_suit = {}
    for c in legal:
        by_suit.setdefault(c[0], []).append(c)

    for suit, cards in by_suit.items():
        if suit == SPADES:
            continue
        top = max(cards, key=lambda c: c[1])
        if not any(c[0] == suit and c[1] > top[1] for c in other_hand):
            return top

    spade_cards = by_suit.get(SPADES, [])
    if spade_cards:
        top = max(spade_cards, key=lambda c: c[1])
        if not any(c[0] == SPADES and c[1] > top[1] for c in other_hand):
            return top

    non_spade_suits = [s for s in by_suit if s != SPADES]
    if non_spade_suits:
        shortest = min(non_spade_suits, key=lambda s: len(by_suit[s]))
        return min(by_suit[shortest], key=lambda c: c[1])
    return min(legal, key=lambda c: c[1])


def _sim_follow(hand, lead_card, spades_broken):
    legal = _legal_cards(hand, lead_card, spades_broken)
    can_beat = [c for c in legal if _beats(c, lead_card)]
    if can_beat:
        return min(can_beat, key=lambda c: (c[0] == SPADES, c[1]))
    non_spade = [c for c in legal if c[0] != SPADES]
    pool = non_spade if non_spade else legal
    return min(pool, key=lambda c: c[1])


def _sim_lead_duck(hand, spades_broken):
    legal = _legal_cards(hand, None, spades_broken)
    non_spade = [c for c in legal if c[0] != SPADES]
    pool = non_spade if non_spade else legal
    counts = {}
    for c in pool:
        counts[c[0]] = counts.get(c[0], 0) + 1
    longest = max(counts, key=lambda s: counts[s])
    cand = [c for c in pool if c[0] == longest]
    return min(cand, key=lambda c: c[1])


def _sim_follow_duck(hand, lead_card, spades_broken):
    legal = _legal_cards(hand, lead_card, spades_broken)
    losers = [c for c in legal if not _beats(c, lead_card)]
    if losers:
        return max(losers, key=lambda c: c[1])
    return min(legal, key=lambda c: c[1])


def _simulate_playout(my_hand, opp_hand, first_leader, duck_mode=False):
    my_cards = list(my_hand)
    opp_cards = list(opp_hand)
    spades_broken = False
    my_tricks = 0
    leader = first_leader
    for _ in range(TRICKS_PER_ROUND):
        if leader == "me":
            if duck_mode:
                lead = _sim_lead_duck(my_cards, spades_broken)
            else:
                lead = _sim_lead(my_cards, opp_cards, spades_broken)
            my_cards.remove(lead)
            spades_broken = spades_broken or lead[0] == SPADES
            follow = _sim_follow(opp_cards, lead, spades_broken)
            opp_cards.remove(follow)
            spades_broken = spades_broken or follow[0] == SPADES
            if _beats(lead, follow):
                my_tricks += 1
                leader = "me"
            else:
                leader = "opp"
        else:
            lead = _sim_lead(opp_cards, my_cards, spades_broken)
            opp_cards.remove(lead)
            spades_broken = spades_broken or lead[0] == SPADES
            if duck_mode:
                follow = _sim_follow_duck(my_cards, lead, spades_broken)
            else:
                follow = _sim_follow(my_cards, lead, spades_broken)
            my_cards.remove(follow)
            spades_broken = spades_broken or follow[0] == SPADES
            if _beats(lead, follow):
                leader = "opp"
            else:
                my_tricks += 1
                leader = "me"
    return my_tricks


# ---------------------------------------------------------------------------
# MyBot
# ---------------------------------------------------------------------------
class MyBot:
    def __init__(self, bid_samples=100, seed=None, boss_threshold=0.28, denial_slack=1):
        self.bid_samples = bid_samples
        self.round_number = None
        self.known = set()
        self.opp_void = set()
        self.opp_cards_played = 0
        self._rng = random.Random(seed)
        self.boss_threshold = boss_threshold
        self.denial_slack = denial_slack

    # ------------------------------------------------------------------
    def __call__(self, gs):
        if gs.round_number != self.round_number:
            self.round_number = gs.round_number
            self.known = set()
            self.opp_void = set()

        if gs.phase == "play":
            self._update_tracking(gs)

        if gs.phase == "bid":
            return self._decide_bid(gs)
        elif gs.phase == "play":
            return self._decide_card(gs)
        raise ValueError(f"Unknown phase: {gs.phase!r}")

    def _update_tracking(self, gs):
        known = set(gs.your_hand)
        void = set()
        opp_played = 0
        for trick in gs.trick_history:
            plays = trick["plays"]
            lead_suit = plays[0][1][0]
            for player, card in plays:
                known.add(card)
                if player == gs.opponent_name:
                    opp_played += 1
                    if card[0] != lead_suit:
                        void.add(lead_suit)
        if gs.current_trick:
            lead_suit = gs.current_trick[0][1][0]
            for player, card in gs.current_trick:
                known.add(card)
                if player == gs.opponent_name:
                    opp_played += 1
                    if card[0] != lead_suit:
                        void.add(lead_suit)
        self.known = known
        self.opp_void = void
        self.opp_cards_played = opp_played

    # ------------------------------------------------------------------
    # Probability that the opponent holds NO card of `suit` ranked above
    # `rank`, given what's been seen so far this round. Accounts for the
    # dead kitty explicitly via the hypergeometric distribution.
    # ------------------------------------------------------------------
    def _prob_no_higher(self, gs, suit, rank):
        if suit in self.opp_void:
            return 1.0

        m = 0
        for r in range(rank + 1, 15):
            if (suit, r) not in self.known:
                m += 1
        if m == 0:
            return 1.0

        opp_hand_size = CARDS_PER_HAND - self.opp_cards_played
        kitty = gs.kitty_remaining if gs.kitty_remaining is not None else 24
        N = opp_hand_size + kitty
        H = opp_hand_size
        if H <= 0:
            return 1.0
        if N <= 0:
            return 1.0
        if N - m < H:
            return 0.0
        top = _ncr(N - m, H)
        bot = _ncr(N, H)
        if bot == 0:
            return 1.0
        return top / bot

    # ------------------------------------------------------------------
    # Bidding
    # ------------------------------------------------------------------
    def _decide_bid(self, gs):
        hand = gs.your_hand
        unseen = [c for c in FULL_DECK if c not in hand]
        first_leader = "opp" if gs.dealer == gs.your_name else "me"

        samples = self.bid_samples
        total_tricks = 0
        nil_success = 0
        for _ in range(samples):
            opp_hand = self._rng.sample(unseen, CARDS_PER_HAND)
            total_tricks += _simulate_playout(hand, opp_hand, first_leader, duck_mode=False)
            nil_success += 1 if _simulate_playout(hand, opp_hand, first_leader, duck_mode=True) == 0 else 0

        avg_tricks = total_tricks / samples
        p_nil = nil_success / samples

        # Rough card-based sanity filter: never nil with a genuinely strong
        # card (keeps the Monte Carlo estimate from being fooled by a small
        # sample on an obviously bad nil hand).
        max_rank = max(r for _, r in hand)
        spades = sorted((r for s, r in hand if s == SPADES), reverse=True)
        card_safe_for_nil = max_rank <= 11 and len(spades) <= 3 and (not spades or spades[0] <= 11)

        if card_safe_for_nil:
            nil_ev = 200 * p_nil - 100
            normal_ev_proxy = avg_tricks * 9  # conservative proxy: ~9 pts/trick after set risk
            if p_nil >= 0.55 and (nil_ev > normal_ev_proxy or avg_tricks <= 2.0):
                return 0

        bid = int(round(avg_tricks))
        return max(1, min(13, bid))

    # ------------------------------------------------------------------
    # Play
    # ------------------------------------------------------------------
    def _decide_card(self, gs):
        hand = gs.your_hand
        trick = gs.current_trick
        lead_card = trick[0][1] if trick else None
        legal = _legal_cards(hand, lead_card, gs.spades_broken)
        if len(legal) == 1:
            return legal[0]

        my_bid = gs.your_bid or 0
        opp_bid = gs.opponent_bid or 0
        my_tricks = gs.tricks_won.get(gs.your_name, 0)
        opp_tricks = gs.tricks_won.get(gs.opponent_name, 0)

        i_am_nil = my_bid == 0
        i_need_tricks = my_tricks < my_bid

        tricks_played = len(gs.trick_history)
        tricks_remaining = TRICKS_PER_ROUND - tricks_played
        opp_needs_tricks = opp_tricks < opp_bid
        opp_deficit = max(0, opp_bid - opp_tricks)
        opp_slack = tricks_remaining - opp_deficit
        denial_live = opp_needs_tricks and not i_am_nil and opp_slack <= self.denial_slack

        my_bags = gs.your_bags
        overtricks_so_far = max(0, my_tricks - my_bid) if my_bid else 0
        bag_room = 10 - ((my_bags + overtricks_so_far) % 10)
        bag_safe = bag_room > 1

        contested = i_need_tricks or (denial_live and bag_safe)

        if lead_card is None:
            return self._choose_lead(gs, legal, i_am_nil, contested)
        return self._choose_follow(gs, legal, lead_card, i_am_nil, contested)

    def _is_likely_boss(self, gs, card, threshold=0.72):
        suit, rank = card
        return self._prob_no_higher(gs, suit, rank) >= threshold

    def _choose_lead(self, gs, legal, i_am_nil, contested):
        by_suit = _by_suit(legal)

        if i_am_nil:
            candidates = [c for c in legal if c[0] != SPADES] or legal
            safe = [c for c in candidates if c[0] not in self.opp_void]
            pool = safe if safe else candidates
            suit_len = {c[0]: sum(1 for x in gs.your_hand if x[0] == c[0]) for c in pool}
            pool.sort(key=lambda c: (-suit_len[c[0]], c[1]))
            return pool[0]

        if contested:
            # Prefer non-spade cards that are very likely (not just strictly)
            # to win, ranked by win-probability then rank, off-suit first.
            non_spade_candidates = [c for c in legal if c[0] != SPADES]
            scored = [(self._prob_no_higher(gs, c[0], c[1]), c) for c in non_spade_candidates]
            good = [c for p, c in scored if p >= self.boss_threshold]
            if good:
                # among likely winners, cash the cheapest one that's still safe
                # (protect our very best cards for later), but prefer the
                # highest-probability one if margins are close.
                scored_good = sorted(
                    ((self._prob_no_higher(gs, c[0], c[1]), c) for c in good),
                    key=lambda t: (-t[0], -t[1][1]),
                )
                return scored_good[0][1]

            for s in SUITS:
                if s == SPADES or s not in self.opp_void:
                    continue
                ranks = by_suit.get(s, [])
                if ranks:
                    return (s, min(ranks))

            spade_candidates = [c for c in legal if c[0] == SPADES]
            spade_scored = [(self._prob_no_higher(gs, c[0], c[1]), c) for c in spade_candidates]
            spade_good = [c for p, c in spade_scored if p >= self.boss_threshold]
            if spade_good:
                return max(spade_good, key=lambda c: c[1])

            spades = sorted(by_suit.get(SPADES, []), reverse=True)
            if len(spades) >= 4 and spades[0] >= 11:
                return (SPADES, spades[0])

            longest = max(
                (s for s in SUITS if s != SPADES and by_suit.get(s)),
                key=lambda s: len(by_suit[s]),
                default=None,
            )
            if longest:
                return (longest, min(by_suit[longest]))
            return min(legal, key=lambda c: c[1])

        non_spade = [c for c in legal if c[0] != SPADES] or legal
        shortest = min((s for s in SUITS if by_suit.get(s)), key=lambda s: len(by_suit[s]))
        pool = [c for c in non_spade if c[0] == shortest] or non_spade
        return min(pool, key=lambda c: c[1])

    def _choose_follow(self, gs, legal, lead_card, i_am_nil, contested):
        trick = gs.current_trick
        winning_card = trick[0][1]
        for _, c in trick[1:]:
            if _beats(c, winning_card):
                winning_card = c

        can_beat = [c for c in legal if _beats(c, winning_card)]

        if i_am_nil:
            if can_beat:
                losers = [c for c in legal if c not in can_beat]
                if losers:
                    return max(losers, key=lambda c: c[1])
                return min(can_beat, key=lambda c: c[1])
            return max(legal, key=lambda c: c[1])

        if contested:
            if can_beat:
                return min(can_beat, key=lambda c: (c[0] == SPADES, c[1]))
            if any(c[0] == SPADES for c in legal) and legal[0][0] != lead_card[0]:
                spades = sorted((c for c in legal if c[0] == SPADES), key=lambda c: c[1])
                return spades[0]
            return min(legal, key=lambda c: (c[0] == SPADES, c[1]))

        if can_beat and len(can_beat) < len(legal):
            losers = [c for c in legal if c not in can_beat]
            return max(losers, key=lambda c: c[1])
        non_spade = [c for c in legal if c[0] != SPADES] or legal
        return min(non_spade, key=lambda c: c[1])


_global_bot = MyBot()


def nextMove(gs):
    """Tournament entry point."""
    return _global_bot(gs)


def make_bot():
    return MyBot()