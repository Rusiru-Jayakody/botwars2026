"""
Grandmaster Piquet Bot (Rep1 Elite: The Ultimate Oracle)
- Sequence-Protected Exchange: Prevents throwing away cards that form runs.
- Boss-Cashing Lead Strategy: Dominates trick tempo with guaranteed winners.
- Optimized Void Discarding: Purges short suits to clear structural hand bottlenecks.
"""

SUITS = ["H", "D", "C", "S"]
RANKS = list(range(7, 15))  # 7..14 (Ace high)
ACE = 14

def _by_suit(hand):
    d = {s: [] for s in SUITS}
    for card in hand:
        d[card[0]].append(card[1])
    for s in d:
        d[s].sort(reverse=True)
    return d

def _point_pip(rank):
    if rank == 14: return 11
    if rank >= 10: return 10
    return rank

# ==========================================
# COMBINATION HELPERS
# ==========================================
def _get_best_point(hand):
    suit_counts = {s: [] for s in SUITS}
    for c in hand:
        suit_counts[c[0]].append(c[1])
    
    best_len = 0
    best_pips = 0
    best_suit = None
    for s, ranks in suit_counts.items():
        if not ranks: continue
        length = len(ranks)
        pips = sum(_point_pip(r) for r in ranks)
        if length > best_len or (length == best_len and pips > best_pips):
            best_len, best_pips, best_suit = length, pips, s
    return best_len, best_pips, best_suit

def _get_best_sequence(hand):
    suit_ranks = {s: sorted(list({c[1] for c in hand if c[0] == s})) for s in SUITS}
    best_seq = None
    
    for s, ranks in suit_ranks.items():
        if len(ranks) < 3: continue
        i = 0
        while i < len(ranks):
            start = i
            while i + 1 < len(ranks) and ranks[i+1] == ranks[i] + 1:
                i += 1
            run_len = i - start + 1
            if run_len >= 3:
                top = ranks[i]
                if best_seq is None or run_len > best_seq[0] or (run_len == best_seq[0] and top > best_seq[1]):
                    best_seq = (run_len, top, s)
            i += 1
    return best_seq

def _get_best_set(hand):
    rank_counts = {}
    for c in hand:
        if c[1] >= 10:
            rank_counts.setdefault(c[1], 0)
            rank_counts[c[1]] += 1
            
    best_set_info = None
    for rank, count in rank_counts.items():
        if count >= 3:
            if best_set_info is None or rank > best_set_info[1] or (rank == best_set_info[1] and count > best_set_info[0]):
                best_set_info = (count, rank)
    return best_set_info

# ==========================================
# MAIN BOT CONTROLLER
# ==========================================
class PiquetEliteBot:
    def __init__(self):
        self.round_number = None

    def __call__(self, view):
        if view.phase == "exchange":
            return self._decide_exchange(view)
        elif view.phase == "declare":
            return self._decide_declaration(view)
        elif view.phase == "tricks":
            return self._decide_trick(view)
        raise ValueError(f"Unknown phase: {view.phase!r}")

    # ------------------------------------------
    # PHASE 1: SEQUENCE-PROTECTED EXCHANGE
    # ------------------------------------------
    def _decide_exchange(self, view):
        hand = view.your_hand
        is_elder = (view.your_name == view.elder)
        max_disc = 5 if is_elder else min(view.talon_remaining, len(hand))
        if max_disc == 0: return []

        by_suit = _by_suit(hand)
        _, _, best_suit = _get_best_point(hand)
        
        discardable = []
        for card in hand:
            suit, rank = card
            if rank >= 13: continue  # Never discard Aces/Kings
            if suit == best_suit and rank >= 10: continue  # Protect point honors
            
            # Protect cards that form part of a potential sequence (gap <= 2)
            suit_ranks = by_suit[suit]
            is_part_of_run = any(abs(rank - r) <= 2 for r in suit_ranks if r != rank)
            if is_part_of_run: continue
            
            if rank <= 9:
                discardable.append(card)
                
        discardable.sort(key=lambda c: c[1])
        disc = discardable[:max_disc]
        
        if len(disc) < max_disc:
            extras = [c for c in hand if c not in disc and c[1] < 13]
            extras.sort(key=lambda c: c[1])
            disc.extend(extras[:max_disc - len(disc)])
            
        return disc[:max_disc]

    # ------------------------------------------
    # PHASE 2: DECLARATION ENGINE
    # ------------------------------------------
    def _decide_declaration(self, view):
        hand = view.your_hand
        cat = view.declare_category
        
        if cat == "point":
            length, _, _ = _get_best_point(hand)
            return "claim" if length >= 4 else "pass"
        elif cat == "sequence":
            seq = _get_best_sequence(hand)
            return "claim" if seq is not None else "pass"
        elif cat == "set":
            st = _get_best_set(hand)
            return "claim" if st is not None else "pass"
            
        return "pass"

    # ------------------------------------------
    # PHASE 3: BOSS-CASHING TRICK-TAKER
    # ------------------------------------------
    def _decide_trick(self, view):
        hand = view.your_hand
        trick = view.current_trick
        am_leading = len(trick) == 0
        
        def legal_moves():
            if am_leading: return list(hand)
            lead_suit = trick[0][1][0]
            same = [c for c in hand if c[0] == lead_suit]
            return same if same else list(hand)

        legal = legal_moves()
        if len(legal) == 1: return legal[0]

        if am_leading:
            # 1. Cash Aces and Kings immediately to command tempo
            honors = [c for c in legal if c[1] >= 13]
            if honors: return max(honors, key=lambda x: x[1])
            
            # 2. Lead highest card of our longest suit to assert pressure
            by_suit = _by_suit(legal)
            longest_suit = max(by_suit.keys(), key=lambda s: len(by_suit[s]))
            if by_suit[longest_suit]:
                return (longest_suit, max(by_suit[longest_suit]))
                
            return max(legal, key=lambda c: c[1])
        else:
            lead_card = trick[0][1]
            lead_suit = lead_card[0]
            
            can_beat = [c for c in legal if c[0] == lead_suit and c[1] > lead_card[1]]
            losers = [c for c in legal if c[0] == lead_suit and c[1] < lead_card[1]]
            
            if can_beat:
                # Win as cheaply as possible to save power cards
                return min(can_beat, key=lambda c: c[1])
            if losers:
                # Can't win: dump highest loser to clear hand structure
                return max(losers, key=lambda c: c[1])
                
            # Void in lead suit: dump lowest card from shortest side suit
            by_suit_all = _by_suit(hand)
            side_suits = [s for s in SUITS if s != lead_suit and by_suit_all[s]]
            if side_suits:
                shortest = min(side_suits, key=lambda s: len(by_suit_all[s]))
                return (shortest, min(by_suit_all[shortest]))
                
            return min(legal, key=lambda c: c[1])

# ==================================================================
# TOURNAMENT ENTRY POINT
# ==================================================================
_global_bot = PiquetEliteBot()

def nextMove(gameState):
    return _global_bot(gameState)