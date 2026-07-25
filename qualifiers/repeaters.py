
SUITS = ["H", "D", "C", "S"]
RANKS = list(range(2, 15))

class BotMemory:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.last_face_up = None          
        self.opponent_known_cards = set() 
        self.all_seen_cards = set()       
        self.unseen_cards = set((s, r) for s in SUITS for r in RANKS)
        
        # NEW: Advanced Strategy Trackers
        self.opponent_voids = set() 
        self.last_move = None
        self.we_led_last = False

# Instantiate global memory
memory = BotMemory()


def mark_seen(card):
    """Helper to register a card as seen and remove it from the unseen pool."""
    global memory
    memory.all_seen_cards.add(card)
    if card in memory.unseen_cards:
        memory.unseen_cards.remove(card)


def update_memory(view):
    """Tracks the game state, crosses off cards, and deduces hidden info."""
    global memory
    
    # 1. Detect a new game
    if view.phase == 1 and view.stock_remaining == 26:
        if memory.last_face_up is not None:
            memory.reset()

    # 2. Deduce who won the LAST trick's prize card
    if memory.last_face_up and memory.last_face_up not in view.your_hand:
        memory.opponent_known_cards.add(memory.last_face_up)
        mark_seen(memory.last_face_up)

    # 3. Mark our current hand as seen
    for card in view.your_hand:
        mark_seen(card)
    
    # 4. Mark the current deck prize as seen
    if view.face_up_card:
        mark_seen(view.face_up_card)

    # 5. If the opponent led this trick, mark their card as seen
    if view.current_trick:
        _, opp_card = view.current_trick[0]
        mark_seen(opp_card)
        
        # They played it, so they no longer hold it
        if opp_card in memory.opponent_known_cards:
            memory.opponent_known_cards.remove(opp_card)

    # 6. GHOST VOID DEDUCTOR (Advanced Hidden Info Tracking)
    # If we led the last trick and the opponent won it...
    if view.phase == 1 and memory.we_led_last and memory.last_move:
        if memory.last_face_up and memory.last_face_up not in view.your_hand:
            lead_suit, lead_rank = memory.last_move
            if lead_suit != view.trump_suit:
                # If we led the highest remaining card in that suit and they won, 
                # they MUST have used a Trump. Therefore, they are void in that suit.
                higher_unseen = [c for c in memory.unseen_cards if c[0] == lead_suit and c[1] > lead_rank]
                if not higher_unseen:
                    memory.opponent_voids.add(lead_suit)

    memory.last_face_up = view.face_up_card


def get_legal_moves(view):
    hand = view.your_hand
    if not view.current_trick:
        return list(hand)
    
    _, lead_card = view.current_trick[0]
    lead_suit = lead_card[0]
    same_suit = [card for card in hand if card[0] == lead_suit]
    return same_suit if same_suit else list(hand)


def card_value(card, trump_suit):
    suit, rank = card
    return rank + (100 if suit == trump_suit else 0)


def beats_card(my_card, opponent_card, trump_suit, lead_card):
    my_suit, my_rank = my_card
    opp_suit, opp_rank = opponent_card
    lead_suit = lead_card[0]

    if my_suit == opp_suit:
        return my_rank > opp_rank
    if my_suit == trump_suit and opp_suit != trump_suit:
        return True
    if opp_suit == trump_suit and my_suit != trump_suit:
        return False
    if my_suit != lead_suit:
        return False
    return True


def is_boss_card(card, trump_suit):
    """Check if a card is the absolute highest remaining card in its suit."""
    global memory
    suit, rank = card
    
    # The opponent's potential hand is any card we know they have + any unseen card
    possible_opponent_cards = memory.opponent_known_cards.union(memory.unseen_cards)
    
    # Check if a higher card of the SAME suit exists in their potential hand
    higher_cards_exist = any(
        c for c in possible_opponent_cards 
        if c[0] == suit and c[1] > rank
    )
    
    return not higher_cards_exist


def play_phase_one(view, legal_moves):
    """Phase 1: Win good prizes, intentionally lose bad prizes (Original Strong Logic)."""
    trump = view.trump_suit
    prize = view.face_up_card
    
    prize_is_valuable = False
    if prize:
        prize_suit, prize_rank = prize
        if prize_rank >= 11 or prize_suit == trump:
            prize_is_valuable = True

    if not view.current_trick:
        if prize_is_valuable:
            return max(legal_moves, key=lambda c: card_value(c, trump))
        else:
            return min(legal_moves, key=lambda c: card_value(c, trump))

    _, opp_card = view.current_trick[0]
    winning_moves = [c for c in legal_moves if beats_card(c, opp_card, trump, opp_card)]

    if prize_is_valuable:
        if winning_moves:
            return min(winning_moves, key=lambda c: card_value(c, trump))
        else:
            return min(legal_moves, key=lambda c: card_value(c, trump))
    else:
        losing_moves = [c for c in legal_moves if not beats_card(c, opp_card, trump, opp_card)]
        if losing_moves:
            return min(losing_moves, key=lambda c: card_value(c, trump))
        else:
            return min(winning_moves, key=lambda c: card_value(c, trump))


def play_phase_two(view, legal_moves):
    """Phase 2: Use the Card Counter & Advanced Strategies to find guaranteed wins."""
    global memory
    trump = view.trump_suit

    # WE ARE LEADING
    if not view.current_trick:
        boss_cards = [c for c in legal_moves if is_boss_card(c, trump)]
        
        # --- ADVANCED STRATEGY 1: TRUMP SQUEEZE ---
        our_trumps = [c for c in legal_moves if c[0] == trump]
        opp_possible_trumps = [c for c in memory.unseen_cards.union(memory.opponent_known_cards) if c[0] == trump]
        trump_bosses = [c for c in boss_cards if c[0] == trump]
        
        # If we have more trumps than they could possibly have, and we hold the highest trump... fire it!
        if trump_bosses and len(our_trumps) > len(opp_possible_trumps):
            return max(trump_bosses, key=lambda c: c[1])

        # Prioritize non-trump boss cards first to drain opponent's cards safely
        non_trump_bosses = [c for c in boss_cards if c[0] != trump]
        if non_trump_bosses:
            return min(non_trump_bosses, key=lambda c: c[1]) 
            
        # --- ADVANCED STRATEGY 2: PROMOTION PLAY ---
        # If we have no guaranteed bosses, try to sacrifice a high card to promote a lower one.
        non_trumps = [c for c in legal_moves if c[0] != trump]
        if non_trumps:
            suits_held = {}
            for c in non_trumps:
                suits_held.setdefault(c[0], []).append(c)
            
            # Find suits where we have 2 or more cards (e.g. Q, J)
            promotion_candidates = [max(cards, key=lambda c: c[1]) for cards in suits_held.values() if len(cards) >= 2]
            if promotion_candidates:
                return max(promotion_candidates, key=lambda c: c[1])

        # Fallback: Lead a strong card, but protect non-trump Aces if opponent has Trumps
        best_card = max(legal_moves, key=lambda c: card_value(c, trump))
        if best_card[0] != trump:
            known_trumps = [c for c in memory.opponent_known_cards if c[0] == trump]
            if known_trumps:
                return min(legal_moves, key=lambda c: card_value(c, trump))
                
        return best_card

    # WE ARE FOLLOWING
    _, opp_card = view.current_trick[0]
    winning_moves = [c for c in legal_moves if beats_card(c, opp_card, trump, opp_card)]

    if winning_moves:
        return min(winning_moves, key=lambda c: card_value(c, trump))
    else:
        return min(legal_moves, key=lambda c: card_value(c, trump))


def nextMove(view):
    global memory
    update_memory(view)
    legal_moves = get_legal_moves(view)
    
    # State tracking for the Ghost Void Deductor
    we_are_leading = (len(view.current_trick) == 0)

    if view.phase == 1:
        move = play_phase_one(view, legal_moves)
    else:
        move = play_phase_two(view, legal_moves)
        
    # Save this turn's action to memory so we can deduce things on the next turn
    memory.last_move = move
    memory.we_led_last = we_are_leading
    
    return move