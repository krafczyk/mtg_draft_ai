import re
import pandas as pd
from typing import Dict

COLUMN_TYPES = (
    # Metadata
    (re.compile(r"^user_n_games_bucket$"), "Int16"),
    (re.compile(r"^user_game_win_rate_bucket$"), "float32"),
    (re.compile(r"^expansion$"), "str"),
    (re.compile(r"^event_type$"), "str"),
    (re.compile(r"^draft_id$"), "str"),
    (re.compile(r"^draft_time$"), "str"),
    (re.compile(r"^rank$"), "str"),
    # Draft
    (re.compile(r"^event_match_wins$"), "Int8"),
    (re.compile(r"^event_match_losses$"), "Int8"),
    (re.compile(r"^pack_number$"), "Int8"),
    (re.compile(r"^pick_number$"), "Int8"),
    (re.compile(r"^pick$"), "str"),
    (re.compile(r"^pick_maindeck_rate$"), "float32"),
    (re.compile(r"^pick_sideboard_in_rate$"), "float32"),
    (re.compile(r"^pool_.*"), "Int8"),
    (re.compile(r"^pack_card_.*"), "Int8"),
    # Game + Replay
    (re.compile(r"^game_time$"), "str"),
    (re.compile(r"^build_index$"), "Int8"),
    (re.compile(r"^match_number$"), "Int8"),
    (re.compile(r"^game_number$"), "Int8"),
    (re.compile(r"^opp_rank$"), "str"),
    (re.compile(r"^main_colors$"), "str"),
    (re.compile(r"^splash_colors$"), "str"),
    (re.compile(r"^on_play$"), "bool"),
    (re.compile(r"^num_mulligans$"), "Int8"),
    (re.compile(r"^opp_num_mulligans$"), "Int8"),
    (re.compile(r"^opp_colors$"), "str"),
    (re.compile(r"^num_turns$"), "Int8"),
    (re.compile(r"^won$"), "bool"),
    (re.compile(r"^deck_.*"), "Int8"),
    (re.compile(r"^sideboard_.*"), "Int8"),
    # Game
    (re.compile(r"^drawn_.*"), "Int8"),
    (re.compile(r"^tutored_.*"), "Int8"),
    (re.compile(r"^opening_hand_.*"), "Int8"),
    # Replay
    (re.compile(r"^candidate_hand_\d$"), "str"),
    (re.compile(r"^opening_hand$"), "str"),
    (re.compile(r"^user_turn_\d+_cards_drawn$"), "str"),
    (re.compile(r"^((user)|(oppo))_turn_\d+_cards_discarded$"), "str"),
    (re.compile(r"^((user)|(oppo))_turn_\d+_lands_played$"), "str"),
    (re.compile(r"^user_turn_\d+_cards_foretold$"), "str"),
    (re.compile(r"^((user)|(oppo))_turn_\d+_creatures_cast$"), "str"),
    (re.compile(r"^((user)|(oppo))_turn_\d+_non_creatures_cast$"), "str"),
    (re.compile(r"^((user)|(oppo))_turn_\d+_((user)|(oppo))_instants_sorceries_cast$"), "str"),
    (re.compile(r"^((user)|(oppo))_turn_\d+_((user)|(oppo))_abilities$"), "str"),
    (re.compile(r"^((user)|(oppo))_turn_\d+_((user)|(oppo))_cards_learned$"), "str"),
    (re.compile(r"^((user)|(oppo))_turn_\d+_creatures_attacked$"), "str"),
    (re.compile(r"^((user)|(oppo))_turn_\d+_creatures_blocked$"), "str"),
    (re.compile(r"^((user)|(oppo))_turn_\d+_creatures_unblocked$"), "str"),
    (re.compile(r"^((user)|(oppo))_turn_\d+_creatures_blocking$"), "str"),
    (re.compile(r"^((user)|(oppo))_turn_\d+_creatures_blitzed$"), "Int8"),
    (re.compile(r"^((user)|(oppo))_turn_\d+_player_combat_damage_dealt$"), "str"),  # DEPRECATED
    (re.compile(r"^((user)|(oppo))_turn_\d+_((user)|(oppo))_combat_damage_taken$"), "str"),
    (re.compile(r"^((user)|(oppo))_turn_\d+_((user)|(oppo))_creatures_killed_combat$"), "str"),
    (re.compile(r"^((user)|(oppo))_turn_\d+_((user)|(oppo))_creatures_killed_non_combat$"), "str"),
    (re.compile(r"^((user)|(oppo))_turn_\d+_((user)|(oppo))_mana_spent$"), "float32"),
    (re.compile(r"^((user)|(oppo))_turn_\d+_eot_user_cards_in_hand$"), "str"),
    (re.compile(r"^((user)|(oppo))_turn_\d+_eot_oppo_cards_in_hand$"), "float32"),
    (re.compile(r"^((user)|(oppo))_turn_\d+_eot_((user)|(oppo))_lands_in_play$"), "str"),
    (re.compile(r"^((user)|(oppo))_turn_\d+_eot_((user)|(oppo))_creatures_in_play$"), "str"),
    (re.compile(r"^((user)|(oppo))_turn_\d+_eot_((user)|(oppo))_non_creatures_in_play$"), "str"),
    (re.compile(r"^((user)|(oppo))_turn_\d+_eot_((user)|(oppo))_life$"), "float32"),
    (re.compile(r"^((user)|(oppo))_turn_\d+_eot_((user)|(oppo))_poison_counters$"), "float32"),
    (re.compile(r"^user_turn_\d+_cards_tutored$"), "str"),
    (re.compile(r"^oppo_turn_\d+_cards_tutored$"), "Int8"),
    (re.compile(r"^oppo_turn_\d+_cards_drawn_or_tutored$"), "Int8"),
    (re.compile(r"^oppo_turn_\d+_cards_drawn$"), "Int8"),
    (re.compile(r"^oppo_turn_\d+_cards_foretold$"), "Int8"),
    (re.compile(r"^((user)|(oppo))_total_cards_drawn$"), "Int8"),
    (re.compile(r"^((user)|(oppo))_total_cards_discarded$"), "Int8"),
    (re.compile(r"^((user)|(oppo))_total_lands_played$"), "Int8"),
    (re.compile(r"^((user)|(oppo))_total_cards_foretold$"), "Int8"),
    (re.compile(r"^((user)|(oppo))_total_creatures_cast$"), "Int8"),
    (re.compile(r"^((user)|(oppo))_total_creatures_blitzed$"), "Int8"),
    (re.compile(r"^((user)|(oppo))_total_non_creatures_cast$"), "Int8"),
    (re.compile(r"^((user)|(oppo))_total_instants_sorceries_cast$"), "Int8"),
    (re.compile(r"^((user)|(oppo))_total_cards_learned$"), "Int8"),
    (re.compile(r"^((user)|(oppo))_total_mana_spent$"), "Int16"),
    (re.compile(r"^oppo_total_cards_drawn_or_tutored$"), "Int8"),
)


def get_dtypes(filename: str, print_missing: bool = False) -> Dict[str, str]:
    dtypes: Dict[str, str] = {}
    for column in pd.read_csv(filename, nrows=0).columns:
        for regex, column_type in COLUMN_TYPES:
            if regex.match(column):
                dtypes[column] = column_type
                break
        else:
            if print_missing:
                print(f"Could not find an appropriate type for {column}")

    return dtypes
