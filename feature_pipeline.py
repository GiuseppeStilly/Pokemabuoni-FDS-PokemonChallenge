import pandas as pd
import numpy as np
from collections import defaultdict

from constants import *
from helpers import *


def compute_choice_features(battle):
    """Deduces switches/boosts. Returns all 6 features."""
    timeline = battle.get("battle_timeline", [])
    if len(timeline) <= 1:
        return {
            "p1_switch_count": 0,
            "p2_switch_count": 0,
            "switch_advantage": 0,
            "p1_boost_move_count": 0,
            "p2_boost_move_count": 0,
            "boost_move_advantage": 0,
        }

    p1_switches, p2_switches, p1_boost_moves, p2_boost_moves = 0, 0, 0, 0
    for i in range(1, len(timeline)):
        prev_turn, curr_turn = timeline[i - 1], timeline[i]
        prev_state_p1, curr_state_p1 = prev_turn.get(
            "p1_pokemon_state", {}
        ), curr_turn.get("p1_pokemon_state", {})
        prev_state_p2, curr_state_p2 = prev_turn.get(
            "p2_pokemon_state", {}
        ), curr_turn.get("p2_pokemon_state", {})
        if not (prev_state_p1 and curr_state_p1 and prev_state_p2 and curr_state_p2):
            continue
        if (
            prev_state_p1.get("name") != curr_state_p1.get("name")
            and prev_state_p1.get("status") != "fnt"
        ):
            p1_switches += 1
        if (
            prev_state_p2.get("name") != curr_state_p2.get("name")
            and prev_state_p2.get("status") != "fnt"
        ):
            p2_switches += 1
        if sum(curr_state_p1.get("boosts", {}).values()) > sum(
            prev_state_p1.get("boosts", {}).values()
        ):
            p1_boost_moves += 1
        if sum(curr_state_p2.get("boosts", {}).values()) > sum(
            prev_state_p2.get("boosts", {}).values()
        ):
            p2_boost_moves += 1
    return {
        "switch_advantage": p1_switches - p2_switches,
        "boost_move_advantage": p1_boost_moves - p2_boost_moves,
    }


def compute_tactical_features(battle):
    """Evaluates tactical features (simple average version)."""
    timeline = battle.get("battle_timeline", [])
    turns = len(timeline)
    if turns <= 1:
        return {
            "hp_delta_balance": 0.0,
            "status_balance": 0.0,
            "momentum_stability": 0.0,
            "avg_damage_turn": 0.0,
        }

    hp_loss_p1, hp_loss_p2, status_score_p1, status_score_p2 = 0, 0, 0, 0
    damage_ratios, total_avg_damage = [], 0.0

    for i in range(1, turns):
        prev, curr = timeline[i - 1], timeline[i]
        prev_state_p1, curr_state_p1 = prev.get("p1_pokemon_state", {}), curr.get(
            "p1_pokemon_state", {}
        )
        prev_state_p2, curr_state_p2 = prev.get("p2_pokemon_state", {}), curr.get(
            "p2_pokemon_state", {}
        )
        if not (prev_state_p1 and curr_state_p1 and prev_state_p2 and curr_state_p2):
            continue

        hp_prev_p1, hp_curr_p1 = prev_state_p1["hp_pct"], curr_state_p1["hp_pct"]
        hp_prev_p2, hp_curr_p2 = prev_state_p2["hp_pct"], curr_state_p2["hp_pct"]
        dmg_p1 = max(0, hp_prev_p2 - hp_curr_p2)
        dmg_p2 = max(0, hp_prev_p1 - hp_curr_p1)
        hp_loss_p1, hp_loss_p2 = hp_loss_p1 + dmg_p2, hp_loss_p2 + dmg_p1
        damage_ratios.append(dmg_p1 - dmg_p2)

        s1, s2 = curr_state_p1["status"].lower(), curr_state_p2["status"].lower()
        status_score_p1 += STATUS_STRENGTH.get(s1, 0)
        status_score_p2 += STATUS_STRENGTH.get(s2, 0)
        total_avg_damage += (dmg_p1 + dmg_p2) / 2.0

    if not damage_ratios or np.mean(np.abs(damage_ratios)) < 1e-5:
        momentum_stability = 0.0
    else:
        momentum_stability = 1 - (
            np.var(damage_ratios) / (np.mean(np.abs(damage_ratios)) + 1e-5)
        )

    return {
        "hp_delta_balance": (hp_loss_p2 - hp_loss_p1) / turns,
        "status_balance": status_score_p1 - status_score_p2,
        "momentum_stability": momentum_stability,
        "avg_damage_turn": total_avg_damage / turns,
    }


def compute_endgame_features(
    battle_data, pokemon_type_map, observed_moves_map, move_to_type_map
):
    """Calculates final snapshot features: remaining counts, total HP, and type advantage."""
    features = {}
    timeline = battle_data.get("battle_timeline", [])
    if not timeline:
        return {
            "remaining_pokemon_advantage": 0,
            "total_hp_advantage": 0,
            "t30_type_advantage": 0,
        }

    p1_team_map = {p["name"]: p for p in battle_data.get("p1_team_details", [])}
    p2_revealed_names = {battle_data["p2_lead_details"]["name"]}
    last_known_state = {}

    for turn in timeline:
        p1_state = turn.get("p1_pokemon_state")
        if p1_state and p1_state.get("name"):
            last_known_state[p1_state["name"]] = p1_state
        p2_state = turn.get("p2_pokemon_state")
        if p2_state and p2_state.get("name"):
            last_known_state[p2_state["name"]] = p2_state
            p2_revealed_names.add(p2_state["name"])

    p1_remaining_squad, p2_remaining_squad = [], []
    p1_total_hp, p2_total_hp = 0.0, 0.0

    for name, details in p1_team_map.items():
        state = last_known_state.get(name)
        hp_pct = 1.0
        is_fainted = False
        if state is None:
            pass
        else:
            is_fainted = state.get("status") == "fnt" or state.get("hp_pct", 0.0) == 0.0
            hp_pct = state.get("hp_pct", 0.0)
        if not is_fainted:
            p1_total_hp += hp_pct
            p1_remaining_squad.append(
                {
                    "name": name,
                    "types": details.get("types", []),
                    "hp_pct": hp_pct,
                    "moves": observed_moves_map.get(name, set()),
                }
            )

    for name in p2_revealed_names:
        state = last_known_state.get(name)
        if state and state.get("status") != "fnt" and state.get("hp_pct", 0.0) > 0.0:
            p2_total_hp += state.get("hp_pct")
            p2_remaining_squad.append(
                {
                    "name": name,
                    "types": pokemon_type_map.get(name, []),
                    "hp_pct": state.get("hp_pct"),
                    "moves": observed_moves_map.get(name, set()),
                }
            )

    p1_remaining_count = len(p1_remaining_squad)
    p2_remaining_count = len(p2_remaining_squad)

    p1_total_hp = sum(p["hp_pct"] for p in p1_remaining_squad)
    p2_total_hp = sum(p["hp_pct"] for p in p2_remaining_squad)

    features["remaining_pokemon_advantage"] = len(p1_remaining_squad) - len(
        p2_remaining_squad
    )
    features["total_hp_advantage"] = p1_total_hp - p2_total_hp

    if p1_remaining_count == 0 or p2_remaining_count == 0:
        features["t30_type_advantage"] = 0.0
        return features

    total_type_advantage = 0.0
    for p1_mon in p1_remaining_squad:
        for p2_mon in p2_remaining_squad:
            best_p1_offense, best_p2_offense = 1.0, 1.0
            for move_name in p1_mon["moves"]:
                move_type = move_to_type_map.get(move_name, "unknown")
                best_p1_offense = max(
                    best_p1_offense, get_type_multiplier(move_type, p2_mon["types"])
                )
            for move_name in p2_mon["moves"]:
                move_type = move_to_type_map.get(move_name, "unknown")
                best_p2_offense = max(
                    best_p2_offense, get_type_multiplier(move_type, p1_mon["types"])
                )
            total_type_advantage += best_p1_offense - best_p2_offense

    features["t30_type_advantage"] = total_type_advantage
    return features


def compute_strategic_features(battle, pokedex):
    """Wrapper function for strategic and tactical features."""
    p1_team = battle["p1_team_details"]
    features = {}
    team_types = set(sum([pokedex.get(p["name"], []) for p in p1_team], []))
    features["p1_team_type_diversity"] = len(team_types)
    tactical = compute_tactical_features(battle)
    features.update(tactical)
    return features


def process_battle_base(battle_data, pokedex, stats_map):
    """
    Creates the base features (T0 strategy, T30 snapshot).
    (Bug-fixed, Restored to 40+ features)
    """
    features = {}
    # Safe Target (handles submission)
    features["player_won"] = battle_data.get("player_won")
    if features["player_won"] is not None:
        features["player_won"] = 1 if features["player_won"] else 0

    p1_team = battle_data["p1_team_details"]
    timeline = battle_data["battle_timeline"]

    p2_revealed_names = set()
    for turn in timeline:
        p2_state = turn.get("p2_pokemon_state")
        if p2_state:
            pokemon_name = p2_state.get("name")
            if pokemon_name:
                p2_revealed_names.add(pokemon_name)
    p2_revealed_names.add(battle_data["p2_lead_details"]["name"])

    p2_team_stats_list = []
    for name in p2_revealed_names:
        p2_team_stats_list.append(stats_map.get(name, {}))

    # --- 3. Strategic Features (T0) ---
    p1_win_rates = [
        stats_map.get(p["name"], {}).get("win_rate_float", 50.0) for p in p1_team
    ]
    # features['p1_team_avg_win_rate'] = np.mean(p1_win_rates)

    meta_count_p1 = sum(1 for p in p1_team if p["name"] in META_POKEMON_SET)
    p1_meta_score = meta_count_p1 / 6.0
    # features['p1_team_meta_score'] = p1_meta_score

    p2_win_rates = [p.get("win_rate_float", 50.0) for p in p2_team_stats_list]
    # features['p2_team_avg_win_rate'] = np.mean(p2_win_rates)

    # Corrected P2 Meta Score logic
    meta_count_p2 = sum(1 for name in p2_revealed_names if name in META_POKEMON_SET)
    p2_meta_score = meta_count_p2 / len(p2_revealed_names) if p2_revealed_names else 0.0
    # features['p2_team_meta_score'] = p2_meta_score

    features["avg_win_rate_advantage"] = np.mean(p1_win_rates) - np.mean(p2_win_rates)
    features["meta_score_advantage"] = p1_meta_score - p2_meta_score
    # --- 4. Base Stat Advantages (Restored 6 features) ---
    stats_to_avg = [
        "base_hp",
        "base_atk",
        "base_def",
        "base_spa",
        "base_spd",
        "base_spe",
    ]
    for stat in stats_to_avg:
        p1_avg_stat = np.mean([p.get(stat, DEFAULT_STAT_VALUE) for p in p1_team])
        p2_avg_stat = np.mean(
            [p.get(stat, DEFAULT_STAT_VALUE) for p in p2_team_stats_list]
        )
        features[f"p1_{stat}_advantage"] = p1_avg_stat - p2_avg_stat

    p1_ko_count, p2_ko_count = get_fainted_counts(timeline)
    features["p1_total_fainted"] = p1_ko_count
    features["p2_total_fainted"] = p2_ko_count

    # --- 6. T30 Snapshot Features (Bug-fixed) ---
    last_turn_data = timeline[29]
    p1_last_state = last_turn_data.get("p1_pokemon_state")
    p2_last_state = last_turn_data.get("p2_pokemon_state")

    features["t30_p1_hp_pct"] = (
        p1_last_state.get("hp_pct", 0.0) if p1_last_state else 0.0
    )
    features["t30_p2_hp_pct"] = (
        p2_last_state.get("hp_pct", 0.0) if p2_last_state else 0.0
    )

    # --- 7. T30 Status (Restored text for get_dummies) ---
    features["t30_p1_status"] = (
        p1_last_state.get("status", "ok") if p1_last_state else "ok"
    )
    features["t30_p2_status"] = (
        p2_last_state.get("status", "ok") if p2_last_state else "ok"
    )

    p1_weighted_boost_avg = get_weighted_boost_avg(
        p1_last_state, BOOST_WEIGHTS, TOTAL_BOOST_WEIGHT_SUM
    )
    p2_weighted_boost_avg = get_weighted_boost_avg(
        p2_last_state, BOOST_WEIGHTS, TOTAL_BOOST_WEIGHT_SUM
    )
    features["t30_weighted_boost_advantage"] = (
        p1_weighted_boost_avg - p2_weighted_boost_avg
    )

    return features


def compute_moveset_features(battle):
    """
    Scans the *entire timeline* to COUNT how many times
    P1 and P2 USED moves of a specific meta category.

    Returns 10 features (8 counts, 2 advantage).
    """
    timeline = battle.get("battle_timeline", [])

    # --- 1. Initialize Counters ---
    p1_status_count = 0
    p1_boost_count = 0
    p1_recovery_count = 0
    p1_unique_count = 0

    p2_status_count = 0
    p2_boost_count = 0
    p2_recovery_count = 0
    p2_unique_count = 0

    # --- 2. Iterate over every turn ---
    for turn in timeline:
        # --- Check P1's move ---
        p1_move = turn.get("p1_move_details")
        if p1_move and p1_move.get("name"):
            move_name = p1_move["name"].lower()

            # Check against each category
            if move_name in META_STATUS_MOVES:
                p1_status_count += 1
            elif move_name in META_BOOST_MOVES:
                p1_boost_count += 1
            elif move_name in META_RECOVERY_MOVES:
                p1_recovery_count += 1
            elif move_name in META_UNIQUE_MOVES:
                p1_unique_count += 1

        # --- Check P2's move ---
        p2_move = turn.get("p2_move_details")
        if p2_move and p2_move.get("name"):
            move_name = p2_move["name"].lower()

            # Check against each category
            if move_name in META_STATUS_MOVES:
                p2_status_count += 1
            elif move_name in META_BOOST_MOVES:
                p2_boost_count += 1
            elif move_name in META_RECOVERY_MOVES:
                p2_recovery_count += 1
            elif move_name in META_UNIQUE_MOVES:
                p2_unique_count += 1

    # --- 3. Return the counts and advantages ---
    features = {
        "p1_used_recovery_move": p1_recovery_count,
        "p1_used_unique_move": p1_unique_count,
        "p2_used_recovery_move": p2_recovery_count,
        "p2_used_unique_move": p2_unique_count,
    }

    features["status_move_advantage"] = p1_status_count - p2_status_count
    features["boost_move_advantage_strategy"] = p1_boost_count - p2_boost_count

    return features


# --- Define global maps (will be populated by training mode) ---
pokemon_type_map = {}
pokemon_stats = {}
observed_moves_map = {}
move_to_type_map = {}


def create_dataset(data, is_training=True):
    """
    Runs the full pre-processing and feature engineering pipeline.
    Handles both training and submission data.

    Parameters:
    - data (list): The raw battle data (train_data or test_data)
    - is_training (bool): If True, creates maps and splits X/y.
                      If False, reuses maps and returns X_test and IDs.

    Returns:
    - (X_processed, y) if is_training=True
    - (X_processed_test, battle_ids_list) if is_training=False
    """

    global pokemon_type_map, pokemon_stats, observed_moves_map, move_to_type_map

    # --- 1. Pre-processing (Build mapping dicts) ---
    if is_training:
        print("Running pre-processing (Training Mode)...")
        # Populate the global maps using training data
        pokemon_type_map = type_map(data)
        pokemon_stats = pokemon_win_rate(pokemon_type_map, data)
        observed_moves_map, move_to_type_map = build_observed_moveset_map(data)
    else:
        print("Running pre-processing (Test Mode)...")
        # Check if maps exist (created during training)
        if "pokemon_type_map" not in globals() or "observed_moves_map" not in globals():
            raise ValueError(
                "Maps not found. Run create_dataset(train_data, is_training=True) first."
            )

    # --- 2. Main Feature Engineering Loop ---
    print(f"\nStarting Main Feature Engineering Loop for {len(data)} battles...")
    processed_data_list = []
    battle_ids_list = []  # Needed for submission

    for battle in data:
        # Pass the global maps to each function
        features = process_battle_base(battle, pokemon_type_map, pokemon_stats)
        strategic_features = compute_strategic_features(battle, pokemon_type_map)
        features.update(strategic_features)
        choice_features = compute_choice_features(battle)
        features.update(choice_features)
        endgame_features = compute_endgame_features(
            battle, pokemon_type_map, observed_moves_map, move_to_type_map
        )
        features.update(endgame_features)

        moveset_features = compute_moveset_features(battle)
        features.update(moveset_features)

        if "battle_id" in battle:
            features["battle_id"] = battle["battle_id"]
            if not is_training:
                battle_ids_list.append(battle["battle_id"])

        processed_data_list.append(features)

    print("Feature engineering complete.")

    # --- 3. Final DataFrame Creation ---
    df = pd.DataFrame(processed_data_list)

    if is_training:
        df_clean = df.dropna(subset=["player_won"])
        X = df_clean.drop(["player_won", "battle_id"], axis=1, errors="ignore")
        y = df_clean["player_won"].astype(int)

        categorical_cols = ["t30_p1_status", "t30_p2_status"]
        X_processed = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

        print(
            f"\nFinal Training DataFrame created with {X_processed.shape[1]} features."
        )
        return X_processed, y

    else:
        # For test data
        X_test = df.drop(["player_won", "battle_id"], axis=1, errors="ignore")
        categorical_cols = ["t30_p1_status", "t30_p2_status"]
        X_processed_test = pd.get_dummies(
            X_test, columns=categorical_cols, drop_first=True
        )

        print(
            f"\nFinal Test DataFrame created with {X_processed_test.shape[1]} features."
        )
        return X_processed_test, battle_ids_list
