from collections import defaultdict
from constants import *
from helpers import *


def get_fainted_counts(timeline):
    """Scans a timeline and returns the KOs counters for p1 and p2."""
    p1_fainted_names = set()
    p2_fainted_names = set()
    for turn_data in timeline:
        p1_state = turn_data.get("p1_pokemon_state")
        if p1_state and p1_state.get("status") == "fnt":
            p1_fainted_names.add(p1_state.get("name"))

        p2_state = turn_data.get("p2_pokemon_state")
        if p2_state and p2_state.get("status") == "fnt":
            p2_fainted_names.add(p2_state.get("name"))

    return len(p1_fainted_names), len(p2_fainted_names)


def get_weighted_boost_avg(pokemon_state, weights, total_weight_sum):
    """Calculates the weighted average of boosts for a single Pokémon."""
    if not pokemon_state:
        return 0.0
    boost_dict = pokemon_state.get("boosts", {})
    weighted_sum = 0.0
    for boost_name, weight in weights.items():
        weighted_sum += boost_dict.get(boost_name, 0) * weight
    if total_weight_sum == 0:
        return 0.0
    return weighted_sum / total_weight_sum


def get_type_multiplier(move_type, target_types):
    """Calculates the damage multiplier (e.g., 2x, 0.5x, 0x)."""
    if move_type == "unknown" or not target_types:
        return 1.0
    chart = GEN1_TYPE_CHART.get(move_type.lower(), {})
    multiplier = 1.0
    for target_type in target_types:
        if target_type == "notype":
            continue
        multiplier *= chart.get(target_type.lower(), 1.0)
    return multiplier


def type_map(data):
    """Creates a Pokedex (name -> types) from the data."""
    pokemon_type_map = {}
    for battle in data:
        for pokemon in battle["p1_team_details"]:
            name, types = pokemon["name"], pokemon["types"]
            if name not in pokemon_type_map:
                pokemon_type_map[name] = types
        p2_lead = battle["p2_lead_details"]
        name, types = p2_lead["name"], p2_lead["types"]
        if name not in pokemon_type_map:
            pokemon_type_map[name] = types
    print(f"Pokedex (type_map) created. Found {len(pokemon_type_map)} Pokémon.")
    return pokemon_type_map


def pokemon_win_rate(pokemon_dictionary, battle_data):
    """Creates a stats map (name -> win_rate_float) from the data."""
    pokemon_stats = {name: {"plays": 0, "wins": 0} for name in pokemon_dictionary}
    for battle in battle_data:
        if "player_won" not in battle:
            continue
        did_p1_win = battle["player_won"]
        p1_team_names = {pokemon["name"] for pokemon in battle["p1_team_details"]}
        for name in p1_team_names:
            if name in pokemon_stats:
                pokemon_stats[name]["plays"] += 1
                if did_p1_win:
                    pokemon_stats[name]["wins"] += 1

    for name, stats in pokemon_stats.items():
        if stats["plays"] > 0:
            pokemon_stats[name]["win_rate_float"] = (
                stats["wins"] / stats["plays"]
            ) * 100
        else:
            pokemon_stats[name]["win_rate_float"] = 50.0

    print("Win rate stats (pokemon_stats) created.")
    return pokemon_stats


def build_observed_moveset_map(battle_data):
    """Builds a map of {pokemon: {moves}} and {move: type}."""
    print("--- Building Observed Moveset Map (from Timeline) ---")
    observed_moves_map = defaultdict(set)
    move_to_type_map = {}
    for battle in battle_data:
        timeline = battle.get("battle_timeline", [])
        for turn in timeline:
            for p in ["p1", "p2"]:
                state = turn.get(f"{p}_pokemon_state")
                move = turn.get(f"{p}_move_details")
                if state and move:
                    pokemon_name, move_name, move_type = (
                        state.get("name"),
                        move.get("name"),
                        move.get("type"),
                    )
                    if pokemon_name and move_name:
                        observed_moves_map[pokemon_name].add(move_name)
                    if move_name and move_type:
                        move_to_type_map[move_name] = move_type
    print(f"Mapped {len(move_to_type_map)} unique moves.")
    return dict(observed_moves_map), move_to_type_map
