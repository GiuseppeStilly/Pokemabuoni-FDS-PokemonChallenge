import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

#---Global constants----

DEFAULT_STAT_VALUE = 80 
META_POKEMON_SET = {'cloyster', 'jynx', 'articuno', 'starmie', 'exeggutor'}

STATUS_STRENGTH = {
    "frz": 4, "slp": 3, "par": 2, "tox": 1.5, 
    "brn": 1, "psn": 1, "fnt": 5, "nostatus": 0, "ok": 0
}

BOOST_WEIGHTS = {
    'atk': 1.0, 'spa': 1.0, 'def': 0.75, 'spd': 0.75, 'spe': 1.5  
}
TOTAL_BOOST_WEIGHT_SUM = sum(BOOST_WEIGHTS.values())

GEN1_TYPE_CHART = {
    'normal': {'rock': 0.5, 'ghost': 0},
    'fighting': {'normal': 2, 'poison': 0.5, 'flying': 0.5, 'psychic': 0.5, 'bug': 0.5, 'rock': 2, 'ghost': 0, 'ice': 2},
    'flying': {'fighting': 2, 'bug': 2, 'rock': 0.5, 'electric': 0.5, 'grass': 2},
    'poison': {'poison': 0.5, 'ground': 0.5, 'bug': 2, 'rock': 0.5, 'ghost': 0.5, 'grass': 2},
    'ground': {'poison': 2, 'flying': 0, 'bug': 0.5, 'rock': 2, 'fire': 2, 'electric': 2, 'grass': 0.5},
    'rock': {'fighting': 0.5, 'ground': 0.5, 'flying': 2, 'bug': 2, 'fire': 2, 'ice': 2},
    'bug': {'fighting': 0.5, 'poison': 2, 'flying': 0.5, 'psychic': 0.5, 'ghost': 0.5, 'fire': 0.5, 'grass': 2},
    'ghost': {'normal': 0, 'psychic': 2, 'ghost': 2},
    'fire': {'bug': 2, 'rock': 0.5, 'fire': 0.5, 'water': 0.5, 'grass': 2, 'ice': 2, 'dragon': 0.5},
    'water': {'ground': 2, 'rock': 2, 'fire': 2, 'water': 0.5, 'grass': 0.5, 'dragon': 0.5},
    'grass': {'ground': 2, 'poison': 0.5, 'flying': 0.5, 'bug': 0.5, 'rock': 2, 'fire': 0.5, 'water': 2, 'grass': 0.5, 'dragon': 0.5},
    'electric': {'ground': 0, 'flying': 2, 'water': 2, 'electric': 0.5, 'grass': 0.5, 'dragon': 0.5},
    'psychic': {'fighting': 2, 'poison': 2, 'psychic': 0.5, 'ghost': 1},
    'ice': {'ground': 0.5, 'flying': 2, 'fire': 0.5, 'water': 0.5, 'grass': 2, 'ice': 0.5, 'dragon': 2},
    'dragon': {'dragon': 2},
    'notype': {}, 
    'unknown': {}
}

# Status moves that debilitate the opponent
META_STATUS_MOVES = {
    'hypnosis', 'sing'
}
# "Setup" moves that boost the Pokémon
META_BOOST_MOVES = {'amnesia', 'agility'}
# Recovery moves
META_RECOVERY_MOVES = {'recover', 'rest'}
# "Unique" moves 
META_UNIQUE_MOVES = {'counter', 'explosion'}
