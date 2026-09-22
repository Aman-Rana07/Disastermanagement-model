# ═══════════════════════════════════════════════
# graph.py — Assam district adjacency graph
# for the GCN layer
# ═══════════════════════════════════════════════

import torch
import numpy as np

ASSAM_DISTRICTS = [
    'Baksa', 'Barpeta', 'Biswanath', 'Bongaigaon', 'Cachar',
    'Charaideo', 'Chirang', 'Darrang', 'Dhemaji', 'Dhubri',
    'Dibrugarh', 'Dima Hasao', 'Goalpara', 'Golaghat', 'Hailakandi',
    'Hojai', 'Jorhat', 'Kamrup', 'Kamrup Metropolitan', 'Karbi Anglong',
    'Karimganj', 'Kokrajhar', 'Lakhimpur', 'Majuli', 'Morigaon',
    'Nagaon', 'Nalbari', 'Sivasagar', 'Sonitpur', 'South Salmara',
    'Tinsukia', 'Udalguri', 'West Karbi Anglong', 'Bajali', 'Tamulpur'
]

DISTRICT_TO_IDX = {d: i for i, d in enumerate(ASSAM_DISTRICTS)}

# Edges: shared borders + river connectivity
# Undirected — both directions added automatically in build fn
EDGES = [
    # Brahmaputra main channel (E → W downstream)
    ('Tinsukia',   'Dibrugarh'),
    ('Dibrugarh',  'Charaideo'),
    ('Charaideo',  'Sivasagar'),
    ('Sivasagar',  'Jorhat'),
    ('Jorhat',     'Golaghat'),
    ('Golaghat',   'Nagaon'),
    ('Nagaon',     'Morigaon'),
    ('Morigaon',   'Kamrup Metropolitan'),
    ('Kamrup Metropolitan', 'Kamrup'),
    ('Kamrup',     'Nalbari'),
    ('Nalbari',    'Barpeta'),
    ('Barpeta',    'Bongaigaon'),
    ('Bongaigaon', 'Kokrajhar'),
    ('Kokrajhar',  'Dhubri'),
    ('Dhubri',     'South Salmara'),
    # North bank tributaries
    ('Dhemaji',    'Lakhimpur'),
    ('Lakhimpur',  'Biswanath'),
    ('Biswanath',  'Sonitpur'),
    ('Sonitpur',   'Darrang'),
    ('Darrang',    'Udalguri'),
    ('Udalguri',   'Baksa'),
    ('Baksa',      'Tamulpur'),
    ('Tamulpur',   'Bajali'),
    ('Bajali',     'Chirang'),
    # South bank / Barak valley
    ('Cachar',     'Hailakandi'),
    ('Hailakandi', 'Karimganj'),
    ('Cachar',     'Dima Hasao'),
    ('Dima Hasao', 'Karbi Anglong'),
    ('Karbi Anglong', 'West Karbi Anglong'),
    ('Karbi Anglong', 'Golaghat'),
    ('Karbi Anglong', 'Nagaon'),
    ('Karbi Anglong', 'Hojai'),
    # Cross-valley connections
    ('Goalpara',   'Dhubri'),
    ('Goalpara',   'Kamrup'),
    ('Majuli',     'Jorhat'),
    ('Majuli',     'Lakhimpur'),
    ('Hojai',      'Nagaon'),
    ('Hojai',      'Dima Hasao'),
    ('Darrang',    'Sonitpur'),
    ('Nalbari',    'Kamrup'),
    ('Chirang',    'Bongaigaon'),
    ('Chirang',    'Kokrajhar'),
]


def build_edge_index(device='cpu'):
    """
    Convert district edge list to PyTorch Geometric edge_index.
    Adds both directions (undirected graph).

    Returns:
        edge_index: (2, num_edges) LongTensor
    """
    src, dst = [], []
    for a, b in EDGES:
        if a in DISTRICT_TO_IDX and b in DISTRICT_TO_IDX:
            i, j = DISTRICT_TO_IDX[a], DISTRICT_TO_IDX[b]
            src += [i, j]
            dst += [j, i]

    edge_index = torch.tensor(
        [src, dst],
        dtype=torch.long,
        device=device
    )
    print(f"Graph built: {len(ASSAM_DISTRICTS)} nodes | "
          f"{len(EDGES)} edges ({edge_index.shape[1]} directed)")
    return edge_index


def get_district_name(idx):
    """Return district name from index"""
    return ASSAM_DISTRICTS[idx] if idx < len(ASSAM_DISTRICTS) else 'Unknown'


def get_district_idx(name):
    """Return district index from name"""
    return DISTRICT_TO_IDX.get(name, -1)