# ═══════════════════════════════════════════════
# config.py — All hyperparameters in one place
# Change values here, everything else updates
# ═══════════════════════════════════════════════

import os

# ── Paths ───────────────────────────────────────
ROOT_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(ROOT_DIR, 'data')
RAW_DIR         = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR   = os.path.join(DATA_DIR, 'processed')
EXPORTS_DIR     = os.path.join(DATA_DIR, 'exports')

FEATURE_STACK   = os.path.join(RAW_DIR, 'assam_2022_feature_stack.tif')
FLOOD_MASK      = os.path.join(RAW_DIR, 'assam_2022_flood_mask.tif')
TERRAIN_STACK   = os.path.join(RAW_DIR, 'assam_terrain_stack_30m.tif')
MODEL_SAVE_PATH = os.path.join(EXPORTS_DIR, 'best_model.pth')

# ── Data ────────────────────────────────────────
PATCH_SIZE      = 256       # pixels × pixels per patch
STRIDE          = 192       # overlap between patches
MIN_VALID_RATIO = 0.7       # drop patches with >30% NoData
IN_CHANNELS     = 22        # 20 satellite + TWI + TRI
N_DISTRICTS     = 35        # Assam districts

# Per-channel normalization bounds (physics-based)
CHANNEL_NAMES = [
    'S1_VV', 'S1_VH',
    'S2_B2', 'S2_B3', 'S2_B4', 'S2_B5',
    'S2_B6', 'S2_B7', 'S2_B8', 'S2_B8A',
    'S2_B11', 'S2_B12',
    'NDVI', 'NDWI', 'MNDWI', 'NBR',
    'elevation', 'slope',
    'TWI', 'TRI',
    'aspect', 'curvature'
]

CHANNEL_BOUNDS = {
    'S1_VV':     (-30,   5),
    'S1_VH':     (-35,   0),
    'S2_B2':     (0,  3000),
    'S2_B3':     (0,  3000),
    'S2_B4':     (0,  3000),
    'S2_B5':     (0,  3000),
    'S2_B6':     (0,  3000),
    'S2_B7':     (0,  3000),
    'S2_B8':     (0,  5000),
    'S2_B8A':    (0,  5000),
    'S2_B11':    (0,  4000),
    'S2_B12':    (0,  3000),
    'NDVI':      (-1,    1),
    'NDWI':      (-1,    1),
    'MNDWI':     (-1,    1),
    'NBR':       (-1,    1),
    'elevation': (0,  2000),
    'slope':     (0,    60),
    'TWI':       (2,    20),
    'TRI':       (0,    60),
    'aspect':    (0,   360),
    'curvature': (-5,    5),
}

# ── Training ────────────────────────────────────
BATCH_SIZE      = 16
EPOCHS          = 50
LEARNING_RATE   = 1e-4
WEIGHT_DECAY    = 1e-4
GRAD_CLIP       = 1.0
NUM_WORKERS     = 4
PIN_MEMORY      = True

# ── Loss ────────────────────────────────────────
FOCAL_ALPHA     = 0.75
FOCAL_GAMMA     = 2.0

# ── Model ────────────────────────────────────────
CNN_OUT_DIM     = 512
GCN_HIDDEN      = 256
GCN_OUT_DIM     = 128
DROPOUT         = 0.3

# ── Spatial CV split (by row blocks) ────────────
# Assam divided into 5 N-S strips
# 0-2 = train, 3 = val, 4 = test
N_BLOCKS        = 5
TRAIN_BLOCKS    = [0, 1, 2]
VAL_BLOCKS      = [3]
TEST_BLOCKS     = [4]

# ── Device ──────────────────────────────────────
DEVICE          = 'cuda'    # change to 'cpu' if no GPU