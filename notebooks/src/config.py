from pathlib import Path

COLAB_BASE = Path('/content')
COLAB_ROOT = COLAB_BASE / 'drive/MyDrive/video_surgery'
WEIGHTS = COLAB_ROOT / 'weights'

DAVIS_URL = 'https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip'

COLAB_GROUNDED_SAM_DIR = COLAB_BASE / 'Grounded-SAM-2'
GROUNDED_SAM_REP = 'https://github.com/IDEA-Research/Grounded-SAM-2.git'

GDINO_ID = 'IDEA-Research/grounding-dino-tiny'

COLAB_RAFT_DIR = COLAB_BASE / 'RAFT'
RAFT_REP = 'https://github.com/princeton-vl/RAFT.git'

SAM2_CONFIG_DIR = COLAB_GROUNDED_SAM_DIR / 'sam2/configs/sam2.1'
SAM2_CONFIG_FILE = 'sam2.1_hiera_l.yaml' 
SAM2_CKPT_PATH = WEIGHTS / 'sam2.1_hiera_large.pt'


specs = {
    'sequence':           'blackswan',
    'prompt':             'black swan',
    'edit_prompt':        'a white swan with golden wings',
    'fps':                10,
    'resolution':         (480, 854),
    'alpha_attention':    0.7,
    'diffusion_strength': 0.6,
    'diffusion_steps':    20,
}

SEQ      = specs['sequence']
FRAMES   = COLAB_ROOT / f"frames/{SEQ}"
MASKS    = COLAB_ROOT / f"masks/{SEQ}"
FLOWS    = COLAB_ROOT / f"flows/{SEQ}"
LATENTS  = COLAB_ROOT / f"latents/{SEQ}"
EDITED   = COLAB_ROOT / f"edited/{SEQ}"
RESULTS  = COLAB_ROOT / f"results"


for d in [FRAMES, MASKS, FLOWS, LATENTS, EDITED, RESULTS]:
    d.mkdir(exist_ok=True)

DATASETS_PATH = COLAB_ROOT / 'datasets'
DAVIS_ROOT = DATASETS_PATH / 'DAVIS'
SEQ_DIR    = DAVIS_ROOT / f"JPEGImages/480p/{SEQ}"
GT_DIR     = MASKS / 'gt'

ANCHOR_PATH = FRAMES / '0.jpg'
ANCHOR_MASK_PATH = MASKS / '0.npy'