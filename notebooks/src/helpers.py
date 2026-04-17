from . import config
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from hydra.core.global_hydra import GlobalHydra
from hydra import initialize_config_dir
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from pathlib import Path
from typing import Tuple, Union
import torch

def load_groundingdino(
        device: Union[str, torch.device]
    ) -> Tuple[AutoProcessor, AutoModelForZeroShotObjectDetection]:
    """Load GroundingDINO processor and model in eval state."""
    print('Loading GroundingDINO model...')
    processor = AutoProcessor.from_pretrained(config.GDINO_ID)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        config.GDINO_ID
    ).to(device)
    model.eval()
    print('GroundingDINO model ready')

    return processor, model

def _reinit_hydra(config_dir: Path):
    """Clear Hydra state, reinitialize with given configs."""
    GlobalHydra.instance().clear()
    initialize_config_dir(
        config_dir=str(config_dir),
        job_name='sam2'
    )

def load_sam2_image(
        device: Union[str, torch.device],
        config_dir: Path = config.SAM2_CONFIG_DIR,
        config_file: Path = config.SAM2_CONFIG_FILE,
        ckpt_path: Path = config.SAM2_CKPT_PATH
    ) -> SAM2ImagePredictor:
    """
    Load SAM 2 predictor for images.
    
    config_dir: directory containing yaml file for Hydra.
    config_file: yaml file for Hydra.
    ckpt_path: weights for SAM 2.
    """
    print('Loading SAM 2 predictor for images...')
    _reinit_hydra(config_dir)

    model = build_sam2(
        config_file=str(config_file),
        ckpt_path=str(ckpt_path),
        device=device
    )
    predictor = SAM2ImagePredictor(model)
    print('SAM 2 image predictor ready') 

    return predictor

def load_sam2_video(  
        device: Union[str, torch.device],      
        config_dir: Path = config.SAM2_CONFIG_DIR,
        config_file: Path = config.SAM2_CONFIG_FILE,
        ckpt_path: Path = config.SAM2_CKPT_PATH
    ) -> SAM2VideoPredictor:
    """
    Load SAM 2 predictor for videos.
    
    config_dir: directory containing yaml file for Hydra.
    config_file: yaml file for Hydra.
    ckpt_path: weights for SAM 2.
    """
    print('Loading SAM 2 predictor for videos...')
    _reinit_hydra(config_dir)

    predictor = build_sam2_video_predictor(
        config_file=str(config_file),
        ckpt_path=str(ckpt_path),
        device=device
    )
    print('SAM 2 video predictor ready')

    return predictor

