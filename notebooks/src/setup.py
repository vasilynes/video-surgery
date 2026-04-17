from . import config
import zipfile
from pathlib import Path
import os
import subprocess
import subprocess
import sys 
from urllib import request
from typing import List

def download(url: str, dst: Path):
    """
    Download a file from URL to destination path.

    url: source.
    dst: destination file path.
    """
    if dst.exists():
        print(f"Already exists: {dst}")
        return 
    print(f"Downloading: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    request.urlretrieve(url, str(dst))
    print('Download complete')

def pip_install(packages: List[str], dst: Path | None = None):
    """
    Make pip download packages.

    packages: list of names.
    dst: directory, in which the subprocess should run.
    """
    if dst is None:
        dst = Path.cwd()
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-q'] + packages,
        cwd=str(dst),
        check=True
    )

def git_clone(rep: str, dst: Path):
    """
    Clone from GitHub repo.

    rep: GitHub repo.
    dst: directory destination.
    """
    subprocess.run(['git', 'clone', rep, str(dst)], check=True)

def clone_grounded_sam2(
        rep: str = config.GROUNDED_SAM_REP, 
        dst: Path = config.COLAB_GROUNDED_SAM_DIR
    ):
    """
    Clone Grounded SAM 2 source code.

    rep: GitHub repo with implementation.
    dst: directory destination.
    """
    if not dst.exists():
        git_clone(rep, dst)
        pip_install(['-e', '.'], dst=dst)   # Install all Grounded SAM 2 python dependencies
        print('Grounded SAM 2 installed')
    else:
        print('Grounded SAM 2 already cloned')

def clone_raft(
        rep: str = config.RAFT_REP, 
        dst:Path = config.COLAB_RAFT_DIR
    ):
    """
    Clone RAFT source code.

    rep: GitHub repo with implementation.
    dst: directory destination.
    """
    if not dst.exists():
        git_clone(rep, dst)
        print('RAFT installed')
    else:
        print('RAFT already cloned')

def prep_env(weights_path: Path = config.WEIGHTS):
    """
    Prepare environment where inference will take place.
    Set environment vars.
    Download packages and source code.

    weights_path: persistent storage to cache Stable Diffusion and 
                  GroundingDINO weights, which must remain
                  across virtual sessions after first
                  download from 🤗 Hugging Face.
    """
    os.environ['HF_HOME'] = str(weights_path / 'huggingface_cache')
    pip_install(['lpips', 'supervision', 'einops'])

    clone_grounded_sam2()
    clone_raft()

    print('\nEnvironment is ready')

def download_models(weights_path: Path = config.WEIGHTS):
    """
    Download SAM 2 and RAFT checkpoints. 
    
    weights_path: pesistent weight storage to store across virtual sessions.
    """
    # GroundingDINO
    # download(
    #     'https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth',
    #     config.WEIGHTS / 'groundingdino_swint_ogc.pth'
    # )

    download(
        'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt',
        weights_path / 'sam2.1_hiera_large.pt'
    )

    download(
        'https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip',
        weights_path / 'raft_models.zip'
    )

    raft_zip = weights_path / 'raft_models.zip'
    raft_dir = weights_path / 'raft'
    if not raft_dir.exists():
        with zipfile.ZipFile(raft_zip, 'r') as z:
            z.extractall(raft_dir)
        print('RAFT extracted')
    else:
        print('RAFT already extracted')

def download_davis(
        dst: Path = config.DATASETS_PATH,
        folder_dir: Path = config.SEQ_DIR
    ):
    """
    Download DAVIS 2017 480p dataset.

    dst: dir, to which it will be downloaded. 
    folder_dir: dir of sequence folder, we download 
                conditioned on its existence.
    """
    dst.mkdir(exist_ok=True)

    if not folder_dir.exists(): 
        print('Downloading DAVIS 2017 480p...')
        zip_path = dst / 'davis.zip'
        download(config.DAVIS_URL, zip_path)
        print('Unzipping DAVIS...')
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(dst)
        zip_path.unlink()
        print('Done')
    else:
        print(f"DAVIS blackswan already on Drive.")
