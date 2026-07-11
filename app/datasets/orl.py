import urllib.request
import zipfile
from pathlib import Path
from app.config import BASE_DIR

DATASET_DIR = BASE_DIR / "datasets" / "orl"
ORL_URL = "https://www.cl.cam.ac.uk/research/dtg/attarchive/pub/data/att_faces.zip"


def download_orl(force=False):
    """Download and extract the ORL face dataset. Skips if already present."""
    if DATASET_DIR.exists() and any(DATASET_DIR.iterdir()) and not force:
        print(f"ORL dataset already exists at {DATASET_DIR}")
        return DATASET_DIR

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATASET_DIR / "att_faces.zip"

    print(f"Downloading ORL dataset from {ORL_URL}...")
    try:
        urllib.request.urlretrieve(ORL_URL, zip_path)
    except Exception as e:
        print(f"Download failed: {e}")
        print("Please manually download from:")
        print("  https://www.kaggle.com/datasets/tavarez/the-orl-database-for-training-and-testing")
        print(f"  and extract to {DATASET_DIR}")
        zip_path.unlink(missing_ok=True)
        return None

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATASET_DIR)
    zip_path.unlink()
    print(f"ORL dataset ready at {DATASET_DIR}")
    return DATASET_DIR


def get_subjects():
    """Return [(subject_name, [image_paths])] sorted by subject number."""
    if not DATASET_DIR.exists():
        return []
    subjects = []
    for d in sorted(DATASET_DIR.iterdir()):
        if d.is_dir() and d.name.startswith("s"):
            images = sorted(d.glob("*.pgm"))
            if images:
                subjects.append((d.name, images))
    return subjects
