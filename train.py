# The downloaded archive and extracted images use non-displayable suffixes to reduce accidental exposure.

''' 
Trains a SafeNet YOLO model from an existing checkpoint.

# Script Execution Instructions

# Updated torch with cuda install command can be found from https://pytorch.org/get-started/locally/
# install CUDA Toolkit from https://developer.nvidia.com/cuda-toolkit
# install cuDNN from https://developer.nvidia.com/cudnn

# linux
python -m venv safenetTraining
source safenetTraining/bin/activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install Pillow ultralytics requests tqdm
python train.py

# windows
python -m venv safenetTraining
safenetTraining\\Scripts\\activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install Pillow ultralytics requests tqdm
python train.py
'''

import subprocess
import logging
from pathlib import Path
import zipfile
import requests
import shutil
from tqdm import tqdm
import torch
from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS

safe_file_extensions = ["jpg_safe", "jpeg_safe", "png_safe"]
if hasattr(IMG_FORMATS, "update"):
    IMG_FORMATS.update(safe_file_extensions)
else:
    IMG_FORMATS.extend(safe_file_extensions)

# === CONFIGURATION ===
DATASET_URL = "https://archive.org/download/NudeNet_classifier_dataset_v1/NudeNet_Classifier_train_data_x320.zip"
BASE_CHECKPOINT_URL = "https://github.com/notAI-tech/NudeNet/releases/download/v3.4-weights/640m.pt"
YOLO11_MODEL = "yolo11n.pt" # Available models: yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
CONFIDENCE_THRESHOLD = 0.25
NMS_THRESHOLD = 0.7
PATIENCE = 5
BATCH_SIZE = 128
IMG_SIZE = 320
WORKERS = 16
DEGREES = 180
SHEAR = 20
FLIPUD = .05
FLIPLR = .05
BGR = .05
# === CONFIGURATION ===

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DATASET_PATH = Path("dataset.bin")
EXTRACTED_FOLDER = Path("SafeNet_dataset_extracted")
BASE_CHECKPOINT_PATH = Path(Path(BASE_CHECKPOINT_URL).name)

all_labels_sfw = [
    "F_GENIE_CROSSED",
    "FAUCET_F",
    "BUTTONS_EXPRESS",
    "F_BEST_EXPRESS",
    "F_GENIE_EXPRESS",
    "M_BEST_EXPRESS",
    "ANGUS_EXPRESS",
    "FIT_EXPRESS",
    "BELL_CROSSED",
    "FIT_CROSSED",
    "ARMCHAIRS_CROSSED",
    "ARMCHAIRS_EXPRESS",
    "FAUCET_M",
    "BELL_EXPRESS",
    "M_GENIE_EXPRESS",
    "ANGUS_CROSSED",
    "F_BEST_CROSSED",
    "BUTTONS_CROSSED",
]

IGNORED_LABELS = [
    "FAUCET_F",
    "FIT_EXPRESS",
    "BELL_CROSSED",
    "FIT_CROSSED",
    "ARMCHAIRS_CROSSED",
    "FAUCET_M",
]

def download_dataset():
    if not DATASET_PATH.exists():
        logging.info(f"Downloading dataset from {DATASET_URL}...")
        response = requests.get(DATASET_URL, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        chunk_size = 8192
        num_bars = total_size // chunk_size if total_size else None
        with DATASET_PATH.open("wb") as f, tqdm(
            desc="Downloading dataset",
            total=num_bars,
            unit='KB',
            unit_scale=True,
            leave=False
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(1)
        logging.info(f"Dataset saved as {DATASET_PATH}.")
    else:
        logging.info(f"Dataset already exists: {DATASET_PATH}")

def extract_dataset():
    download_dataset()
    if not EXTRACTED_FOLDER.exists():
        EXTRACTED_FOLDER.mkdir(parents=True, exist_ok=True)

        counter = 0
        allowed_folders = {"training", "validation", "testing"}
        with zipfile.ZipFile(DATASET_PATH, 'r') as zip_ref:
            for file in tqdm(zip_ref.namelist(), desc="Extracting dataset", unit="file"):
                file_path = Path(file)

                if file.endswith("/"):
                    continue

                if file_path.parts[0] == "nude_sexy_safe_v1_x320":
                    file_path = Path(*file_path.parts[1:])

                if len(file_path.parts) < 2:
                    continue

                if file_path.parts[0] not in allowed_folders:
                    continue

                if file_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue

                dest_dir = EXTRACTED_FOLDER / file_path.parent               
                dest = dest_dir / f"{counter}{file_path.suffix}_safe"
                counter += 1
                if dest.exists():
                    continue

                dest_dir.mkdir(parents=True, exist_ok=True)

                with zip_ref.open(file) as source, open(dest.resolve(), 'wb') as target:
                    shutil.copyfileobj(source, target)

        logging.info(f"Dataset extracted to {EXTRACTED_FOLDER}")
    else:
        logging.info(f"Dataset already extracted to {EXTRACTED_FOLDER}")

def download_base_checkpoint():
    if not BASE_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Download the base checkpoint from {BASE_CHECKPOINT_URL} "
            f"and save it as {BASE_CHECKPOINT_PATH.resolve()}"
        )
    else:
        logging.info(f"Base checkpoint already exists: {BASE_CHECKPOINT_PATH}")

def get_next_multiple_of_32(image_path):
    from PIL import Image
    with Image.open(image_path) as img:
        width, height = img.size
    max_dim = max(width, height)
    return ((max_dim + 31) // 32) * 32
        
def process_images():
    download_base_checkpoint()
    model = YOLO(str(BASE_CHECKPOINT_PATH))
    
    images_to_process = [file for file in EXTRACTED_FOLDER.rglob("*")
                         if file.is_file() and file.suffix != ".txt"]
    
    for image_path in tqdm(images_to_process, desc="Processing images", unit="file"):
        txt_output_path = image_path.with_suffix(".txt")
        if txt_output_path.exists():
            continue
            
        try:
            results = model.predict(str(image_path), conf=CONFIDENCE_THRESHOLD, iou=NMS_THRESHOLD, agnostic_nms=True, imgsz=get_next_multiple_of_32(image_path))
            filtered_boxes = []
            for result in results:
                for box in result.boxes:
                    try:
                        x,y,w,h = box.xywhn[0].tolist()
                        class_id = box.cls[0].tolist()
                        label = all_labels_sfw[int(class_id)]
                        if label not in IGNORED_LABELS:
                            filtered_boxes.append((label, x, y, w, h))
                    except Exception as e:
                        pass
            if not filtered_boxes:
                if "safe" in txt_output_path.parts:
                    txt_output_path.touch()
                else:
                    logging.error(f"Skipping nude image file with no detections {image_path}")
                    image_path.unlink()
            else:
                if "safe" in txt_output_path.parts:
                    logging.error(f"Skipping safe image file with detections {image_path}")
                    image_path.unlink()
                else:
                    with txt_output_path.open("w") as f:
                        for label, centerX, centerY, width, height in filtered_boxes:                            
                            f.write(f"{all_labels_sfw.index(label)} {centerX:.6f} {centerY:.6f} {width:.6f} {height:.6f}\n")
                            
        except Exception as e:
            logging.error(f"Skipping invalid image file {image_path}: {e}")
            image_path.unlink()
            continue
    
def download_yolo11_model(model_name):
    url = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{model_name}"
    model_path = Path(model_name)
    if not model_path.exists():
        logging.info(f"Downloading {model_name} from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status() 
        with model_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logging.info(f"Model {model_name} downloaded successfully.")
    else:
        logging.info(f"Model already exists: {model_path}")

def create_data_yaml_file():
    names_yaml = ""
    for index, label in enumerate(all_labels_sfw):
        names_yaml += f"  {index}: {label}\n"
    data_yaml = f"""path: ./{str(EXTRACTED_FOLDER)}
train: {Path('SafeNet_dataset_extracted/training').resolve()}
val: {Path('SafeNet_dataset_extracted/validation').resolve()}
test: {Path('SafeNet_dataset_extracted/testing').resolve()}

# Classes
names:
{names_yaml}"""
    data_yaml_path = Path("data.yaml")
    with data_yaml_path.open("w") as f:
        f.write(data_yaml)
    logging.info("data.yaml created.")

def train_yolo11():
    extract_dataset()
    process_images()
    download_yolo11_model(YOLO11_MODEL)
    create_data_yaml_file()
    valid_class_indices = [i for i, label in enumerate(all_labels_sfw) if label not in IGNORED_LABELS]
    classes_str = "[" + ", ".join(map(str, valid_class_indices)) + "]"
    logging.info("Starting YOLO11 training...")

    result = subprocess.run([
        "yolo",
        "task=detect",
        "mode=train",
        f"model={YOLO11_MODEL}",
        f"data={Path('data.yaml').resolve()}",
        f"patience={PATIENCE}",
        f"batch={BATCH_SIZE}",
        f"imgsz={IMG_SIZE}",
        f"workers={WORKERS}",
        "resume=False",
        f"classes={classes_str}",
        f"degrees={DEGREES}",
        f"shear={SHEAR}",
        f"flipud={FLIPUD}",
        f"fliplr={FLIPLR}",
        f"bgr={BGR}",
    ])
    if result.returncode != 0:
        logging.error("YOLO training failed.")

if __name__ == "__main__":
    train_yolo11()