from pathlib import Path
from threading import Lock

from safenet import SafeDetector

models_dir = Path(__file__).resolve().parent.parent / "models"
detector_lock = Lock()
active_model_name = None
active_detector = None


def predictor(image_paths, batch_size=1, model_name="320n.onnx"):
    global active_model_name, active_detector

    model_name = Path(model_name).name
    if not model_name.endswith(".onnx"):
        raise ValueError("model_name must select an ONNX model")

    model_path = models_dir / model_name
    if not model_path.is_file():
        raise FileNotFoundError(f"model not found: {model_path}")

    with detector_lock:
        if active_model_name != model_name:
            active_detector = SafeDetector(model_path=model_path)
            active_model_name = model_name
        detector = active_detector

    return detector.detect_batch(image_paths, batch_size=batch_size)



