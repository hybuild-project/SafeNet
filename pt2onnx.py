from ultralytics import YOLO

from pathlib import Path

for model_path in sorted(Path("models").glob("*.pt")):
    model = YOLO(str(model_path))
    model.export(format="onnx")