from pathlib import Path
import hashlib
import sys
import threading

import cv2


DEFAULT_MODEL_FILES = ("320n", "640m", "top_yolov8s")
MODEL_LABELS = {
    "320n": {
        "BUTTONS_EXPRESS": "BACK",
        "F_BEST_EXPRESS": "BEST",
        "F_GENIE_EXPRESS": "GEN",
        "ANGUS_EXPRESS": "ASS",
        "BELL_EXPRESS": "BELLY",
    },
    "640m": {
        "BUTTONS_EXPRESS": "BACK",
        "F_BEST_EXPRESS": "BEST",
        "F_GENIE_EXPRESS": "GEN",
        "ANGUS_EXPRESS": "ASS",
        "BELL_EXPRESS": "BELLY",
    },
    "top_yolov8s": {"top": "TOP"},
}
LABEL_COLORS = {
    "BACK": (45, 130, 75),
    "BEST": (50, 105, 220),
    "GEN": (220, 115, 35),
    "ASS": (180, 80, 160),
    "BELLY": (30, 155, 190),
    "TOP": (40, 80, 225),
}
DEFAULT_MODELS = ("320n", "top_yolov8s")


class DetectionCancelledError(RuntimeError):
    pass


class DetectionTimelineAnalyzer:
    def __init__(self, models_dir: Path | None = None):
        if models_dir is not None:
            self.models_dir = models_dir
        else:
            module_dir = Path(__file__).resolve().parent
            candidates = [
                module_dir.parent / "models",
                module_dir / "models",
                Path(getattr(sys, "_MEIPASS", module_dir)) / "models",
            ]
            self.models_dir = next((path for path in candidates if path.is_dir()), candidates[0])

    def available_models(self) -> dict[str, str]:
        if not self.models_dir.is_dir():
            return {}
        files = {path.stem: path.name for path in self.models_dir.glob("*.onnx")}
        order = {name: index for index, name in enumerate(DEFAULT_MODEL_FILES)}
        return dict(sorted(files.items(), key=lambda item: (order.get(item[0], len(order)), item[0].casefold())))

    @staticmethod
    def model_labels(model_name: str, detector) -> dict[str, str]:
        known_labels = MODEL_LABELS.get(model_name)
        if known_labels is not None:
            return known_labels
        return {str(label): str(label) for label in detector.class_names}

    @staticmethod
    def label_color(label: str) -> tuple[int, int, int]:
        if label in LABEL_COLORS:
            return LABEL_COLORS[label]
        digest = hashlib.sha256(label.encode("utf-8")).digest()
        return tuple(55 + value % 170 for value in digest[:3])

    @classmethod
    def label_color_hex(cls, label: str) -> str:
        blue, green, red = cls.label_color(label)
        return f"#{red:02x}{green:02x}{blue:02x}"

    def load_detectors(self, models=DEFAULT_MODELS):
        repository_root = str(self.models_dir.parent)
        if repository_root not in sys.path:
            sys.path.insert(0, repository_root)
        try:
            from safenet import SafeDetector
        except ImportError as error:
            raise RuntimeError(
                "SafeNet non è installato. Dalla root del repository esegui: "
                "python -m pip install -e ."
            ) from error

        selected = list(dict.fromkeys(models))
        available = self.available_models()
        unknown = set(selected) - set(available)
        if unknown:
            raise ValueError(f"Modelli non supportati: {', '.join(sorted(unknown))}")
        if not selected:
            raise ValueError("Seleziona almeno un modello.")

        detectors = {}
        for model_name in selected:
            filename = available[model_name]
            model_path = self.models_dir / filename
            if not model_path.is_file():
                raise FileNotFoundError(f"Modello non trovato: {model_path}")
            detectors[model_name] = SafeDetector(model_path=str(model_path))
        return detectors

    def analyze(self, video_path: str, cancelled: threading.Event, report_progress, detectors=None):
        detectors = detectors or self.load_detectors()
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise RuntimeError("Impossibile aprire il video per l’analisi detect.")
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps <= 0 or frame_count <= 0:
                raise RuntimeError("FPS o numero di frame non disponibili.")

            frames = []
            frame_index = 0
            while True:
                if cancelled.is_set():
                    raise DetectionCancelledError("Analisi detect annullata.")
                ok, frame = capture.read()
                if not ok or frame is None:
                    break

                frame_models = {}
                for model_name, detector in detectors.items():
                    labels = self.model_labels(model_name, detector)
                    for item in detector.detect(frame):
                        label = labels.get(item["class"])
                        if label:
                            frame_models.setdefault(model_name, []).append({
                                "label": label,
                                "score": float(item["score"]),
                            })
                if frame_models:
                    frames.append({
                        "frameIndex": frame_index,
                        "models": frame_models,
                    })

                frame_index += 1
                if frame_index % 10 == 0 or frame_index == frame_count:
                    progress = min(99, round(frame_index * 100 / frame_count))
                    report_progress(
                        progress,
                        f"Analisi frame {frame_index:,} / {frame_count:,}",
                    )

            if cancelled.is_set():
                raise DetectionCancelledError("Analisi detect annullata.")
            report_progress(100, f"Analizzati {frame_index:,} frame")
            return {
                "version": 1,
                "fps": fps,
                "frameCount": frame_index,
                "duration": frame_index / fps,
                "models": list(detectors),
                "classes": {
                    model_name: list(self.model_labels(model_name, detector).values())
                    for model_name, detector in detectors.items()
                },
                "colors": {
                    label: self.label_color_hex(label)
                    for model_name, detector in detectors.items()
                    for label in self.model_labels(model_name, detector).values()
                },
                "frames": frames,
            }, detectors
        finally:
            capture.release()


def annotated_frame(
    video_path: str,
    frame_index: int,
    detectors: dict,
    active_classes: dict[str, list[str]] | None = None,
) -> bytes:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError("Impossibile riaprire il video.")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_index))
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError("Frame non disponibile.")

        height, width = frame.shape[:2]
        for model_name, detector in detectors.items():
            labels = DetectionTimelineAnalyzer.model_labels(model_name, detector)
            enabled = active_classes.get(model_name) if active_classes is not None else None
            for detection in detector.detect(frame):
                label = labels.get(detection["class"])
                if label is None:
                    continue
                if enabled is not None and label not in enabled:
                    continue
                x, y, box_width, box_height = detection["box"]
                start = (max(0, x), max(0, y))
                end = (min(width - 1, x + box_width), min(height - 1, y + box_height))
                color = DetectionTimelineAnalyzer.label_color(label)
                text = f"{label} {detection['score'] * 100:.1f}%"
                cv2.rectangle(frame, start, end, color, max(2, round(min(width, height) / 400)))
                cv2.putText(
                    frame,
                    text,
                    (start[0], max(18, start[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.45, min(width, height) / 1200),
                    color,
                    2,
                    cv2.LINE_AA,
                )

        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise RuntimeError("Impossibile codificare il frame annotato.")
        return encoded.tobytes()
    finally:
        capture.release()
