import os
import _io
import ast
import cv2
import numpy as np
import onnxruntime

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


def _read_image(image_path, target_size=(320, 320)):
    if isinstance(image_path, str):
        mat = cv2.imread(image_path)
    elif isinstance(image_path, np.ndarray):
        mat = image_path
    elif isinstance(image_path, bytes):
        mat = cv2.imdecode(np.frombuffer(image_path, np.uint8), -1)
    elif isinstance(image_path, _io.BufferedReader):
        mat = cv2.imdecode(np.frombuffer(image_path.read(), np.uint8), -1)
    else:
        raise ValueError(
            "please make sure the image_path is str or np.ndarray or bytes"
        )

    if mat is None:
        raise ValueError("the image could not be decoded")
    image_original_width, image_original_height = mat.shape[1], mat.shape[0]

    if len(mat.shape) == 2:
        mat_c3 = cv2.cvtColor(mat, cv2.COLOR_GRAY2BGR)
    elif mat.shape[2] == 4:
        mat_c3 = cv2.cvtColor(mat, cv2.COLOR_BGRA2BGR)
    else:
        mat_c3 = mat

    max_size = max(mat_c3.shape[:2])  # get max size from width and height
    x_pad = max_size - mat_c3.shape[1]  # set xPadding
    x_ratio = max_size / mat_c3.shape[1]  # set xRatio
    y_pad = max_size - mat_c3.shape[0]  # set yPadding
    y_ratio = max_size / mat_c3.shape[0]  # set yRatio

    mat_pad = cv2.copyMakeBorder(mat_c3, 0, y_pad, 0, x_pad, cv2.BORDER_CONSTANT)

    input_blob = cv2.dnn.blobFromImage(
        mat_pad,
        1 / 255.0,  # normalize
        target_size,  # resize to model input size
        (0, 0, 0),  # mean subtraction
        swapRB=True,  # swap red and blue channels
        crop=False,  # don't crop
    )

    return (
        input_blob,
        x_ratio,
        y_ratio,
        x_pad,
        y_pad,
        image_original_width,
        image_original_height,
    )


def _postprocess(
    output,
    x_pad,
    y_pad,
    x_ratio,
    y_ratio,
    image_original_width,
    image_original_height,
    model_width,
    model_height,
    class_names,
):
    outputs = np.squeeze(output[0])
    if outputs.ndim != 2:
        raise ValueError(f"unexpected model output shape: {outputs.shape}")
    expected_channels = len(class_names) + 4
    if outputs.shape[0] == expected_channels:
        outputs = np.transpose(outputs)
    elif outputs.shape[1] != expected_channels:
        raise ValueError(
            f"model output has {outputs.shape} values, expected "
            f"{expected_channels} values per detection"
        )
    rows = outputs.shape[0]
    boxes = []
    scores = []
    class_ids = []

    for i in range(rows):
        classes_scores = outputs[i][4:]
        max_score = np.amax(classes_scores)

        if max_score >= 0.2:
            class_id = np.argmax(classes_scores)
            x, y, w, h = outputs[i][0:4]

            # Convert from center coordinates to top-left corner coordinates
            x = x - w / 2
            y = y - h / 2

            # Scale coordinates to original image size
            x = x * (image_original_width + x_pad) / model_width
            y = y * (image_original_height + y_pad) / model_height
            w = w * (image_original_width + x_pad) / model_width
            h = h * (image_original_height + y_pad) / model_height

            # Remove padding
            x = x
            y = y

            # Clip coordinates to image boundaries
            x = max(0, min(x, image_original_width))
            y = max(0, min(y, image_original_height))
            w = min(w, image_original_width - x)
            h = min(h, image_original_height - y)

            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([x, y, w, h])

    indices = np.asarray(cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45)).reshape(-1)

    detections = []
    for i in indices:
        box = boxes[i]
        score = scores[i]
        class_id = class_ids[i]

        x, y, w, h = box
        detections.append(
            {
                "class": class_names[class_id],
                "score": float(score),
                "box": [int(x), int(y), int(w), int(h)],
            }
        )

    return detections


class SafeDetector:
    def __init__(self, model_path=None, providers=None, inference_resolution=None):
        if model_path is None:
            repository_model = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "models", "320n.onnx"
            )
            packaged_model = os.path.join(os.path.dirname(__file__), "320n.onnx")
            model_path = (
                repository_model if os.path.isfile(repository_model) else packaged_model
            )
        if providers is not None:
            self.onnx_session = onnxruntime.InferenceSession(
                model_path, providers=providers
            )
        else:
            available_providers = onnxruntime.get_available_providers()
            if "CUDAExecutionProvider" in available_providers:
                try:
                    self.onnx_session = onnxruntime.InferenceSession(
                        model_path,
                        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                    )
                except Exception:
                    self.onnx_session = onnxruntime.InferenceSession(
                        model_path, providers=["CPUExecutionProvider"]
                    )
            else:
                self.onnx_session = onnxruntime.InferenceSession(
                    model_path, providers=["CPUExecutionProvider"]
                )
        model_inputs = self.onnx_session.get_inputs()
        input_shape = model_inputs[0].shape
        self.has_static_batch = isinstance(input_shape[0], int)
        metadata = self.onnx_session.get_modelmeta().custom_metadata_map
        try:
            metadata_size = ast.literal_eval(metadata.get("imgsz", "[320, 320]"))
        except (ValueError, SyntaxError):
            metadata_size = [320, 320]

        self.input_height = self._static_dimension(input_shape[2], metadata_size[0])
        self.input_width = self._static_dimension(input_shape[3], metadata_size[1])
        if inference_resolution is not None:
            if isinstance(inference_resolution, (tuple, list)):
                self.input_width, self.input_height = inference_resolution
            else:
                self.input_width = self.input_height = inference_resolution

        try:
            model_names = ast.literal_eval(metadata.get("names", "{}"))
        except (ValueError, SyntaxError):
            model_names = {}
        model_name = os.path.basename(model_path)
        if model_name == "top_yolov8s.onnx":
            self.class_names = ["top"]
        elif model_name in {"320n.onnx", "640m.onnx"}:
            self.class_names = all_labels_sfw
        elif isinstance(model_names, dict) and model_names:
            self.class_names = [
                str(model_names[key]) for key in sorted(model_names, key=lambda key: int(key))
            ]
        else:
            self.class_names = None

        self.input_name = model_inputs[0].name

        if self.class_names is None:
            output_shape = self.onnx_session.get_outputs()[0].shape
            channel_count = next(
                (dimension for dimension in output_shape[1:] if isinstance(dimension, int)),
                None,
            )
            if channel_count is None or channel_count <= 4:
                raise ValueError("the model does not provide class names or a valid output shape")
            self.class_names = [f"CLASS_{index}" for index in range(channel_count - 4)]

    @staticmethod
    def _static_dimension(dimension, fallback):
        return dimension if isinstance(dimension, int) else int(fallback)

    def detect(self, image_path):
        (
            preprocessed_image,
            x_ratio,
            y_ratio,
            x_pad,
            y_pad,
            image_original_width,
            image_original_height,
        ) = _read_image(image_path, (self.input_width, self.input_height))
        outputs = self.onnx_session.run(None, {self.input_name: preprocessed_image})
        detections = _postprocess(
            outputs,
            x_pad,
            y_pad,
            x_ratio,
            y_ratio,
            image_original_width,
            image_original_height,
            self.input_width,
            self.input_height,
            self.class_names,
        )

        return detections

    def detect_batch(self, image_paths, batch_size=4):
        """
        Perform batch detection on a list of images.

        Args:
            image_paths (List[Union[str, np.ndarray]]): List of image paths or numpy arrays.
            batch_size (int): Number of images to process in each batch.

        Returns:
            List of detection results for each image.
        """
        if self.has_static_batch:
            return [self.detect(image_path) for image_path in image_paths]

        all_detections = []

        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i : i + batch_size]
            batch_inputs = []
            batch_metadata = []

            for image_path in batch:
                (
                    preprocessed_image,
                    x_ratio,
                    y_ratio,
                    x_pad,
                    y_pad,
                    image_original_width,
                    image_original_height,
                ) = _read_image(image_path, (self.input_width, self.input_height))
                batch_inputs.append(preprocessed_image)
                batch_metadata.append(
                    (
                        x_ratio,
                        y_ratio,
                        x_pad,
                        y_pad,
                        image_original_width,
                        image_original_height,
                    )
                )

            # Stack the preprocessed images into a single numpy array
            batch_input = np.vstack(batch_inputs)

            # Run inference on the batch
            outputs = self.onnx_session.run(None, {self.input_name: batch_input})

            # Process the outputs for each image in the batch
            for j, metadata in enumerate(batch_metadata):
                (
                    x_ratio,
                    y_ratio,
                    x_pad,
                    y_pad,
                    image_original_width,
                    image_original_height,
                ) = metadata
                detections = _postprocess(
                    [outputs[0][j : j + 1]],  # Select the output for this image
                    x_pad,
                    y_pad,
                    x_ratio,
                    y_ratio,
                    image_original_width,
                    image_original_height,
                    self.input_width,
                    self.input_height,
                    self.class_names,
                )
                all_detections.append(detections)

        return all_detections

    def censor(self, image_path, classes=[], output_path=None):
        detections = self.detect(image_path)
        if classes:
            detections = [
                detection for detection in detections if detection["class"] in classes
            ]

        img = cv2.imread(image_path)

        for detection in detections:
            box = detection["box"]
            x, y, w, h = box[0], box[1], box[2], box[3]
            # change these pixels to pure black
            img[y : y + h, x : x + w] = (0, 0, 0)

        if not output_path:
            image_path, ext = os.path.splitext(image_path)
            output_path = f"{image_path}_censored{ext}"

        cv2.imwrite(output_path, img)

        return output_path


if __name__ == "__main__":
    detector = SafeDetector()
    # detections = detector.detect("/Users/praneeth.bedapudi/Desktop/cory.jpeg")
    print(
        detector.detect_batch(
            [
                "/Users/praneeth.bedapudi/Desktop/d.jpg",
                "/Users/praneeth.bedapudi/Desktop/a.jpeg",
            ]
        )[0]
    )
    print(detector.detect_batch(["/Users/praneeth.bedapudi/Desktop/d.jpg"])[0])

    print(
        detector.detect_batch(
            [
                "/Users/praneeth.bedapudi/Desktop/d.jpg",
                "/Users/praneeth.bedapudi/Desktop/a.jpeg",
            ]
        )[1]
    )
    print(detector.detect_batch(["/Users/praneeth.bedapudi/Desktop/a.jpeg"])[0])
