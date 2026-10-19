import React, { useCallback, useEffect, useState, useRef } from "react";
import cv from "@techstark/opencv-js";
import { Tensor, InferenceSession } from "onnxruntime-web";
import Loader from "./components/loader";
import { detectImage } from "./utils/detect";
import { download } from "./utils/download";
import labelsSfw from "./utils/labels.json";
import "./style/App.css";

const modelOptions = [
  { file: "320n.onnx", label: "320n", width: 320, height: 320, labels: labelsSfw },
  { file: "640m.onnx", label: "640m", width: 640, height: 640, labels: labelsSfw },
  {
    file: "f_best-v4.2.onnx",
    label: "f_best-v4.2",
    width: 640,
    height: 640,
    labels: ["breast_pair", "breast_single"],
  },
  {
    file: "f_best_v3.2.onnx",
    label: "f_best_v3.2",
    width: 640,
    height: 640,
    labels: ["breast_pair", "breast_single"],
  },
  {
    file: "top_yolov8s.onnx",
    label: "top_yolov8s",
    width: 640,
    height: 640,
    labels: ["nipples"],
  },
];

const App = () => {
  const [session, setSession] = useState(null);
  const [nmsSession, setNmsSession] = useState(null);
  const [selectedModel, setSelectedModel] = useState("");
  const [loading, setLoading] = useState({ text: "Loading OpenCV.js", progress: null });
  const [loadError, setLoadError] = useState(null);
  const [image, setImage] = useState(null);
  const inputImage = useRef(null);
  const imageRef = useRef(null);
  const canvasRef = useRef(null);
  const modelLoadId = useRef(0);
  const activeSession = useRef(null);
  const activeDetections = useRef(new Set());

  const topk = 100;
  const iouThreshold = 0.45;
  const scoreThreshold = 0.25;

  const loadModel = async (model, nms) => {
    const loadId = ++modelLoadId.current;
    const baseModelURL = `${process.env.PUBLIC_URL}/models`;
    setLoading({ text: `Loading ${model.label}`, progress: null });
    setLoadError(null);

    try {
      const modelBuffer = await download(`${baseModelURL}/${model.file}`, [
        `Loading ${model.label}`,
        setLoading,
      ]);
      const net = await InferenceSession.create(modelBuffer);
      const inputShape = [1, 3, model.height, model.width];
      const warmupTensor = new Tensor(
        "float32",
        new Float32Array(inputShape.reduce((size, dimension) => size * dimension)),
        inputShape
      );
      await net.run({ images: warmupTensor });

      if (loadId !== modelLoadId.current) {
        await net.release();
        return;
      }
      while (activeDetections.current.size) {
        await Promise.allSettled(Array.from(activeDetections.current));
      }
      if (loadId !== modelLoadId.current) {
        await net.release();
        return;
      }
      if (activeSession.current) await activeSession.current.net.release();
      const nextSession = { net, nms, inputShape, labels: model.labels };
      activeSession.current = nextSession;
      setSession(nextSession);
      setLoading(null);
    } catch (error) {
      if (loadId === modelLoadId.current) {
        setLoadError(`Could not load ${model.label}: ${error.message}`);
        setLoading(null);
      }
    }
  };

  const runDetection = useCallback(() => {
    if (session && imageRef.current && canvasRef.current) {
      let detection;
      detection = detectImage(
        imageRef.current,
        canvasRef.current,
        session,
        topk,
        iouThreshold,
        scoreThreshold
      )
        .catch((error) => {
          console.error("Detection failed", error);
        })
        .finally(() => activeDetections.current.delete(detection));
      activeDetections.current.add(detection);
    }
  }, [session, topk, iouThreshold, scoreThreshold]);

  useEffect(() => {
    if (image && imageRef.current?.complete) runDetection();
  }, [image, runDetection]);

  cv["onRuntimeInitialized"] = async () => {
    const nmsModelURL = `${process.env.PUBLIC_URL}/model/nms-yolov8.onnx`;
    try {
      const nmsBuffer = await download(nmsModelURL, [
        "Loading NMS model",
        setLoading,
      ]);
      const nms = await InferenceSession.create(nmsBuffer);
      setNmsSession(nms);
      setLoading(null);
    } catch (error) {
      setLoadError(`Could not initialize detector: ${error.message}`);
      setLoading(null);
    }
  };

  return (
    <div className="App">
      {loading && (
        <Loader>
          {loading.progress ? `${loading.text} - ${loading.progress}%` : loading.text}
        </Loader>
      )}
      <div className="header">
        <h1>SafeNet in-browser</h1>
        <label className="model-picker">
          Modello
          <select
            aria-label="Seleziona modello"
            value={selectedModel}
            disabled={!nmsSession || Boolean(loading)}
            onChange={(event) => {
              const model = modelOptions.find((option) => option.file === event.target.value);
              setSelectedModel(model.file);
              setLoadError(null);
              loadModel(model, nmsSession);
            }}
          >
            <option value="" disabled>
              Seleziona un modello
            </option>
            {modelOptions.map((model) => (
              <option key={model.file} value={model.file}>
                {model.label}
              </option>
            ))}
          </select>
        </label>
        {loadError && <p role="alert">{loadError}</p>}
      </div>

      <div className="content">
        <img
          ref={imageRef}
          src="#"
          alt=""
          style={{ display: image ? "block" : "none" }}
          onLoad={runDetection}
        />
        <canvas
          id="canvas"
          width={session?.inputShape[3] || 320}
          height={session?.inputShape[2] || 320}
          ref={canvasRef}
        />
      </div>

      <input
        type="file"
        ref={inputImage}
        accept="image/*"
        style={{ display: "none" }}
        onChange={(e) => {
          const file = e.target.files?.[0];
          if (!file) return;
          // handle next image to detect
          if (image) {
            URL.revokeObjectURL(image);
            setImage(null);
          }

          const url = URL.createObjectURL(file); // create image url
          imageRef.current.src = url; // set image source
          setImage(url);
        }}
      />
      <div className="btn-container">
        <button
          disabled={!session || Boolean(loading)}
          onClick={() => {
            inputImage.current.click();
          }}
        >
          Open local image
        </button>
        {image && (
          /* show close btn when there is image */
          <button
            onClick={() => {
              inputImage.current.value = "";
              imageRef.current.src = "#";
              URL.revokeObjectURL(image);
              setImage(null);
            }}
          >
            Close image
          </button>
        )}
      </div>
    </div>
  );
};

export default App;
