const CopyPlugin = require("copy-webpack-plugin");
const path = require("path");

module.exports = {
  webpack: {
    plugins: {
      add: [
        new CopyPlugin({
          // Use copy plugin to copy *.wasm to output folder.
          patterns: [
            { from: "node_modules/onnxruntime-web/dist/*.wasm", to: "static/js/[name][ext]" },
            { from: "../models/*.onnx", to: "models/[name][ext]", noErrorOnMissing: true },
          ],
        }),
      ],
    },
    configure: (config) => {
      // set resolve.fallback for opencv.js
      config.resolve.fallback = {
        fs: false,
        path: false,
        crypto: false,
      };
      return config;
    },
  },
  devServer: (config) => {
    config.static = [
      ...(Array.isArray(config.static) ? config.static : [config.static]),
      {
        directory: path.resolve(__dirname, "../models"),
        publicPath: "/models",
        watch: true,
      },
    ];
    return config;
  },
};
