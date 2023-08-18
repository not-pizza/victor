const CopyWebpackPlugin = require("copy-webpack-plugin");
const path = require("path");

module.exports = {
  entry: "./bootstrap.ts", // Assuming your entry point is now a TypeScript file
  output: {
    path: path.resolve(__dirname, "dist"),
    filename: "bootstrap.js",
  },
  mode: "development",
  experiments: {
    asyncWebAssembly: true,
  },
  module: {
    rules: [
      // Add the TypeScript loader
      {
        test: /\.tsx?$/,
        use: "ts-loader",
        exclude: /node_modules/,
      },
      {
        test: /\.wasm$/,
        type: "webassembly/async",
      },
    ],
  },
  resolve: {
    extensions: [".tsx", ".ts", ".js"], // Resolve these extensions
  },
  plugins: [new CopyWebpackPlugin([{ from: "index.html", to: "index.html" }])],
};
