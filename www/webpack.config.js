const CopyWebpackPlugin = require('copy-webpack-plugin');
const path = require('path');

module.exports = {
  entry: './bootstrap.ts',
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: 'bootstrap.js',
  },
  mode: 'development',
  experiments: {
    asyncWebAssembly: true,
    syncWebAssembly: true,
  },
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        use: 'ts-loader',
        exclude: /node_modules/,
      },
      {
        test: /\.wasm$/,
        type: 'webassembly/async',
      },
    ],
  },
  resolve: {
    extensions: ['.tsx', '.ts', '.js', '.wasm'],
  },
  plugins: [new CopyWebpackPlugin({
    patterns: [
      { from: 'index.html' }
    ]
  })],
};
