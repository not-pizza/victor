## Victor

Web-optimized vector database (written in Rust).

## Features

1. Rust API (using native filesystem, or a transient in-memory filesystem)
2. Web API (Using the [Private Origin File System](https://web.dev/origin-private-file-system/))
3. Very efficient vector storage format
4. PCA for vector compression when storage space is low

## Setup

1. install wasm pack
   `cargo install wasm-pack` or `npm i -g wasm-pack`
   https://rustwasm.github.io/wasm-pack/installer/

2. install dependencies
   we use node 20 in this project
   if you use nvm, you can just run `nvm use`

   `npm i`

3.
