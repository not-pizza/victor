## Victor

Web-optimized vector database (written in Rust).

## Features

1. Rust API (using native filesystem, or a transient in-memory filesystem)
2. Web API (Using the [Private Origin File System](https://web.dev/origin-private-file-system/))
3. Very efficient vector storage format
   1. For a vector with 1536 dimensions, our representation consumes 1.5 KB, while naively encoding with JSON would consume 20.6 KB.
4. PCA for vector compression when storage space is low


## JS Example

#### Installation

```
npm install victor-db
```

#### Usage

```ts
import { Db } from "victor";

const db = await Db.new();

const content = "My content!";
const tags = ["these", "are", "tags"];
const embedding = new Float64Array(/* your embedding here */);

// write to victor
await db.insert(content, embedding, tags);

// read from victor
const result = await db.search(embedding, ["tags"]);
assert(result == content);

// clear database
await db.clear();
```

See `www/` for a more complete example, including fetching embeddings from OpenAI.

## Rust Example

#### Installation

```
cargo add victor-db
```

#### Usage

```ts
use std::path::PathBuf;

use victor_db::native::Db;

let _ = std::fs::create_dir("./victor_test_data");
let mut victor = Db::new(PathBuf::from("./victor_test_data"));

victor.clear_db().await.unwrap();

victor
    .write(
        "Test Vector 1",
        vec![1.0, 0.0, 0.0],
        vec!["Test".to_string()],
    )
    .await;
victor
    .write(
        "Test Vector 2",
        vec![0.0, 1.0, 0.0],
        vec!["Test".to_string()],
    )
    .await;

let nearest = victor
    .find_nearest_neighbor(vec![0.9, 0.0, 0.0], vec![])
    .await
    .unwrap()
    .content;
assert_eq!(nearest, "Test Vector 1".to_string());
```

This example is also in the `/examples` directory. If you've cloned this repository, you can run it with `cargo run --example native_filesystem`.

## Hacking

1. Victor is written in Rust, and compiled to wasm with wasm-pack.

   **Install wasm** pack with `cargo install wasm-pack` or `npm i -g wasm-pack`
   (https://rustwasm.github.io/wasm-pack/installer/)

2. **Build Victor** with `wasm-pack build`

3. **Set up the example project**, which is in `www/`.

   If you use nvm, you can just run `cd www/ && nvm use`

   Then, `npm i`.

4. From `www/`, start the example project with `npm run start`.

## Architecture

Relevant code at `src/packed_vector.rs`.

![Packed vector storage explanation](./assets/packed_vector_storage.png)

---

![File structure explanation](assets/file_structure.png)

## Us

[Sam Hall](https://twitter.com/Shmall27)

[Andre Popovitch](https://twitter.com/ChadNauseam)
