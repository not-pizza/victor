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

// read the 10 closest results from victor that are tagged with "tags"
// (only 1 will be returned because we only inserted one embedding)
const result = await db.search(embedding, ["tags"], 10);
assert(result[0].content == content);

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

```rust
use std::path::PathBuf;

use victor_db::native::Db;

let _ = std::fs::create_dir("./victor_test_data");
let mut victor = Db::new(PathBuf::from("./victor_test_data"));

victor.clear_db().await.unwrap();

victor
    .add_many(
        vec!["Pinapple", "Rocks"], // documents
        vec!["PizzaToppings"], // tags (only used for filtering)
    )
    .await;

// read the 10 closest results from victor that are tagged with "tags"
// (only 2 will be returned because we only inserted two embeddings)
let nearest = victor
   .search(vec!["Hawaiian pizza".to_string()], 10)
   .await
   .first()
   .unwrap()
   .content
   .clone();
assert_eq!(nearest, "Pineapple".to_string());
```

This example is also in the `/examples` directory. If you've cloned this repository, you can run it with `cargo run --example native_filesystem`.

## Hacking

1. Victor is written in Rust, and compiled to wasm with wasm-pack.

   **Install wasm** pack with `cargo install wasm-pack` or `npm i -g wasm-pack`
   (https://rustwasm.github.io/wasm-pack/installer/)

2. **Build Victor** with `wasm-pack build --target web`

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
