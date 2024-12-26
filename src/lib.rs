//! A browser-optimized vector database. Backed by the private virtual filesystem API on web.
//!
//! You're viewing this on crates.io, so you're probably interested in the native version. The native version supports running with the native filesystem or in memory.
//!
//! If you want to use it on the web, [check out victor-db on npm](https://www.npmjs.com/package/victor-db).
//!
//! ## In-memory database
//!
//! Use this if you want to run victor in-memory (all data is lost when the program exits).
//!
//! The in-memory version is useful for testing and applications where you don't need to persist data:
//! ```rust
//! # tokio_test::block_on(async {
//! // use victor_db::memory for the in-memory implementation
//! use victor_db::memory::{Db, DirectoryHandle};
//!
//! // create a new in-memory database
//! let mut victor = Db::new(DirectoryHandle::default());
//!
//! // add some embeddings to the database
//! victor
//!     .add(
//!         vec!["Pineapple", "Rocks"], // documents
//!         vec!["Pizza Toppings"],     // tags (only used for filtering)
//!     )
//!     .await;
//!
//! // add another embedding to the database, this time with no tags
//! victor.add_single("Cheese pizza", vec!["Pizza Flavors"]).await;
//!
//! // read the 10 closest results from victor that are tagged with "Pizza Toppings"
//! // (only 2 will be returned because we only inserted two embeddings)
//! let nearest = victor
//!     .search("Hawaiian pizza", vec!["Pizza Toppings"], 10)
//!     .await
//!     .first()
//!     .unwrap()
//!     .content
//!     .clone();
//! assert_eq!(nearest, "Pineapple".to_string());
//!
//! // Clear the database
//! victor.clear_db().await.unwrap();
//! # })
//! ```
//!
//! ## Native database
//!
//! Use this if you want to persist your database to disk.
//!
//! ```rust
//! # tokio_test::block_on(async {
//! // use victor_db::native for the native filesystem implementation
//! use victor_db::native::Db;
//! use std::path::PathBuf;
//!
//! // create a new native database under "./victor_test_data"
//! let _ = std::fs::create_dir("./victor_test_data");
//! let mut victor = Db::new(PathBuf::from("./victor_test_data"));
//!
//! // add some embeddings to the database
//! victor
//!     .add(
//!         vec!["Pineapple", "Rocks"], // documents
//!         vec!["Pizza Toppings"],     // tags (only used for filtering)
//!     )
//!     .await;
//!
//! // add another embedding to the database, this time with no tags
//! victor.add_single("Cheese pizza", vec!["Pizza Flavors"]).await;
//!
//! // read the 10 closest results from victor that are tagged with "Pizza Toppings"
//! // (only 2 will be returned because we only inserted two embeddings)
//! let nearest = victor
//!     .search("Hawaiian pizza", vec!["Pizza Toppings"], 10)
//!     .await
//!     .first()
//!     .unwrap()
//!     .content
//!     .clone();
//! assert_eq!(nearest, "Pineapple".to_string());
//!
//! // Clear the database
//! victor.clear_db().await.unwrap();
//! # })
//! ```
//!
//! See the docs for [`Victor`] for more information.

#![deny(missing_docs)]

mod db;
mod decomposition;
mod filesystem;
mod packed_vector;
mod similarity;
mod utils;

#[cfg(not(target_arch = "wasm32"))]
pub use db::Victor;

#[cfg(test)]
mod tests;

#[cfg(target_arch = "wasm32")]
use {
    wasm_bindgen::prelude::*, wasm_bindgen_futures::JsFuture, web_sys::FileSystemDirectoryHandle,
};

#[cfg(target_arch = "wasm32")]
type Victor = crate::db::Victor<filesystem::web::DirectoryHandle>;

// Native

/// Victor's native filesystem implementation.
///
/// Use this if you want to persist your database to disk.
#[cfg(not(target_arch = "wasm32"))]
pub mod native {
    use crate::db::Victor;

    /// A native vector database.
    pub type Db = Victor<crate::filesystem::native::DirectoryHandle>;
}

/// Victor's in-memory implementation.
///
/// Use this if you want to run victor in-memory (all data is lost when the program exits).
#[cfg(not(target_arch = "wasm32"))]
pub mod memory {
    use crate::db::Victor;

    /// The directory handle type for the in-memory filesystem.
    pub use crate::filesystem::memory::DirectoryHandle;

    /// An in-memory vector database.
    pub type Db = Victor<DirectoryHandle>;
}

// Wasm

#[cfg(target_arch = "wasm32")]
#[allow(unused_macros)]
macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

#[cfg(target_arch = "wasm32")]
#[allow(unused_macros)]
macro_rules! console_warn {
    ($($t:tt)*) => (warn(&format_args!($($t)*).to_string()))
}
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
    #[wasm_bindgen(js_namespace = console)]
    fn warn(s: &str);
}

/// A browser-optimized vector database.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct Db {
    victor: crate::db::Victor<filesystem::web::DirectoryHandle>,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl Db {
    /// Connect to victor.
    #[wasm_bindgen(constructor)]
    pub async fn new() -> Self {
        utils::set_panic_hook();

        let window = web_sys::window().ok_or(JsValue::NULL).unwrap();
        let navigator = window.navigator();
        let file_system_directory_handle = FileSystemDirectoryHandle::from(
            JsFuture::from(navigator.storage().get_directory())
                .await
                .unwrap(),
        );

        let victor = Victor::new(file_system_directory_handle);

        Self { victor }
    }

    /// Add a document to the database.
    pub async fn insert(&mut self, content: &str, embedding: &[f64], tags: Option<Vec<JsValue>>) {
        let embedding = embedding.iter().map(|x| *x as f32).collect::<Vec<_>>();

        let tags = tags
            .map(|tags| {
                tags.into_iter()
                    .map(|x| x.as_string().unwrap())
                    .collect::<Vec<_>>()
            })
            .unwrap_or(vec![]);

        self.victor
            .add_single_embedding(content, embedding, tags)
            .await;
    }

    /// Search the database for the nearest neighbors to a given embedding.
    pub async fn search(
        &mut self,
        embedding: &[f64],
        tags: Option<Vec<JsValue>>,
        top_n: Option<f64>,
    ) -> JsValue {
        let embedding = embedding.iter().map(|x| *x as f32).collect::<Vec<_>>();

        let tags = tags
            .map(|tags| {
                tags.into_iter()
                    .map(|x| x.as_string().unwrap())
                    .collect::<Vec<_>>()
            })
            .unwrap_or(vec![]);

        let nearest_neighbors = self
            .victor
            .search_embedding(embedding, tags, top_n.unwrap_or(10.0) as u32)
            .await;

        serde_wasm_bindgen::to_value(&nearest_neighbors).unwrap()
    }

    /// Clear the database, permanently removing all data.
    pub async fn clear(&mut self) {
        utils::set_panic_hook();

        let result = self.victor.clear_db().await; // ignore the error if there is one
        if !result.is_ok() {
            console_warn!("Failed to clear victor data: {:?}", result);
        }
    }
}
