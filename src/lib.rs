//! A browser-optimized vector database. Backed by the private virtual filesystem API on web. On native, supports running with the native filesystem or in memory.

#![deny(missing_docs)]

mod db;
mod decomposition;
mod filesystem;
mod packed_vector;
mod similarity;
mod utils;

#[cfg(test)]
mod tests;

#[cfg(target_arch = "wasm32")]
use {
    wasm_bindgen::prelude::*, wasm_bindgen_futures::JsFuture, web_sys::FileSystemDirectoryHandle,
};

#[cfg(target_arch = "wasm32")]
type Victor = crate::db::Victor<filesystem::web::DirectoryHandle>;

// Native

/// Used to tell victor to use the native filesystem.
#[cfg(not(target_arch = "wasm32"))]
pub mod native {
    use crate::db::Victor;

    /// A native vector database.
    pub type Db = Victor<crate::filesystem::native::DirectoryHandle>;
}

/// Used to tell victor to use an in-memory filesystem.
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

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub struct Db {
    victor: crate::db::Victor<filesystem::web::DirectoryHandle>,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
impl Db {
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

    pub async fn insert(&mut self, content: &str, embedding: &[f64], tags: Option<Vec<JsValue>>) {
        let embedding = embedding.iter().map(|x| *x as f32).collect::<Vec<_>>();

        let tags = tags
            .map(|tags| {
                tags.into_iter()
                    .map(|x| x.as_string().unwrap())
                    .collect::<Vec<_>>()
            })
            .unwrap_or(vec![]);

        self.victor.add_embedding(content, embedding, tags).await;
    }

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

    pub async fn clear(&mut self) {
        utils::set_panic_hook();

        let result = self.victor.clear_db().await; // ignore the error if there is one
        if !result.is_ok() {
            console_warn!("Failed to clear victor data: {:?}", result);
        }
    }
}
