mod db;
mod filesystem;
mod packed_vector;
mod similarity;
mod utils;

#[cfg(test)]
mod tests;

#[cfg(target_arch = "wasm32")]
use {wasm_bindgen::prelude::*, web_sys::FileSystemDirectoryHandle};

// Native

#[cfg(not(target_arch = "wasm32"))]
pub mod native {
    use crate::db::Victor;

    pub type Db = Victor<crate::filesystem::native::DirectoryHandle>;
}

#[cfg(not(target_arch = "wasm32"))]
pub mod memory {
    use crate::db::Victor;

    pub use crate::filesystem::memory::DirectoryHandle;

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
/// Assumes all the embeddings are the size of `embedding`
/// TODO: Record the embedding size somewhere so we can return an error if
/// the sizes are wrong (as otherwise this will corrupt the entire db)
#[wasm_bindgen]
pub async fn write_embedding(root: FileSystemDirectoryHandle, embedding: &[f64], content: &str) {
    utils::set_panic_hook();

    let mut victor = Victor::new(filesystem::web::DirectoryHandle::from(root));

    let embedding = embedding.iter().map(|x| *x as f32).collect::<Vec<_>>();

    victor.write(embedding, content, vec![]).await;
}

#[cfg(target_arch = "wasm32")]
/// Assumes all the embeddings are the size of `embedding`
#[wasm_bindgen]
pub async fn find_nearest_neighbor(root: FileSystemDirectoryHandle, embedding: &[f64]) -> JsValue {
    utils::set_panic_hook();

    let mut victor = Victor::new(filesystem::web::DirectoryHandle::from(root));

    let embedding = embedding.iter().map(|x| *x as f32).collect::<Vec<_>>();

    let nearest = victor.find_nearest_neighbor(embedding, vec![]).await;

    if let Some(nearest) = nearest {
        wasm_bindgen::JsValue::from_str(&nearest.content)
    } else {
        wasm_bindgen::JsValue::NULL
    }
}

#[cfg(target_arch = "wasm32")]
/// Assumes all the embeddings are the size of `embedding`
#[wasm_bindgen]
pub async fn clear_db(root: FileSystemDirectoryHandle) {
    utils::set_panic_hook();

    let mut victor = Victor::new(filesystem::web::DirectoryHandle::from(root));
    let result = victor.clear_db().await; // ignore the error if there is one
    if result.is_ok() {
        console_log!("Victor data cleared");
    } else {
        console_warn!("Failed to clear victor data: {:?}", result);
    }
}
