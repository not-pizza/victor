mod db;
mod filesystem;
mod similarity;
mod utils;

#[cfg(test)]
mod tests;

use db::Victor;
use wasm_bindgen::prelude::*;
use web_sys::FileSystemDirectoryHandle;

#[allow(unused_macros)]
macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}

#[allow(unused_macros)]
macro_rules! console_warn {
    ($($t:tt)*) => (warn(&format_args!($($t)*).to_string()))
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
    #[wasm_bindgen(js_namespace = console)]
    fn warn(s: &str);
}

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
