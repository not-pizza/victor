use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    FileSystemDirectoryHandle, FileSystemFileHandle, FileSystemGetFileOptions,
    FileSystemWritableFileStream,
};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use uuid::Uuid;
use wasm_bindgen::prelude::*;

#[derive(Serialize, Deserialize, Debug)]
struct Embedding {
    pub id: Uuid,
    pub vector: Vec<f32>,
    pub metadata: Option<HashMap<String, String>>,
}

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

#[wasm_bindgen]
pub async fn embed(root: FileSystemDirectoryHandle, embedding: &[f64]) {
    console_log!("are we get to file handle?");

    let file_handle: FileSystemFileHandle =
        JsFuture::from(root.get_file_handle_with_options(
            "victor.bin",
            FileSystemGetFileOptions::new().create(true),
        ))
        .await
        .unwrap()
        .into();

    console_log!("{:?}", file_handle);

    let writable = FileSystemWritableFileStream::unchecked_from_js(
        JsFuture::from(file_handle.create_writable()).await.unwrap(),
    );

    console_log!("embedding: {:?}", embedding);

    JsFuture::from(writable.write_with_str("init db contents").unwrap())
        .await
        .unwrap();
    JsFuture::from(writable.close()).await.unwrap();
}
