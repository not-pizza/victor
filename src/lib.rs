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

mod filesystem;
use filesystem::DirectoryHandle;

#[derive(Serialize, Deserialize, Debug)]
struct Embedding {
    pub id: Uuid,
    pub vector: Vec<f64>,
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
    let root = DirectoryHandle::from(root);

    let file_handle = root
        .get_file_handle_with_options("victor.bin", FileSystemGetFileOptions::new().create(true))
        .await
        .unwrap();

    console_log!("File handle: {:?}", file_handle);

    let writable = file_handle.create_writable().await.unwrap();

    console_log!("embedding: {:?}", embedding);

    let embedding = Embedding {
        id: Uuid::new_v4(),
        vector: embedding.iter().map(|x| *x).collect(),
        metadata: None,
    };

    let mut embedding = bincode::serialize(&embedding).expect("Failed to serialize embedding");

    writable.write_with_u8_array(&mut embedding).await.unwrap();

    writable.close().await.unwrap();
}
