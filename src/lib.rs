mod filesystem;
mod similarity;

use filesystem::DirectoryHandle;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use wasm_bindgen::prelude::*;
use web_sys::{
    FileSystemCreateWritableOptions, FileSystemDirectoryHandle, FileSystemGetFileOptions,
};

#[derive(Serialize, Deserialize, Debug)]
struct Embedding {
    pub id: Uuid,
    pub vector: Vec<f64>,
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

/// Assumes all the embeddings are the size of `embedding`
/// TODO: Record the embedding size somewhere so we can return an error if
/// the sizes are wrong (as otherwise this will corrupt the entire db)
#[wasm_bindgen]
pub async fn write_embedding(root: FileSystemDirectoryHandle, embedding: &[f64]) {
    let root = DirectoryHandle::from(root);

    let file_handle = root
        .get_file_handle_with_options("victor.bin", FileSystemGetFileOptions::new().create(true))
        .await
        .unwrap();

    console_log!("File handle: {:?}", file_handle);

    let writable = file_handle
        .create_writable_with_options(
            FileSystemCreateWritableOptions::new().keep_existing_data(true),
        )
        .await
        .unwrap();

    let offset = file_handle.get_size().await.unwrap();

    console_log!("offset: {:?}", offset);

    writable.seek(offset).await.unwrap();

    let embedding = Embedding {
        id: Uuid::new_v4(),
        vector: embedding.iter().map(|x| *x).collect(),
    };

    let mut embedding = bincode::serialize(&embedding).expect("Failed to serialize embedding");

    writable.write_with_u8_array(&mut embedding).await.unwrap();

    writable.close().await.unwrap();
}

/// Assumes all the embeddings are the size of `embedding`
#[wasm_bindgen]
pub async fn find_nearest_neighbors(root: FileSystemDirectoryHandle, embedding: &[f64]) -> () {
    let root = DirectoryHandle::from(root);

    let file_handle = root
        .get_file_handle_with_options("victor.bin", FileSystemGetFileOptions::new().create(true))
        .await
        .unwrap();

    // Serialize the given embedding to get the size
    let embedding_size = {
        let embedding = bincode::serialize(&embedding).expect("Failed to serialize embedding");
        embedding.len()
    };

    // sanity check
    let file = file_handle.read().await.unwrap();
    {
        let file_size = file.len();
        assert_eq!(
            file_size as usize % embedding_size,
            0,
            "file_size ({}) was not a multiple of embedding_size ({embedding_size})",
            file_size
        );
    }
}
