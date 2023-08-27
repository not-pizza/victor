mod filesystem;
mod similarity;
mod utils;

use filesystem::{
    web, CreateWritableOptions, DirectoryHandle, FileHandle, GetFileHandleOptions,
    WritableFileStream,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use wasm_bindgen::prelude::*;
use web_sys::FileSystemDirectoryHandle;

#[derive(Serialize, Deserialize, Debug)]
struct Embedding {
    pub id: Uuid,
    pub vector: Vec<f32>,
}

#[derive(Serialize, Deserialize, Debug)]
struct Content {
    pub id: Uuid,
    pub content: String,
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

async fn write_to_victor(root: FileSystemDirectoryHandle, embedding: &[f64], id: Uuid) {
    let mut root = web::DirectoryHandle::from(root);

    let mut victor_file_handle = root
        .get_file_handle_with_options("victor.bin", &GetFileHandleOptions { create: true })
        .await
        .unwrap();

    let mut victor_writable = victor_file_handle
        .create_writable_with_options(&CreateWritableOptions {
            keep_existing_data: true,
        })
        .await
        .unwrap();

    let victor_offset = victor_file_handle.get_size().await.unwrap();

    victor_writable.seek(victor_offset).await.unwrap();

    let embedding = Embedding {
        id,
        vector: embedding.iter().map(|x| *x as f32).collect(),
    };

    let embedding = bincode::serialize(&embedding).expect("Failed to serialize embedding");

    victor_writable
        .write_at_cursor_pos(embedding)
        .await
        .unwrap();

    victor_writable.close().await.unwrap();
}

async fn write_to_content(root: FileSystemDirectoryHandle, content: &str, id: Uuid) {
    let mut root = web::DirectoryHandle::from(root);

    let mut content_file_handle = root
        .get_file_handle_with_options("content.bin", &GetFileHandleOptions { create: true })
        .await
        .unwrap();

    let existing_content = content_file_handle.read().await.unwrap();

    let mut hashmap: HashMap<Uuid, String> = if existing_content.is_empty() {
        HashMap::new()
    } else {
        bincode::deserialize(&existing_content).expect("Failed to deserialize existing data")
    };

    hashmap.insert(id, content.to_string());

    let updated_data = bincode::serialize(&hashmap).expect("Failed to serialize hashmap");

    let mut content_writable = content_file_handle
        .create_writable_with_options(&CreateWritableOptions {
            keep_existing_data: true,
        })
        .await
        .unwrap();

    content_writable
        .write_at_cursor_pos(updated_data)
        .await
        .unwrap();

    content_writable.close().await.unwrap();
}

async fn get_content(mut root: impl DirectoryHandle, id: Uuid) -> String {
    let content_file_handle = root
        .get_file_handle_with_options("content.bin", &GetFileHandleOptions { create: true })
        .await
        .unwrap();

    let existing_content = content_file_handle.read().await.unwrap();

    let hashmap: HashMap<Uuid, String> =
        bincode::deserialize(&existing_content).expect("Failed to deserialize existing data");

    let content = hashmap.get(&id).unwrap();

    content.to_string()
}

/// Assumes all the embeddings are the size of `embedding`
/// TODO: Record the embedding size somewhere so we can return an error if
/// the sizes are wrong (as otherwise this will corrupt the entire db)
#[wasm_bindgen]
pub async fn write_embedding(root: FileSystemDirectoryHandle, embedding: &[f64], content: &str) {
    utils::set_panic_hook();

    let id = Uuid::new_v4();

    write_to_victor(root.clone(), embedding, id).await;
    write_to_content(root.clone(), content, id).await;
}

/// Assumes all the embeddings are the size of `embedding`
#[wasm_bindgen]
pub async fn find_nearest_neighbors(root: FileSystemDirectoryHandle, vector: &[f64]) {
    utils::set_panic_hook();

    let vector = vector.iter().map(|x| *x as f32).collect::<Vec<_>>();

    let mut root = web::DirectoryHandle::from(root);

    let file_handle = root
        .get_file_handle_with_options("victor.bin", &GetFileHandleOptions { create: true })
        .await
        .unwrap();

    // Serialize the given embedding to get the size
    let embedding_size = {
        let embedding = Embedding {
            id: Uuid::new_v4(),
            vector: vector.clone(),
        };
        let embedding_bytes =
            bincode::serialize(&embedding).expect("Failed to serialize embedding");
        embedding_bytes.len()
    };

    let file = file_handle.read().await.unwrap();

    // sanity check
    {
        let file_size = file.len();
        assert_eq!(
            file_size % embedding_size,
            0,
            "file_size ({}) was not a multiple of embedding_size ({embedding_size})",
            file_size
        );
        console_log!("File looks ok");
    }

    let embeddings = file
        .chunks(embedding_size)
        .map(|chunk| bincode::deserialize::<Embedding>(chunk).unwrap());

    // find max similarity
    let nearest = embeddings.max_by_key(|potential_match| {
        (similarity::cosine(potential_match.vector.clone(), vector.clone()).unwrap() * 1000.0)
            as i32
    });

    if let Some(nearest) = nearest {
        let embedding_id = nearest.id;

        let content = get_content(root, embedding_id).await;
        console_log!("nearest: {:?}", content);
    } else {
        // idk how this could run
        console_log!("No nearest neighbor found");
    }
}
