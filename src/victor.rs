use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    filesystem::{
        CreateWritableOptions, DirectoryHandle, FileHandle, GetFileHandleOptions,
        WritableFileStream,
    },
    similarity,
};

#[derive(Serialize, Deserialize, Debug)]
pub struct Embedding {
    pub id: Uuid,
    pub embedding: Vec<f32>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Content {
    pub id: Uuid,
    pub content: String,
}

pub(crate) async fn write(mut root: impl DirectoryHandle, embedding: Vec<f32>, content: &str) {
    let id = Uuid::new_v4();

    let embedding = Embedding { id, embedding };

    write_embedding(&mut root, embedding).await;
    write_content(&mut root, content, id).await;
}

pub(crate) async fn find_nearest_neighbor(
    mut root: impl DirectoryHandle,
    vector: Vec<f32>,
) -> Option<Content> {
    let file_handle = root
        .get_file_handle_with_options("victor.bin", &GetFileHandleOptions { create: true })
        .await
        .unwrap();

    // Serialize the given embedding to get the size
    let embedding_size = {
        let embedding = Embedding {
            id: Uuid::new_v4(),
            embedding: vector.clone(),
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
    }

    let embeddings = file
        .chunks(embedding_size)
        .map(|chunk| bincode::deserialize::<Embedding>(chunk).unwrap());

    // find max similarity
    let nearest = embeddings.max_by_key(|potential_match| {
        (similarity::cosine(potential_match.embedding.clone(), vector.clone()).unwrap() * 1000.0)
            as i32
    });

    if let Some(nearest) = nearest {
        let content = get_content(root, nearest.id).await;

        Some(Content {
            id: nearest.id,
            content,
        })
    } else {
        None
    }
}

// utils

async fn write_embedding(root: &mut impl DirectoryHandle, embedding: Embedding) {
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

    let embedding = bincode::serialize(&embedding).expect("Failed to serialize embedding");

    victor_writable
        .write_at_cursor_pos(embedding)
        .await
        .unwrap();

    victor_writable.close().await.unwrap();
}

async fn write_content(root: &mut impl DirectoryHandle, content: &str, id: Uuid) {
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
