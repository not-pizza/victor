use std::collections::{BTreeSet, HashMap, HashSet};

use serde::{Deserialize, Serialize};
use sha256::digest;
use uuid::Uuid;

use crate::{
    filesystem::{
        CreateWritableOptions, DirectoryHandle, FileHandle, GetFileHandleOptions,
        WritableFileStream,
    },
    similarity,
};

pub struct Victor<D> {
    root: D,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Embedding {
    pub id: Uuid,
    #[serde(
        serialize_with = "crate::packed_vector::PackedVector::serialize_embedding",
        deserialize_with = "crate::packed_vector::PackedVector::deserialize_embedding"
    )]
    pub embedding: Vec<f32>,
}

#[derive(Serialize, Deserialize, Debug, Eq, PartialEq)]
pub struct Content {
    pub id: Uuid,
    pub content: String,
}

#[derive(Serialize, Deserialize, Debug, Default, Eq, PartialEq)]
pub struct Index {
    files: HashSet<BTreeSet<String>>,
}

impl<D: DirectoryHandle> Victor<D> {
    pub(crate) fn new(root: D) -> Self {
        Self { root }
    }

    pub(crate) async fn write(&mut self, embedding: Vec<f32>, content: &str, tags: Vec<String>) {
        let id = Uuid::new_v4();

        let embedding = Embedding { id, embedding };

        self.write_embedding(embedding, tags).await.unwrap();
        self.write_content(content, id).await.unwrap();
    }

    pub(crate) async fn find_nearest_neighbor(
        &mut self,
        vector: Vec<f32>,
        with_tags: Vec<String>,
    ) -> Option<Content> {
        let with_tags = with_tags.into_iter().collect::<BTreeSet<_>>();
        let file_handles = Index::get_matching_db_files(&mut self.root, with_tags)
            .await
            .unwrap();

        let mut nearest: Option<Embedding> = None;
        for file_handle in file_handles {
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
            let nearest_in_file = embeddings.max_by_key(|potential_match| {
                (similarity::cosine(potential_match.embedding.clone(), vector.clone()).unwrap()
                    * 1000.0) as i32
            });
            match (&mut nearest, nearest_in_file) {
                (Some(ref mut nearest), Some(nearest_in_file))
                    if similarity::cosine(nearest.embedding.clone(), vector.clone()).unwrap()
                        < similarity::cosine(nearest_in_file.embedding.clone(), vector.clone())
                            .unwrap() =>
                {
                    *nearest = nearest_in_file
                }
                (None, nearest_in_file) => nearest = nearest_in_file,
                _ => {}
            }
        }

        if let Some(nearest) = nearest {
            let content = self.get_content(nearest.id).await;

            Some(Content {
                id: nearest.id,
                content,
            })
        } else {
            None
        }
    }

    // utils

    async fn write_embedding(
        &mut self,
        embedding: Embedding,
        tags: Vec<String>,
    ) -> Result<(), D::Error> {
        let mut file_handle = Index::get_exact_db_file(&mut self.root, tags).await?;

        let mut writable = file_handle
            .create_writable_with_options(&CreateWritableOptions {
                keep_existing_data: true,
            })
            .await?;

        writable.seek(file_handle.size().await?).await?;

        let embedding = bincode::serialize(&embedding).expect("Failed to serialize embedding");

        writable.write_at_cursor_pos(embedding).await?;
        writable.close().await?;

        Ok(())
    }

    async fn write_content(&mut self, content: &str, id: Uuid) -> Result<(), D::Error> {
        let mut content_file_handle = self
            .root
            .get_file_handle_with_options("content.bin", &GetFileHandleOptions { create: true })
            .await?;

        let existing_content = content_file_handle.read().await?;

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
            .await?;

        content_writable.write_at_cursor_pos(updated_data).await?;
        content_writable.close().await?;

        Ok(())
    }

    async fn get_content(&mut self, id: Uuid) -> String {
        let content_file_handle = self
            .root
            .get_file_handle_with_options("content.bin", &GetFileHandleOptions { create: true })
            .await
            .unwrap();

        let existing_content = content_file_handle.read().await.unwrap();

        let hashmap: HashMap<Uuid, String> =
            bincode::deserialize(&existing_content).expect("Failed to deserialize existing data");

        let content = hashmap.get(&id).unwrap();

        content.to_string()
    }

    pub(crate) async fn clear_db(&mut self) -> Result<(), D::Error> {
        // clear db files
        let files = Index::get_all_db_filenames(&mut self.root).await?;
        for file in files {
            self.root.remove_entry(&file).await?;
        }

        // clear index file
        self.root.remove_entry("index.bin").await?;

        // clear content file
        self.root.remove_entry("content.bin").await?;

        Ok(())
    }
}

impl Index {
    async fn load<D: DirectoryHandle>(root: &mut D) -> Result<(D::FileHandleT, Self), D::Error> {
        let file_handle = root
            .get_file_handle_with_options("index.bin", &GetFileHandleOptions { create: true })
            .await?;

        if file_handle.size().await? == 0 {
            let index = Self::default();
            Ok((file_handle, index))
        } else {
            let index_bytes = file_handle.read().await?;
            let index =
                bincode::deserialize::<Self>(&index_bytes).expect("Failed to deserialize index");
            Ok((file_handle, index))
        }
    }

    fn filename_for_tags(tags: BTreeSet<String>) -> String {
        let mut tags = tags.into_iter().collect::<Vec<_>>();
        tags.sort();
        let input = format!("{:?}", tags);
        format!("{}.bin", digest(input))
    }

    async fn file_handle_for_tag<D: DirectoryHandle>(
        root: &mut D,
        tags: BTreeSet<String>,
    ) -> Result<D::FileHandleT, D::Error> {
        // Get the filename by just hashing the tags
        let filename = Self::filename_for_tags(tags);

        root.get_file_handle_with_options(&filename, &GetFileHandleOptions { create: true })
            .await
    }

    async fn get_exact_db_file<D: DirectoryHandle>(
        root: &mut D,
        tags: Vec<String>,
    ) -> Result<D::FileHandleT, D::Error> {
        let (mut index_file, mut index) = Self::load(root).await?;
        let tags = tags.into_iter().collect::<BTreeSet<_>>();

        // If the set of tags isn't in the index, add it
        if !index.files.contains(&tags) {
            index.files.insert(tags.clone());

            let index_bytes = bincode::serialize(&index).expect("Failed to serialize index");
            let mut writable = index_file
                .create_writable_with_options(&CreateWritableOptions {
                    keep_existing_data: false,
                })
                .await?;
            writable.write_at_cursor_pos(index_bytes).await?;
            writable.close().await?;
        }

        Self::file_handle_for_tag(root, tags).await
    }

    async fn get_matching_db_files<D: DirectoryHandle>(
        root: &mut D,
        tags: BTreeSet<String>,
    ) -> Result<Vec<D::FileHandleT>, D::Error> {
        let (_, index) = Self::load(root).await?;

        let matching_tags = index
            .files
            .iter()
            .filter(|file_tags| file_tags.is_superset(&tags))
            .cloned();

        let mut files = Vec::new();
        for tags in matching_tags {
            let file = Self::file_handle_for_tag(root, tags.clone()).await?;
            files.push(file)
        }

        Ok(files)
    }

    async fn get_all_db_filenames<D: DirectoryHandle>(
        root: &mut D,
    ) -> Result<Vec<String>, D::Error> {
        let (_, index) = Self::load(root).await?;

        Ok(index
            .files
            .into_iter()
            .map(Self::filename_for_tags)
            .collect())
    }
}
