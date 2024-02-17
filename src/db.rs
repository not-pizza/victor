use std::collections::{BTreeSet, HashMap, HashSet};

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};
use sha256::digest;
use uuid::Uuid;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::wasm_bindgen;

use crate::decomposition::{center_data, embeddings_to_dmatrix, project_to_lower_dimension};

use crate::{
    filesystem::{
        CreateWritableOptions, DirectoryHandle, FileHandle, GetFileHandleOptions,
        WritableFileStream,
    },
    gpu, similarity,
};

pub struct Victor<D> {
    root: D,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Embedding {
    pub id: Uuid,
    #[serde(
        serialize_with = "crate::packed_vector::PackedVector::serialize_embedding",
        deserialize_with = "crate::packed_vector::PackedVector::deserialize_embedding"
    )]
    pub vector: Vec<f32>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct VectorProjection {
    pub eigen: DMatrix<f32>,
    pub means: Vec<f32>,
}

#[derive(Serialize, Deserialize, Debug, Eq, PartialEq)]
pub struct Content {
    pub id: Uuid,
    pub content: String,
}

#[derive(Serialize, Deserialize, Debug, Default, Eq, PartialEq, Clone)]
pub struct Index {
    files: HashSet<BTreeSet<String>>,
}

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

impl<D: DirectoryHandle> Victor<D> {
    pub fn new(root: impl Into<D>) -> Self {
        let root = root.into();
        Self { root }
    }

    pub async fn write(&mut self, content: impl Into<String>, vector: Vec<f32>, tags: Vec<String>) {
        let content = content.into();

        let id = Uuid::new_v4();

        let embedding = Embedding { id, vector };

        self.write_embedding(embedding, tags).await.unwrap();
        self.write_content(content, id).await.unwrap();
    }

    pub async fn find_nearest_neighbor(
        &mut self,
        mut vector: Vec<f32>,
        with_tags: Vec<String>,
    ) -> Option<Content> {
        let with_tags = with_tags.into_iter().collect::<BTreeSet<_>>();
        let file_handles = Index::get_matching_db_files(&mut self.root, with_tags)
            .await
            .unwrap();

        let is_projected: bool = self
            .root
            .get_file_handle_with_options("eigen.bin", &GetFileHandleOptions { create: false })
            .await
            .is_ok();

        let mut nearest_similarity: Option<f32> = None;
        let mut nearest_embedding: Option<Embedding> = None;

        if is_projected {
            vector = self.project_single_vector(vector).await;
        }

        for file_handle in file_handles {
            let file = file_handle.read().await.unwrap();
            let embeddings = self.get_embeddings_by_file(file).await;

            // find max similarity in this file
            for potential_match in &embeddings {
                let sim;
                if is_projected {
                    sim = similarity::euclidean(&potential_match.vector, &vector).unwrap();
                } else {
                    sim = similarity::cosine(&potential_match.vector, &vector).unwrap();
                }
                if nearest_similarity.is_none() || sim > nearest_similarity.unwrap() {
                    nearest_similarity = Some(sim);
                    nearest_embedding = Some(potential_match.clone());
                }
            }
        }

        if let Some(nearest) = nearest_embedding {
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

    async fn project_embeddings(&mut self) {
        let prev_embeddings = self.get_all_embeddings().await;

        let (eigenvectors, means) = project_to_lower_dimension(prev_embeddings.clone(), 500);
        let vector_projection: VectorProjection = VectorProjection {
            eigen: eigenvectors.clone(),
            means: means,
        };

        self.write_projection(vector_projection.clone()).await;

        self.update_all_embeddings(vector_projection).await;
    }

    async fn update_all_embeddings(&mut self, vector_projection: VectorProjection) {
        let file_handles = Index::get_matching_db_files(
            &mut self.root,
            Vec::new().into_iter().collect::<BTreeSet<_>>(),
        )
        .await
        .unwrap();

        for mut file_handle in file_handles {
            let file = file_handle.read().await.unwrap();
            // need to accumulate these over all the indices
            let embeddings = self.get_embeddings_by_file(file).await;
            let matrix = embeddings_to_dmatrix(
                embeddings
                    .clone()
                    .into_iter()
                    .map(|embedding| embedding.vector)
                    .collect(),
            );
            let (centered_data, _) = center_data(&matrix);

            let projected_data = centered_data * &vector_projection.eigen;

            let projected_vectors: Vec<Vec<f32>> = projected_data
                .row_iter()
                .map(|row| row.iter().cloned().collect())
                .collect();

            let new_embeddings: Vec<Embedding> = embeddings
                .iter()
                .enumerate()
                .map(|(index, embedding)| Embedding {
                    id: embedding.id,
                    vector: projected_vectors[index].clone(),
                })
                .collect();

            gpu::GLOBAL_WGPU.with(|g| {
                if let Some(g) = &*g.borrow() {
                    let device = &g.device;
                    let queue = &g.queue;
                    let pipeline = &g.pipeline;

                    gpu::load_embeddings_gpu(device, embeddings);
                }
            });

            let len_as_u32 = bincode::serialize(&new_embeddings[0])
                .expect("Failed to serialize embeddings")
                .len() as u32;

            let serialized_size =
                bincode::serialize(&len_as_u32).expect("Failed to serialize size");

            let serialized_embeddings =
                bincode::serialize(&new_embeddings).expect("Failed to serialize embeddings");

            let mut writable = file_handle
                .create_writable_with_options(&CreateWritableOptions {
                    keep_existing_data: false,
                })
                .await
                .unwrap();

            let mut combined = serialized_size;
            combined.extend(
                &serialized_embeddings
                    [bincode::serialized_size(&Vec::<Embedding>::new()).unwrap() as usize..],
            );

            writable.seek(0).await.unwrap();

            writable.write_at_cursor_pos(combined).await.unwrap();

            writable.close().await.unwrap();
        }
    }

    async fn write_projection(&mut self, vector_projection: VectorProjection) {
        let mut eigen_file_handle = self
            .root
            .get_file_handle_with_options("eigen.bin", &GetFileHandleOptions { create: true })
            .await
            .unwrap();

        let mut writable = eigen_file_handle
            .create_writable_with_options(&CreateWritableOptions {
                keep_existing_data: false,
            })
            .await
            .unwrap();

        let vector_projection_bytes =
            bincode::serialize(&vector_projection).expect("Failed to serialize embedding");

        writable
            .write_at_cursor_pos(vector_projection_bytes)
            .await
            .unwrap();

        writable.close().await.unwrap();
    }

    async fn get_all_embeddings(&mut self) -> Vec<Embedding> {
        let file_handles = Index::get_matching_db_files(
            &mut self.root,
            Vec::new().into_iter().collect::<BTreeSet<_>>(),
        )
        .await
        .unwrap();

        let mut prev_embeddings: Vec<Embedding> = Vec::new();

        for file_handle in file_handles {
            let file = file_handle.read().await.unwrap();
            let mut embeddings = self.get_embeddings_by_file(file).await;
            prev_embeddings.append(&mut embeddings);
        }

        prev_embeddings
    }

    async fn get_embeddings_by_file(&mut self, file: Vec<u8>) -> Vec<Embedding> {
        let header_size = std::mem::size_of::<u32>();

        let embedding_size: u32 = Self::get_embedding_size(file.clone());

        let file_content = &file[header_size..];

        // sanity check
        {
            let file_size = file_content.len() as u32;
            assert_eq!(
                file_size % embedding_size,
                0,
                "file_size ({file_size} after subtracting header size {header_size}) was not a multiple of embedding_size ({embedding_size})",
            );
        }

        let embeddings = file_content
            .chunks(embedding_size as usize)
            .map(|chunk| bincode::deserialize::<Embedding>(chunk).unwrap());

        embeddings.collect()
    }

    fn get_embedding_size(file: Vec<u8>) -> u32 {
        // Read the embedding size from the header.
        let header_size = std::mem::size_of::<u32>(); // Assuming your header is u32

        let embedding_size_bytes = &file[0..header_size];
        let embedding_size = bincode::deserialize::<u32>(embedding_size_bytes)
            .expect("Failed to deserialize header");

        embedding_size
    }

    async fn project_single_vector(&mut self, vector: Vec<f32>) -> Vec<f32> {
        let eigen_file_handle = self
            .root
            .get_file_handle_with_options("eigen.bin", &GetFileHandleOptions { create: true })
            .await
            .unwrap();

        let eigen_file = eigen_file_handle.read().await.unwrap();
        let vector_projection: VectorProjection = bincode::deserialize(&eigen_file).unwrap();

        let centered_vector = vector
            .iter()
            .zip(vector_projection.means.iter())
            .map(|(x, mean)| x - mean)
            .collect::<Vec<_>>();

        let centered_matrix = embeddings_to_dmatrix(vec![centered_vector]);

        let projected_vector = (centered_matrix * vector_projection.eigen)
            .as_mut_slice()
            .to_vec();
        projected_vector
    }

    async fn write_embedding(
        &mut self,
        mut embedding: Embedding,
        tags: Vec<String>,
    ) -> Result<(), D::Error> {
        let mut file_handle = Index::get_exact_db_file(&mut self.root, tags).await?;

        let is_projected: bool = self
            .root
            .get_file_handle_with_options("eigen.bin", &GetFileHandleOptions { create: false })
            .await
            .is_ok();

        if is_projected {
            let vector = self.project_single_vector(embedding.vector.clone()).await;
            embedding = Embedding {
                id: embedding.id,
                vector,
            };
        }

        let mut writable = file_handle
            .create_writable_with_options(&CreateWritableOptions {
                keep_existing_data: true,
            })
            .await?;

        writable.seek(file_handle.size().await?).await?;

        let embedding_serialized =
            bincode::serialize(&embedding).expect("Failed to serialize embedding");

        if file_handle.size().await? == 0 {
            let len_as_u32: u32 = embedding_serialized.len() as u32;

            let serialized_size =
                bincode::serialize(&len_as_u32).expect("Failed to serialize size");

            writable.write_at_cursor_pos(serialized_size).await?;
        } else {
            let embedding_size = Self::get_embedding_size(file_handle.read().await?);
            if embedding_serialized.len() as u32 != embedding_size {
                panic!(
                    "Embedding size mismatch: expected {} but got {}",
                    embedding_size,
                    embedding_serialized.len()
                );
            }
        }

        writable.write_at_cursor_pos(embedding_serialized).await?;
        writable.close().await?;

        if file_handle.size().await? > 1000 && !is_projected {
            self.project_embeddings().await;
        }

        Ok(())
    }

    async fn write_content(&mut self, content: String, id: Uuid) -> Result<(), D::Error> {
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

        hashmap.insert(id, content);

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

    pub async fn clear_db(&mut self) -> Result<(), D::Error> {
        // clear db files
        let files = Index::get_all_db_filenames(&mut self.root).await?;
        for file in files {
            self.root.remove_entry(&file).await?;
        }

        // clear index file
        let _ = self.root.remove_entry("index.bin").await;

        // clear content file
        let _ = self.root.remove_entry("content.bin").await;

        // clear content file
        let _ = self.root.remove_entry("eigen.bin").await;

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
