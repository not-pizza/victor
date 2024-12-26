use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap, HashMap, HashSet};

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
    similarity,
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

    /// Add many documents to the database.
    /// Embeddings will be generated for each document.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn add_many(
        &mut self,
        content: Vec<impl Into<String>>,
        tags: Vec<impl Into<String>>,
    ) {
        let tags = tags.into_iter().map(|t| t.into()).collect::<Vec<String>>();
        let model = fastembed::TextEmbedding::try_new(Default::default()).unwrap();
        let content = content
            .into_iter()
            .map(|c| c.into())
            .collect::<Vec<String>>();

        let vectors = model.embed(content.clone(), None).unwrap();

        let to_add = content.into_iter().zip(vectors.into_iter()).collect();
        self.add_embedding_many(to_add, tags).await;
    }

    /// Add a single document to the database.
    /// Embedding will be generated for the document.
    /// When adding many documents, it is more efficient to use `add_many`.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn add(&mut self, content: impl Into<String>, tags: Vec<impl Into<String>>) {
        self.add_many(vec![content], tags).await;
    }

    /// Add many documen/embedding pairs to the database.
    /// This is useful for adding embeddings that have already been generated.
    pub async fn add_embedding_many(
        &mut self,
        to_add: Vec<(impl Into<String>, Vec<f32>)>,
        tags: Vec<impl Into<String>>,
    ) {
        let tags = tags.into_iter().map(|t| t.into()).collect::<Vec<String>>();
        let (contents, embeddings) = to_add
            .into_iter()
            .map(|(content, embedding)| {
                let uuid = Uuid::new_v4();
                (
                    (content.into(), uuid),
                    Embedding {
                        id: uuid,
                        vector: embedding,
                    },
                )
            })
            .unzip();

        self.write_embeddings(embeddings, tags).await.unwrap();
        self.write_contents(contents).await.unwrap();
    }

    /// Add a single document/embedding pair to the database.
    /// This is useful for adding embeddings that have already been generated.
    /// When adding many documents, it is more efficient to use `add_embedding_many`.
    pub async fn add_embedding(
        &mut self,
        content: impl Into<String>,
        vector: Vec<f32>,
        tags: Vec<impl Into<String>>,
    ) {
        self.add_embedding_many(vec![(content, vector)], tags).await;
    }

    /// Search the database for the nearest neighbors to a given document.
    /// An embedding will be generated for the document being searched for.
    /// This will return the top `top_n` nearest neighbors.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn search(
        &self,
        content: impl Into<String>,
        with_tags: Vec<impl Into<String>>,
        top_n: u32,
    ) -> Vec<NearestNeighborsResult> {
        let model = fastembed::TextEmbedding::try_new(Default::default()).unwrap();
        let content = content.into();
        let vector = model
            .embed(vec![content.clone()], None)
            .unwrap()
            .first()
            .cloned()
            .unwrap();
        self.search_embedding(vector, with_tags, top_n).await
    }

    /// Search the database for the nearest neighbors to a given embedding.
    /// This will return the top `top_n` nearest neighbors.
    pub async fn search_embedding(
        &self,
        mut vector: Vec<f32>,
        with_tags: Vec<impl Into<String>>,
        top_n: u32,
    ) -> Vec<NearestNeighborsResult> {
        let with_tags = with_tags
            .into_iter()
            .map(|t| t.into())
            .collect::<Vec<String>>();
        let top_n = top_n as usize;
        let with_tags = with_tags.into_iter().collect::<BTreeSet<_>>();
        let file_handles = Index::get_matching_db_files(&self.root, with_tags)
            .await
            .unwrap();

        let is_projected: bool = self
            .root
            .get_file_handle_with_options("eigen.bin", &GetFileHandleOptions { create: false })
            .await
            .is_ok();

        if is_projected {
            let eigen_file = self.eigen_file().await;
            vector = self.project_single_vector(vector, eigen_file);
        }

        let mut nearest_neighbors = BinaryHeap::with_capacity(top_n);
        for file_handle in file_handles {
            let file = file_handle.read().await.unwrap();
            let embeddings = self.get_embeddings_by_file(file).await;

            // find max similarity in this file
            for potential_match in &embeddings {
                let sim = if is_projected {
                    similarity::euclidean(&potential_match.vector, &vector).unwrap()
                } else {
                    similarity::cosine(&potential_match.vector, &vector).unwrap()
                };

                if nearest_neighbors.len() < top_n {
                    let result = NearestNeighborsResult {
                        similarity: sim,
                        embedding: potential_match.clone(),
                        content: self.get_content(potential_match.id).await,
                    };
                    nearest_neighbors.push(Reverse(result));
                } else if sim > nearest_neighbors.peek().unwrap().0.similarity {
                    let result = NearestNeighborsResult {
                        similarity: sim,
                        embedding: potential_match.clone(),
                        content: self.get_content(potential_match.id).await,
                    };
                    nearest_neighbors.pop();
                    nearest_neighbors.push(Reverse(result));
                }
            }
        }

        let mut nearest = nearest_neighbors
            .into_iter()
            .map(|r| r.0)
            .collect::<Vec<_>>();
        nearest.sort();
        nearest.reverse();
        nearest
    }

    // utils

    async fn project_embeddings(&mut self) {
        let prev_embeddings = self.get_all_embeddings().await;

        let (eigenvectors, means) = project_to_lower_dimension(prev_embeddings.clone(), 500);
        let vector_projection = VectorProjection {
            eigen: eigenvectors.clone(),
            means,
        };

        self.write_projection(vector_projection.clone()).await;

        self.update_all_embeddings(vector_projection).await;
    }

    async fn update_all_embeddings(&mut self, vector_projection: VectorProjection) {
        let file_handles = Index::get_matching_db_files(
            &self.root,
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

    async fn get_all_embeddings(&self) -> Vec<Embedding> {
        let file_handles = Index::get_matching_db_files(
            &self.root,
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

    async fn get_embeddings_by_file(&self, file: Vec<u8>) -> Vec<Embedding> {
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

        bincode::deserialize::<u32>(embedding_size_bytes).expect("Failed to deserialize header")
    }

    async fn eigen_file(&self) -> Vec<u8> {
        let eigen_file_handle = self
            .root
            .get_file_handle_with_options("eigen.bin", &GetFileHandleOptions { create: true })
            .await
            .unwrap();

        eigen_file_handle.read().await.unwrap()
    }

    fn project_single_vector(&self, vector: Vec<f32>, eigen_file: Vec<u8>) -> Vec<f32> {
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

    async fn write_embeddings(
        &mut self,
        mut embeddings: Vec<Embedding>,
        tags: Vec<String>,
    ) -> Result<(), D::Error> {
        let mut file_handle = Index::get_exact_db_file(&mut self.root, tags).await?;

        let is_projected: bool = self
            .root
            .get_file_handle_with_options("eigen.bin", &GetFileHandleOptions { create: false })
            .await
            .is_ok();

        if is_projected {
            let eigen_file = self.eigen_file().await;
            embeddings = embeddings
                .into_iter()
                .map(|embedding| {
                    let vector =
                        self.project_single_vector(embedding.vector.clone(), eigen_file.clone());
                    Embedding {
                        id: embedding.id,
                        vector,
                    }
                })
                .collect();
        }

        let mut writable = file_handle
            .create_writable_with_options(&CreateWritableOptions {
                keep_existing_data: true,
            })
            .await?;

        writable.seek(file_handle.size().await?).await?;

        let embeddings_serialized = embeddings
            .into_iter()
            .map(|embedding| bincode::serialize(&embedding).expect("Failed to serialize embedding"))
            .collect::<Vec<_>>();

        // check that the embeddings are all the same size
        // and get that size
        let embedding_size = match &embeddings_serialized
            .iter()
            .map(|embedding| embedding.len())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()[..]
        {
            [size] => *size as u32,
            _ => panic!("All embeddings must be the same size"),
        };

        if file_handle.size().await? == 0 {
            let serialized_size =
                bincode::serialize(&embedding_size).expect("Failed to serialize size");

            writable.write_at_cursor_pos(serialized_size).await?;
        } else {
            let previous_embedding_size = Self::get_embedding_size(file_handle.read().await?);
            assert_eq!(
                embedding_size, previous_embedding_size,
                "Embedding size mismatch: expected {} but got {}",
                previous_embedding_size, embedding_size
            );
        }

        let all_embeddings_serialized = embeddings_serialized
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        writable
            .write_at_cursor_pos(all_embeddings_serialized)
            .await?;

        writable.close().await?;

        if cfg!(target_arch = "wasm32") && file_handle.size().await? > 1000000 && !is_projected {
            self.project_embeddings().await;
        }

        Ok(())
    }

    async fn write_contents(&mut self, content: Vec<(String, Uuid)>) -> Result<(), D::Error> {
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

        for (content, id) in content {
            hashmap.insert(id, content);
        }

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

    async fn get_content(&self, id: Uuid) -> String {
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
    async fn load<D: DirectoryHandle>(root: &D) -> Result<(D::FileHandleT, Self), D::Error> {
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
        root: &D,
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
        root: &D,
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

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct NearestNeighborsResult {
    pub similarity: f32,
    pub embedding: Embedding,
    pub content: String,
}

impl PartialEq for NearestNeighborsResult {
    fn eq(&self, other: &Self) -> bool {
        self.similarity == other.similarity
    }
}

impl Eq for NearestNeighborsResult {}

impl PartialOrd for NearestNeighborsResult {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for NearestNeighborsResult {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.similarity
            .partial_cmp(&other.similarity)
            .expect("could not compare, most likely a NaN is involved")
    }
}
