use nalgebra::{DMatrix, DVector};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::wasm_bindgen;

use crate::db::Embedding;

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

pub fn embeddings_to_dmatrix(embeddings: Vec<Vec<f32>>) -> DMatrix<f32> {
    // Get the number of rows and columns
    let nrows = embeddings.len();
    let ncols = embeddings[0].len();

    // Flatten all vectors into a single Vec<f32>
    let data: Vec<f32> = embeddings.into_iter().flatten().collect();

    // Convert the data into a DMatrix
    DMatrix::from_row_slice(nrows, ncols, &data)
}

pub fn center_data(matrix: &DMatrix<f32>) -> (DMatrix<f32>, Vec<f32>) {
    let means: Vec<f32> = (0..matrix.ncols())
        .map(|col_index| matrix.column(col_index).mean())
        .collect();

    let centered_data: DMatrix<f32> = matrix.map_with_location(|_r, c, val| val - means[c]);

    (centered_data, means)
}

fn compute_covariance_matrix(matrix: &DMatrix<f32>) -> DMatrix<f32> {
    let n_samples = matrix.nrows() as f32;
    let matrix_transposed = matrix.transpose();

    matrix_transposed * matrix / n_samples
}

fn compute_eigenvectors_and_eigenvalues(matrix: &DMatrix<f32>) -> (DVector<f32>, DMatrix<f32>) {
    let eig = matrix.clone().symmetric_eigen();
    (eig.eigenvalues, eig.eigenvectors)
}

fn sort_eigenvectors_and_eigenvalues(
    eigenvalues: DVector<f32>,
    eigenvectors: DMatrix<f32>,
) -> (DVector<f32>, DMatrix<f32>) {
    // Pair each eigenvalue with its corresponding eigenvector column.
    let mut pairs: Vec<(f32, DVector<f32>)> = eigenvalues
        .iter()
        .zip(eigenvectors.column_iter())
        .map(|(&val, vec)| (val, vec.clone_owned()))
        .collect();

    // Sort pairs in descending order based on the eigenvalues.
    pairs.sort_by(|(val1, _vec1), (val2, _vec2)| val2.partial_cmp(val1).unwrap());

    // Unzip the sorted pairs.
    let (sorted_eigenvalues, sorted_eigenvectors_list): (Vec<_>, Vec<_>) =
        pairs.into_iter().unzip();

    // Convert the vectors of sorted eigenvalues and eigenvectors into nalgebra structures.
    let sorted_eigenvalues = DVector::from_vec(sorted_eigenvalues);
    let sorted_eigenvectors = DMatrix::from_columns(&sorted_eigenvectors_list);

    (sorted_eigenvalues, sorted_eigenvectors)
}

pub fn project_to_lower_dimension(data: Vec<Embedding>, k: usize) -> (DMatrix<f32>, Vec<f32>) {
    let matrix =
        embeddings_to_dmatrix(data.into_iter().map(|embedding| embedding.vector).collect());

    let (centered_data, means) = center_data(&matrix);
    let covariance_matrix = compute_covariance_matrix(&centered_data);

    let (eigenvalues, eigenvectors) = compute_eigenvectors_and_eigenvalues(&covariance_matrix);
    let (_sorted_eigenvalues, sorted_eigenvectors) =
        sort_eigenvectors_and_eigenvalues(eigenvalues, eigenvectors);

    let top_k_eigenvectors = sorted_eigenvectors.columns(0, k);

    (top_k_eigenvectors.into(), means)
}
