use rand::Rng;

pub(crate) fn cosine(v1: &[f32], v2: &[f32]) -> Result<f32, String> {
    if v1.len() != v2.len() {
        return Err(format!(
            "Vector lengths do not match: {} != {}",
            v1.len(),
            v2.len()
        ));
    }

    let mut dot_product = 0.0;
    let mut v1_norm = 0.0;
    let mut v2_norm = 0.0;

    for i in 0..v1.len() {
        dot_product += v1[i] * v2[i];
        v1_norm += v1[i] * v1[i];
        v2_norm += v2[i] * v2[i];
    }

    v1_norm = v1_norm.sqrt();
    v2_norm = v2_norm.sqrt();

    Ok(dot_product / (v1_norm * v2_norm))
}

pub(crate) fn euclidean(v1: &[f32], v2: &[f32]) -> Result<f32, String> {
    if v1.len() != v2.len() {
        return Err(format!(
            "Vector lengths do not match: {} != {}",
            v1.len(),
            v2.len()
        ));
    }

    let mut sum_of_squares = 0.0;

    for i in 0..v1.len() {
        let difference = v1[i] - v2[i];
        sum_of_squares += difference * difference;
    }

    Ok(sum_of_squares.sqrt())
}

pub(crate) fn kmeans(
    data: &[Vec<f32>],
    k: usize,
    max_iter: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<&Vec<f32>>>) {
    let mut rng = rand::thread_rng();
    let num_dims = data[0].len();

    // instead of creating a hashmap, centroid[0] corresponds to cluster[0]
    // this is to avoid hashing errors when using f32 as keys
    let mut centroids: Vec<Vec<f32>> = Vec::new();
    let mut clusters: Vec<Vec<&Vec<f32>>> = vec![Vec::new(); k];

    // randomly select k centroids
    let mut selected_indices = Vec::new();
    while centroids.len() < k {
        let random_index = rng.gen_range(0..data.len());
        if !selected_indices.contains(&random_index) {
            centroids.push(data[random_index].clone());
            selected_indices.push(random_index);
        }
    }

    for _ in 0..max_iter {
        // assign each data point to the nearest centroid
        for point in data {
            let mut min_distance = std::f32::MAX;
            let mut closest_centroid = 0;
            for (i, centroid) in centroids.iter().enumerate() {
                let distance = euclidean(&point, centroid).unwrap();
                if distance < min_distance {
                    min_distance = distance;
                    closest_centroid = i;
                }
            }
            clusters[closest_centroid].push(point);
        }

        // move the centroids to the center of their clusters
        for (i, cluster) in clusters.iter().enumerate() {
            if cluster.is_empty() {
                continue;
            }

            let cluster_size = cluster.len() as f32;
            // sum of each dimensions across all vectors in the cluster
            let sums = cluster
                .iter()
                .fold(vec![0f32; num_dims], |mut acc, &point| {
                    for (i, val) in point.iter().enumerate() {
                        acc[i] += val;
                    }
                    acc
                });

            // get the mean of each dimension
            let new_centroid = sums.iter().map(|&sum| sum / cluster_size).collect();

            // update the centroid to have the mean of each dimension
            centroids[i] = new_centroid;
        }
    }
    (centroids, clusters)
}

#[test]
fn cosine_test() {
    let v1 = vec![1.0, 2.0, 3.0];
    let v2 = vec![3.0, 2.0, 1.0];
    let result = cosine(&v1, &v2).unwrap();
    let expected = 0.7142857;
    assert!(
        (result - expected).abs() < 0.001,
        "result ({}) != expected ({})",
        result,
        expected
    );
}

#[test]
fn cosine_test_same() {
    let v1 = vec![1.0, 2.0, 3.0];
    let v2 = vec![1.0, 2.0, 3.0];
    let result = cosine(&v1, &v2).unwrap();
    let expected = 1.0;
    assert!(
        (result - expected).abs() < 0.001,
        "result ({}) != expected ({})",
        result,
        expected
    );
}

#[test]
fn cosine_test_opposite() {
    let v1 = vec![1.0, 2.0, 3.0];
    let v2 = vec![-1.0, -2.0, -3.0];
    let result = cosine(&v1, &v2).unwrap();
    let expected = -1.0;
    assert!(
        (result - expected).abs() < 0.001,
        "result ({}) != expected ({})",
        result,
        expected
    );
}
