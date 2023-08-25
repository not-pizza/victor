pub(crate) fn cosine(v1: Vec<f32>, v2: Vec<f32>) -> Result<f32, String> {
    // written by copilot
    // wtf
    // seems right from looking at wikipedia
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

#[test]
fn cosine_test() {
    let v1 = vec![1.0, 2.0, 3.0];
    let v2 = vec![3.0, 2.0, 1.0];
    let result = cosine(v1, v2).unwrap();
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
    let result = cosine(v1, v2).unwrap();
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
    let result = cosine(v1, v2).unwrap();
    let expected = -1.0;
    assert!(
        (result - expected).abs() < 0.001,
        "result ({}) != expected ({})",
        result,
        expected
    );
}
