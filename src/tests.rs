use crate::{filesystem::memory, victor};

#[tokio::test]
async fn store_and_retrieve() {
    let vector = vec![1.0, 2.0, 3.0];

    let mut root = memory::DirectoryHandle::new();
    victor::write(&mut root, vector.clone(), "hello").await;
    let result = victor::find_nearest_neighbor(&mut root, vector)
        .await
        .unwrap()
        .content;
    assert_eq!(result, "hello".to_string());
}
