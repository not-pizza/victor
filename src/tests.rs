use crate::{filesystem::memory, db::Victor};

#[tokio::test]
async fn store_and_retrieve() {
    let embedding = vec![1.0, 2.0, 3.0];

    let mut victor = Victor::new(memory::DirectoryHandle::new());

    victor.write(embedding.clone(), "hello").await;

    let result = victor
        .find_nearest_neighbor(embedding)
        .await
        .unwrap()
        .content;

    assert_eq!(result, "hello".to_string());
}
