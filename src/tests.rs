use crate::{db::Victor, filesystem::memory};

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

#[tokio::test]
async fn store_two_and_retrieve() {
    let embedding_1 = vec![1.0, 2.0, 3.0];
    let embedding_2 = vec![-1.0, -2.0, -3.0];

    let mut victor = Victor::new(memory::DirectoryHandle::new());

    victor.write(embedding_1.clone(), "hello").await;
    victor.write(embedding_2.clone(), "goodbye").await;

    {
        let result = victor
            .find_nearest_neighbor(embedding_1)
            .await
            .unwrap()
            .content;

        assert_eq!(result, "hello".to_string());
    }
    {
        let result = victor
            .find_nearest_neighbor(embedding_2)
            .await
            .unwrap()
            .content;

        assert_eq!(result, "goodbye".to_string());
    }
}
