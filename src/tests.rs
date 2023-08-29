use crate::{db::Victor, filesystem::memory};

#[tokio::test]
async fn store_and_retrieve() {
    let embedding = vec![1.0, 2.0, 3.0];

    let mut victor = Victor::new(memory::DirectoryHandle::new());

    victor.write(embedding.clone(), "hello", vec![]).await;

    let result = victor
        .find_nearest_neighbor(embedding, vec![])
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

    victor.write(embedding_1.clone(), "hello", vec![]).await;
    victor.write(embedding_2.clone(), "goodbye", vec![]).await;

    {
        let result = victor
            .find_nearest_neighbor(embedding_1, vec![])
            .await
            .unwrap()
            .content;

        assert_eq!(result, "hello".to_string());
    }
    {
        let result = victor
            .find_nearest_neighbor(embedding_2, vec![])
            .await
            .unwrap()
            .content;

        assert_eq!(result, "goodbye".to_string());
    }
}

#[tokio::test]
async fn store_two_and_retrieve_with_tags() {
    let embedding_1 = vec![1.0, 2.0, 3.0];
    let embedding_2 = vec![-1.0, -2.0, -3.0];

    let mut victor = Victor::new(memory::DirectoryHandle::new());

    victor
        .write(embedding_1.clone(), "hello", vec!["greetings".to_string()])
        .await;
    victor
        .write(embedding_2.clone(), "goodbye", vec!["goodbyes".to_string()])
        .await;

    {
        let result = victor
            .find_nearest_neighbor(embedding_1.clone(), vec![])
            .await
            .unwrap()
            .content;

        assert_eq!(result, "hello".to_string());
    }
    {
        let result = victor
            .find_nearest_neighbor(embedding_2.clone(), vec![])
            .await
            .unwrap()
            .content;

        assert_eq!(result, "goodbye".to_string());
    }

    {
        let result = victor
            .find_nearest_neighbor(embedding_1.clone(), vec!["goodbyes".to_string()])
            .await
            .unwrap()
            .content;

        assert_eq!(result, "goodbye".to_string());
    }
    {
        let result = victor
            .find_nearest_neighbor(embedding_2, vec!["greetings".to_string()])
            .await
            .unwrap()
            .content;

        assert_eq!(result, "hello".to_string());
    }
    {
        let result = victor
            .find_nearest_neighbor(embedding_1, vec!["mysterious".to_string()])
            .await;

        assert_eq!(result, None);
    }
}
