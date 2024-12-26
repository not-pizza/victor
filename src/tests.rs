use crate::memory::{Db, DirectoryHandle};

#[tokio::test]
async fn store_and_retrieve() {
    let embedding = vec![1.0, 2.0, 3.0];

    let mut victor = Db::new(DirectoryHandle::new());

    victor.write("hello", embedding.clone(), vec![]).await;

    let result = victor
        .search_embedding(embedding, vec![], 1)
        .await
        .first()
        .unwrap()
        .content
        .clone();

    assert_eq!(result, "hello".to_string());
}

#[tokio::test]
async fn store_two_and_retrieve() {
    let embedding_1 = vec![1.0, 2.0, 3.0];
    let embedding_2 = vec![-1.0, -2.0, -3.0];

    let mut victor = Db::new(DirectoryHandle::new());

    victor.write("hello", embedding_1.clone(), vec![]).await;
    victor.write("goodbye", embedding_2.clone(), vec![]).await;

    {
        let result = victor
            .search_embedding(embedding_1, vec![], 1)
            .await
            .first()
            .unwrap()
            .content
            .clone();

        assert_eq!(result, "hello".to_string());
    }
    {
        let result = victor
            .search_embedding(embedding_2, vec![], 1)
            .await
            .first()
            .unwrap()
            .content
            .clone();

        assert_eq!(result, "goodbye".to_string());
    }
}

#[tokio::test]
async fn store_two_and_retrieve_with_tags() {
    let embedding_1 = vec![1.0, 2.0, 3.0];
    let embedding_2 = vec![-1.0, -2.0, -3.0];

    let mut victor = Db::new(DirectoryHandle::new());

    victor
        .write("hello", embedding_1.clone(), vec!["greetings".to_string()])
        .await;
    victor
        .write("goodbye", embedding_2.clone(), vec!["goodbyes".to_string()])
        .await;

    {
        let result = victor
            .search_embedding(embedding_1.clone(), vec![], 1)
            .await
            .first()
            .unwrap()
            .content
            .clone();

        assert_eq!(result, "hello".to_string());
    }
    {
        let result = victor
            .search_embedding(embedding_2.clone(), vec![], 1)
            .await
            .first()
            .unwrap()
            .content
            .clone();

        assert_eq!(result, "goodbye".to_string());
    }

    {
        let result = victor
            .search_embedding(embedding_1.clone(), vec!["goodbyes".to_string()], 1)
            .await
            .first()
            .unwrap()
            .content
            .clone();

        assert_eq!(result, "goodbye".to_string());
    }
    {
        let result = victor
            .search_embedding(embedding_2, vec!["greetings".to_string()], 1)
            .await
            .first()
            .unwrap()
            .clone();

        assert_eq!(result.content, "hello");
    }
    {
        let result = victor
            .search_embedding(embedding_1, vec!["mysterious".to_string()], 1)
            .await;

        assert_eq!(result.first(), None);
    }
}

#[should_panic]
#[tokio::test]
async fn incompatible_size_panic() {
    let embedding_1 = vec![1.0, 2.0, 3.0];
    let embedding_2 = vec![1.0, 2.0, 3.0, 4.0];

    let mut victor = Db::new(DirectoryHandle::new());

    victor.write("hello", embedding_1, vec![]).await;
    victor.write("hello", embedding_2, vec![]).await;
}

#[tokio::test]
async fn add_many() {
    let mut victor = Db::new(DirectoryHandle::new());

    victor.add_many(vec!["pinapple", "rocks"], vec![]).await;

    let result = victor
        .search("hawaiian pizza", vec![], 1)
        .await
        .first()
        .unwrap()
        .content
        .clone();
    assert_eq!(result, "pinapple");
}
