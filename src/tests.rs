use crate::memory::{Db, DirectoryHandle};

#[tokio::test]
async fn store_and_retrieve() {
    let embedding = vec![1.0, 2.0, 3.0];

    let mut victor = Db::new(DirectoryHandle::default());

    victor
        .add_single_embedding("hello", embedding.clone(), Vec::<String>::new())
        .await;

    let result = victor
        .search_embedding(embedding, Vec::<String>::new(), 1)
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

    let mut victor = Db::new(DirectoryHandle::default());

    victor
        .add_single_embedding("hello", embedding_1.clone(), Vec::<String>::new())
        .await;
    victor
        .add_single_embedding("goodbye", embedding_2.clone(), Vec::<String>::new())
        .await;

    {
        let result = victor
            .search_embedding(embedding_1, Vec::<String>::new(), 1)
            .await
            .first()
            .unwrap()
            .content
            .clone();

        assert_eq!(result, "hello".to_string());
    }
    {
        let result = victor
            .search_embedding(embedding_2, Vec::<String>::new(), 1)
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

    let mut victor = Db::new(DirectoryHandle::default());

    victor
        .add_single_embedding("hello", embedding_1.clone(), vec!["greetings".to_string()])
        .await;
    victor
        .add_single_embedding("goodbye", embedding_2.clone(), vec!["goodbyes".to_string()])
        .await;

    {
        let result = victor
            .search_embedding(embedding_1.clone(), Vec::<String>::new(), 1)
            .await
            .first()
            .unwrap()
            .content
            .clone();

        assert_eq!(result, "hello".to_string());
    }
    {
        let result = victor
            .search_embedding(embedding_2.clone(), Vec::<String>::new(), 1)
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

    let mut victor = Db::new(DirectoryHandle::default());

    victor
        .add_single_embedding("hello", embedding_1, Vec::<String>::new())
        .await;
    victor
        .add_single_embedding("hello", embedding_2, Vec::<String>::new())
        .await;
}

#[tokio::test]
async fn add() {
    let mut victor = Db::new(DirectoryHandle::default());

    victor
        .add(vec!["pineapple", "rocks"], Vec::<String>::new())
        .await;

    let result = victor
        .search("hawaiian pizza", Vec::<String>::new(), 1)
        .await
        .first()
        .unwrap()
        .content
        .clone();
    assert_eq!(result, "pineapple");
}
