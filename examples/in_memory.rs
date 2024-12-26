#[tokio::main(flavor = "current_thread")]
async fn main() {
    use victor_db::memory::{Db, DirectoryHandle};

    let mut victor = Db::new(DirectoryHandle::new());

    victor.clear_db().await.unwrap();

    victor
        .add_many(
            vec!["Pineapple", "Rocks"], // documents
            vec!["PizzaToppings"],      // tags (only used for filtering)
        )
        .await;

    victor.add("Cheese pizza", vec![]).await; // Add another entry with no tags

    // read the 10 closest results from victor that are tagged with "tags"
    // (only 2 will be returned because we only inserted two embeddings)
    let nearest = victor
        .search("Hawaiian pizza", vec!["PizzaToppings"], 10)
        .await
        .first()
        .unwrap()
        .content
        .clone();
    assert_eq!(nearest, "Pineapple".to_string());
}
