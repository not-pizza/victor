#[tokio::main(flavor = "current_thread")]
async fn main() {
    use victor_db::memory::{Db, DirectoryHandle};

    let mut victor = Db::new(DirectoryHandle::default());

    victor.clear_db().await.unwrap();

    victor
        .add(
            vec!["Pineapple", "Rocks"], // documents
            vec!["Pizza Toppings"],     // tags (only used for filtering)
        )
        .await;

    victor
        .add_single("Cheese pizza", Vec::<String>::new())
        .await; // Add another entry with no tags

    // read the 10 closest results from victor that are tagged with "Pizza Toppings"
    // (only 2 will be returned because we only inserted two embeddings with the "Pizza Toppings" tag)
    let nearest = victor
        .search("Hawaiian pizza", vec!["Pizza Toppings"], 10)
        .await
        .first()
        .unwrap()
        .content
        .clone();
    assert_eq!(nearest, "Pineapple".to_string());
}
