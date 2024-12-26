#[tokio::main(flavor = "current_thread")]
async fn main() {
    use std::path::PathBuf;

    use victor_db::native::Db;

    let _ = std::fs::create_dir("./victor_test_data");
    let mut victor = Db::new(PathBuf::from("./victor_test_data"));

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
