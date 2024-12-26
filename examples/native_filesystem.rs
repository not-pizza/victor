use std::path::PathBuf;

use victor_db::native::Db;

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let _ = std::fs::create_dir("./victor_test_data");
    let mut victor = Db::new(PathBuf::from("./victor_test_data"));

    victor.clear_db().await.unwrap();

    victor
        .add_embedding(
            "Test Vector 1",
            vec![1.0, 0.0, 0.0],
            vec!["Test".to_string()],
        )
        .await;
    victor
        .add_embedding(
            "Test Vector 2",
            vec![0.0, 1.0, 0.0],
            vec!["Test".to_string()],
        )
        .await;

    // read the 10 closest results from victor that are tagged with "tags"
    // (only 2 will be returned because we only inserted two embeddings)
    let nearest = victor
        .search_embedding(vec![0.9, 0.0, 0.0], vec!["Test".to_string()], 10)
        .await
        .first()
        .unwrap()
        .content
        .clone();
    assert_eq!(nearest, "Test Vector 1".to_string());
}
