use std::path::PathBuf;

use victor::native::Db;

#[tokio::main(flavor = "current_thread")]
async fn main() {
    let _ = std::fs::create_dir("./victor_test_data");
    let mut victor = Db::new(PathBuf::from("./victor_test_data"));

    victor.clear_db().await.unwrap();

    victor
        .write(
            vec![1.0, 0.0, 0.0],
            "Test Vector 1",
            vec!["Test".to_string()],
        )
        .await;
    victor
        .write(
            vec![0.0, 1.0, 0.0],
            "Test Vector 2",
            vec!["Test".to_string()],
        )
        .await;

    let nearest = victor
        .find_nearest_neighbor(vec![0.9, 0.0, 0.0], vec![])
        .await
        .unwrap()
        .content;
    assert_eq!(nearest, "Test Vector 1".to_string());
}
