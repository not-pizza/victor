mod utils;

use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{FileSystemFileHandle, FileSystemWritableFileStream};

#[allow(unused_macros)]
macro_rules! console_log {
    ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
}
#[allow(unused_macros)]
macro_rules! console_warn {
    ($($t:tt)*) => (warn(&format_args!($($t)*).to_string()))
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
    #[wasm_bindgen(js_namespace = console)]
    fn warn(s: &str);
}

#[wasm_bindgen]
pub async fn init_db_contents(file_handle: FileSystemFileHandle) {
    utils::set_panic_hook();

    let writable = FileSystemWritableFileStream::unchecked_from_js(
        JsFuture::from(file_handle.create_writable()).await.unwrap(),
    );
    JsFuture::from(writable.write_with_str("init db contents").unwrap())
        .await
        .unwrap();
    JsFuture::from(writable.close()).await.unwrap();
}
