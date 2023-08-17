mod utils;

use wasm_bindgen::prelude::*;
use web_sys::FileSystemDirectoryHandle;

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
pub fn greet(handle: FileSystemDirectoryHandle) {
    utils::set_panic_hook();
    console_log!("handle: {:?}", handle);
}
