use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    FileSystemDirectoryHandle, FileSystemFileHandle, FileSystemGetFileOptions,
    FileSystemWritableFileStream,
};

#[derive(Debug)]
pub struct DirectoryHandle(FileSystemDirectoryHandle);

#[derive(Debug)]
pub struct FileHandle(FileSystemFileHandle);

#[derive(Debug)]
pub struct WritableFileStream(FileSystemWritableFileStream);

impl From<FileSystemDirectoryHandle> for DirectoryHandle {
    fn from(handle: FileSystemDirectoryHandle) -> Self {
        Self(handle)
    }
}

impl From<FileSystemFileHandle> for FileHandle {
    fn from(handle: FileSystemFileHandle) -> Self {
        Self(handle)
    }
}

impl From<FileSystemWritableFileStream> for WritableFileStream {
    fn from(handle: FileSystemWritableFileStream) -> Self {
        Self(handle)
    }
}

impl DirectoryHandle {
    pub async fn get_file_handle_with_options(
        &self,
        name: &str,
        options: &FileSystemGetFileOptions,
    ) -> Result<FileHandle, JsValue> {
        let file_system_file_handle = FileSystemFileHandle::from(
            JsFuture::from(self.0.get_file_handle_with_options(name, options)).await?,
        );
        Ok(FileHandle(file_system_file_handle))
    }
}

impl FileHandle {
    pub async fn create_writable(&self) -> Result<WritableFileStream, JsValue> {
        let file_system_writable_file_stream = FileSystemWritableFileStream::unchecked_from_js(
            JsFuture::from(self.0.create_writable()).await?,
        );
        Ok(WritableFileStream(file_system_writable_file_stream))
    }
}

impl WritableFileStream {
    pub async fn write_with_u8_array(&self, data: &mut [u8]) -> Result<(), JsValue> {
        JsFuture::from(self.0.write_with_u8_array(data)?).await?;
        Ok(())
    }

    pub async fn close(&self) -> Result<(), JsValue> {
        JsFuture::from(self.0.close()).await?;
        Ok(())
    }
}
