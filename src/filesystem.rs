use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    Blob, FileSystemCreateWritableOptions, FileSystemDirectoryHandle, FileSystemFileHandle,
    FileSystemGetFileOptions, FileSystemWritableFileStream,
};

#[derive(Debug)]
pub(crate) struct DirectoryHandle(FileSystemDirectoryHandle);

#[derive(Debug)]
pub(crate) struct FileHandle(FileSystemFileHandle);

#[derive(Debug)]
pub(crate) struct WritableFileStream(FileSystemWritableFileStream);

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
    pub(crate) async fn get_file_handle_with_options(
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
    pub(crate) async fn create_writable_with_options(
        &self,
        options: &FileSystemCreateWritableOptions,
    ) -> Result<WritableFileStream, JsValue> {
        let file_system_writable_file_stream = FileSystemWritableFileStream::unchecked_from_js(
            JsFuture::from(self.0.create_writable_with_options(options)).await?,
        );
        Ok(WritableFileStream(file_system_writable_file_stream))
    }

    pub(crate) async fn get_file(&self) -> Result<Blob, JsValue> {
        let file: Blob = JsFuture::from(self.0.get_file()).await.unwrap().into();
        Ok(file)
    }

    pub(crate) async fn get_size(&self) -> Result<f64, JsValue> {
        let file: Blob = JsFuture::from(self.0.get_file()).await.unwrap().into();
        Ok(file.size())
    }
}

impl WritableFileStream {
    pub(crate) async fn write_with_u8_array(&self, data: &mut [u8]) -> Result<(), JsValue> {
        JsFuture::from(self.0.write_with_u8_array(data)?).await?;
        Ok(())
    }

    pub(crate) async fn close(&self) -> Result<(), JsValue> {
        JsFuture::from(self.0.close()).await?;
        Ok(())
    }

    pub(crate) async fn seek_with_f64(&self, offset: f64) -> Result<(), JsValue> {
        JsFuture::from(self.0.seek_with_f64(offset).unwrap()).await?;
        Ok(())
    }
}
