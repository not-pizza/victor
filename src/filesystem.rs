use js_sys::{ArrayBuffer, Uint8Array};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    FileSystemCreateWritableOptions, FileSystemDirectoryHandle, FileSystemFileHandle,
    FileSystemGetFileOptions, FileSystemWritableFileStream,
};

#[derive(Debug)]
pub(crate) struct DirectoryHandle(FileSystemDirectoryHandle);

#[derive(Debug)]
pub(crate) struct FileHandle(FileSystemFileHandle);

#[derive(Debug)]
pub(crate) struct WritableFileStream(FileSystemWritableFileStream);
#[derive(Debug)]
struct Blob(web_sys::Blob);

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

impl From<web_sys::Blob> for Blob {
    fn from(handle: web_sys::Blob) -> Self {
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

    pub(crate) async fn read(&self) -> Result<Vec<u8>, JsValue> {
        self.get_file().await?.read().await
    }

    async fn get_file(&self) -> Result<Blob, JsValue> {
        let file: web_sys::Blob = JsFuture::from(self.0.get_file()).await?.into();
        Ok(Blob(file))
    }

    pub(crate) async fn get_size(&self) -> Result<usize, JsValue> {
        let size = self.get_file().await?.size();
        Ok(size)
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

    pub(crate) async fn seek(&self, offset: usize) -> Result<(), JsValue> {
        JsFuture::from(self.0.seek_with_u32(offset as u32)?).await?;
        Ok(())
    }
}

impl Blob {
    fn size(&self) -> usize {
        self.0.size() as usize
    }

    async fn read(&self) -> Result<Vec<u8>, JsValue> {
        let buffer = ArrayBuffer::unchecked_from_js(JsFuture::from(self.0.array_buffer()).await?);
        let uint8_array = Uint8Array::new(&buffer);
        let mut vec = vec![0; self.size()];
        uint8_array.copy_to(&mut vec);
        Ok(vec)
    }
}
