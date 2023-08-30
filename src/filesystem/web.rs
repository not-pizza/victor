use async_trait::async_trait;
use js_sys::{ArrayBuffer, Uint8Array};
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    FileSystemCreateWritableOptions, FileSystemDirectoryHandle, FileSystemFileHandle,
    FileSystemGetFileOptions, FileSystemWritableFileStream,
};

use crate::filesystem;

#[derive(Debug)]
pub(crate) struct DirectoryHandle(FileSystemDirectoryHandle);

#[derive(Debug)]
pub(crate) struct FileHandle(FileSystemFileHandle);

#[derive(Debug)]
pub(crate) struct WritableFileStream(FileSystemWritableFileStream);
#[derive(Debug)]
pub(crate) struct Blob(web_sys::Blob);

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

#[async_trait(?Send)]
impl filesystem::DirectoryHandle for DirectoryHandle {
    type Error = JsValue;
    type FileHandleT = FileHandle;

    async fn get_file_handle_with_options(
        &mut self,
        name: &str,
        options: &filesystem::GetFileHandleOptions,
    ) -> Result<Self::FileHandleT, Self::Error> {
        let file_system_file_handle = FileSystemFileHandle::from(
            JsFuture::from(self.0.get_file_handle_with_options(
                name,
                FileSystemGetFileOptions::new().create(options.create),
            ))
            .await?,
        );
        Ok(FileHandle(file_system_file_handle))
    }

    async fn remove_entry(&mut self, name: &str) -> Result<(), Self::Error> {
        JsFuture::from(self.0.remove_entry(name)).await?;
        Ok(())
    }
}

#[async_trait(?Send)]
impl filesystem::FileHandle for FileHandle {
    type Error = JsValue;
    type WritableFileStreamT = WritableFileStream;

    async fn create_writable_with_options(
        &mut self,
        options: &filesystem::CreateWritableOptions,
    ) -> Result<Self::WritableFileStreamT, Self::Error> {
        let file_system_writable_file_stream = FileSystemWritableFileStream::unchecked_from_js(
            JsFuture::from(
                self.0.create_writable_with_options(
                    FileSystemCreateWritableOptions::new()
                        .keep_existing_data(options.keep_existing_data),
                ),
            )
            .await?,
        );
        Ok(WritableFileStream(file_system_writable_file_stream))
    }

    async fn read(&self) -> Result<Vec<u8>, Self::Error> {
        self.get_file().await?.read().await
    }

    async fn size(&self) -> Result<usize, JsValue> {
        let size = self.get_file().await?.size();
        Ok(size)
    }
}

impl FileHandle {
    pub(crate) async fn get_file(&self) -> Result<Blob, JsValue> {
        let file: web_sys::Blob = JsFuture::from(self.0.get_file()).await?.into();
        Ok(Blob(file))
    }
}

#[async_trait(?Send)]
impl filesystem::WritableFileStream for WritableFileStream {
    type Error = JsValue;

    async fn write_at_cursor_pos(&mut self, mut data: Vec<u8>) -> Result<(), Self::Error> {
        JsFuture::from(self.0.write_with_u8_array(data.as_mut_slice())?).await?;
        Ok(())
    }

    async fn close(&mut self) -> Result<(), Self::Error> {
        JsFuture::from(self.0.close()).await?;
        Ok(())
    }

    async fn seek(&mut self, offset: usize) -> Result<(), Self::Error> {
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

    #[allow(dead_code)]
    pub(crate) async fn text(&self) -> Result<String, JsValue> {
        JsFuture::from(self.0.text())
            .await?
            .as_string()
            .ok_or(JsValue::NULL)
    }
}
