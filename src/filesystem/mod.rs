pub mod memory;
pub mod web;

use std::fmt::Debug;

use async_trait::async_trait;

pub struct GetFileHandleOptions {
    pub create: bool,
}

pub struct CreateWritableOptions {
    pub keep_existing_data: bool,
}

#[async_trait(?Send)]
pub trait DirectoryHandle: Debug {
    type Error: Debug;
    type FileHandleT: FileHandle<Error = Self::Error>;

    async fn get_file_handle_with_options(
        &mut self,
        name: &str,
        options: &GetFileHandleOptions,
    ) -> Result<Self::FileHandleT, Self::Error>;
}

#[async_trait(?Send)]
pub trait FileHandle: Debug {
    type Error: Debug;
    type WritableFileStreamT: WritableFileStream<Error = Self::Error>;

    async fn create_writable_with_options(
        &mut self,
        options: &CreateWritableOptions,
    ) -> Result<Self::WritableFileStreamT, Self::Error>;

    async fn read(&self) -> Result<Vec<u8>, Self::Error>;

    async fn size(&self) -> Result<usize, Self::Error>;
}

#[async_trait(?Send)]
pub trait WritableFileStream: Debug {
    type Error: Debug;

    async fn write_at_cursor_pos(&mut self, data: Vec<u8>) -> Result<(), Self::Error>;

    async fn close(&mut self) -> Result<(), Self::Error>;

    async fn seek(&mut self, offset: usize) -> Result<(), Self::Error>;
}
