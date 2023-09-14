use std::{io::SeekFrom, path::PathBuf};

use async_trait::async_trait;
use tokio::io::{AsyncSeekExt, AsyncWriteExt};

use crate::filesystem;

#[derive(Debug)]
pub struct DirectoryHandle(PathBuf);

#[derive(Debug)]
pub struct FileHandle(PathBuf);

#[derive(Debug)]
pub struct WritableFileStream(tokio::fs::File);

impl From<PathBuf> for DirectoryHandle {
    fn from(handle: PathBuf) -> Self {
        Self(handle)
    }
}

impl From<PathBuf> for FileHandle {
    fn from(handle: PathBuf) -> Self {
        Self(handle)
    }
}

impl From<tokio::fs::File> for WritableFileStream {
    fn from(handle: tokio::fs::File) -> Self {
        Self(handle)
    }
}

#[async_trait(?Send)]
impl filesystem::DirectoryHandle for DirectoryHandle {
    type Error = std::io::Error;
    type FileHandleT = FileHandle;

    async fn get_file_handle_with_options(
        &self,
        name: &str,
        options: &filesystem::GetFileHandleOptions,
    ) -> Result<Self::FileHandleT, Self::Error> {
        let mut path = self.0.clone();
        path.push(name);

        // Make sure the file exists
        let _ = tokio::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(options.create)
            .open(&path)
            .await?;

        Ok(FileHandle(path))
    }

    async fn remove_entry(&mut self, name: &str) -> Result<(), Self::Error> {
        let mut path = self.0.clone();
        path.push(name);

        let metadata = tokio::fs::metadata(&path).await?;
        if metadata.is_file() {
            tokio::fs::remove_file(&path).await?;
        } else if metadata.is_dir() {
            tokio::fs::remove_dir(&path).await?;
        }

        Ok(())
    }
}

#[async_trait(?Send)]
impl filesystem::FileHandle for FileHandle {
    type Error = std::io::Error;
    type WritableFileStreamT = WritableFileStream;

    async fn create_writable_with_options(
        &mut self,
        options: &filesystem::CreateWritableOptions,
    ) -> Result<Self::WritableFileStreamT, Self::Error> {
        let file = tokio::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(!options.keep_existing_data)
            .open(&self.0)
            .await?;

        Ok(WritableFileStream(file))
    }

    async fn read(&self) -> Result<Vec<u8>, Self::Error> {
        use tokio::io::AsyncReadExt;

        let mut file = tokio::fs::File::open(&self.0).await?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).await?;
        Ok(buffer)
    }

    async fn size(&self) -> Result<usize, Self::Error> {
        let metadata = tokio::fs::metadata(&self.0).await?;
        Ok(metadata.len() as usize)
    }
}

#[async_trait(?Send)]
impl filesystem::WritableFileStream for WritableFileStream {
    type Error = std::io::Error;

    async fn write_at_cursor_pos(&mut self, data: Vec<u8>) -> Result<(), Self::Error> {
        self.0.write_all(&data).await?;
        Ok(())
    }

    async fn close(&mut self) -> Result<(), Self::Error> {
        self.0.shutdown().await?;
        Ok(())
    }

    async fn seek(&mut self, offset: usize) -> Result<(), Self::Error> {
        self.0.seek(SeekFrom::Start(offset as u64)).await?;
        Ok(())
    }
}
