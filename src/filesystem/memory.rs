//! "in-memory" filesystem for use in tests or when persistence isn't necessary

use std::{
    cell::{RefCell, RefMut},
    collections::HashMap,
    sync::{RwLock, RwLockWriteGuard},
};

use async_trait::async_trait;

use crate::filesystem;

#[derive(Debug)]
pub(crate) enum DirectoryEntry {
    Directory(RwLock<DirectoryHandle>),
    File(RwLock<FileHandle>),
}

#[derive(Debug)]
pub(crate) struct DirectoryHandle(HashMap<String, RefCell<DirectoryEntry>>);

#[derive(Debug)]
pub(crate) struct FileHandle(RefCell<Vec<u8>>);

#[derive(Debug)]
pub(crate) struct WritableFileStream<'a>(RefMut<'a, Vec<u8>>);

#[async_trait(?Send)]
impl filesystem::DirectoryHandle for DirectoryHandle {
    type Error = String;
    type FileHandleT = FileHandle;

    async fn get_file_handle_with_options(
        &self,
        name: &str,
        options: &filesystem::GetFileHandleOptions,
    ) -> Result<Self::FileHandleT, Self::Error> {
        let directory = self.0;
        let entry = directory.get(name);
        

        let file_system_file_handle = FileSystemFileHandle::from(
            JsFuture::from(self.0.get_file_handle_with_options(
                name,
                FileSystemGetFileOptions::new().create(options.create),
            ))
            .await?,
        );
        Ok(FileHandle(file_system_file_handle))
    }
}
