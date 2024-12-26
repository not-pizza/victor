//! "in-memory" filesystem for use in tests or when persistence isn't necessary

use std::{cell::RefCell, collections::HashMap, rc::Rc};

use async_trait::async_trait;

use crate::filesystem;

/// An entry in a virtual directory in the in-memory filesystem.
#[derive(Debug, Clone)]
pub enum DirectoryEntry {
    #[allow(dead_code)]
    Directory(DirectoryHandle),
    File(FileHandle),
}

/// A virtual directory in the in-memory filesystem.
#[derive(Debug, Clone)]
pub struct DirectoryHandle(Rc<RefCell<HashMap<String, DirectoryEntry>>>);

/// A virtual file in the in-memory filesystem.
#[derive(Debug, Clone)]
pub struct FileHandle(WritableFileStream);

/// A writable file stream in the in-memory filesystem.
#[derive(Debug, Clone)]
pub struct WritableFileStream {
    cursor_pos: usize,
    stream: Rc<RefCell<Vec<u8>>>,
}

#[async_trait(?Send)]
impl filesystem::DirectoryHandle for DirectoryHandle {
    type Error = String;
    type FileHandleT = FileHandle;

    async fn get_file_handle_with_options(
        &self,
        name: &str,
        options: &filesystem::GetFileHandleOptions,
    ) -> Result<Self::FileHandleT, Self::Error> {
        let mut directory = self.0.borrow_mut();
        let entry = match directory.entry(name.to_string()) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.get().clone(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                if options.create {
                    let file_handle = FileHandle::new();
                    entry.insert(DirectoryEntry::File(file_handle.clone()));
                    DirectoryEntry::File(file_handle)
                } else {
                    return Err(format!("'{name}' does not exist"));
                }
            }
        };

        match entry {
            DirectoryEntry::Directory(_) => Err(format!("'{name}' is a directory")),
            DirectoryEntry::File(file) => Ok(file),
        }
    }

    async fn remove_entry(&mut self, name: &str) -> Result<(), Self::Error> {
        let mut directory = self.0.borrow_mut();
        directory.remove(name);
        Ok(())
    }
}

impl DirectoryHandle {
    #[allow(dead_code)]
    pub(crate) fn new() -> Self {
        Self(Rc::new(RefCell::new(HashMap::new())))
    }
}

#[async_trait(?Send)]
impl filesystem::FileHandle for FileHandle {
    type Error = String;
    type WritableFileStreamT = WritableFileStream;

    async fn create_writable_with_options(
        &mut self,
        options: &filesystem::CreateWritableOptions,
    ) -> Result<Self::WritableFileStreamT, Self::Error> {
        if !options.keep_existing_data {
            self.0.stream.borrow_mut().clear();
        }
        Ok(WritableFileStream {
            cursor_pos: 0,
            ..self.0.clone()
        })
    }

    async fn read(&self) -> Result<Vec<u8>, Self::Error> {
        let stream = self.0.stream.clone();
        let data = stream.borrow().clone();
        Ok(data)
    }

    async fn size(&self) -> Result<usize, Self::Error> {
        Ok(self.0.len())
    }
}

#[async_trait(?Send)]
impl filesystem::WritableFileStream for WritableFileStream {
    type Error = String;

    async fn write_at_cursor_pos(&mut self, data: Vec<u8>) -> Result<(), Self::Error> {
        let data_len = data.len();

        let mut stream = self.stream.borrow_mut();
        *stream = stream[0..self.cursor_pos]
            .iter()
            .cloned()
            .chain(data)
            .collect::<Vec<u8>>();

        self.cursor_pos += data_len;

        Ok(())
    }

    async fn close(&mut self) -> Result<(), Self::Error> {
        // no op
        Ok(())
    }

    async fn seek(&mut self, offset: usize) -> Result<(), Self::Error> {
        if offset > self.len() {
            return Err(format!(
                "cannot seek to {offset} because the file is only {len} bytes long",
                len = self.len()
            ));
        }
        self.cursor_pos = offset;
        Ok(())
    }
}

impl FileHandle {
    fn new() -> Self {
        Self(WritableFileStream::new())
    }
}

impl WritableFileStream {
    fn new() -> Self {
        Self {
            cursor_pos: 0,
            stream: Rc::new(RefCell::new(Vec::new())),
        }
    }

    fn len(&self) -> usize {
        self.stream.borrow().len()
    }
}
