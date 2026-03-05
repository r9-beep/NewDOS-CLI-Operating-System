//! Simple in-memory virtual filesystem.

use alloc::{string::String, vec::Vec};

#[derive(Debug, Clone)]
pub enum Entry {
    File { name: String, data: Vec<u8> },
    Dir  { name: String, children: Vec<Entry> },
}

impl Entry {
    pub fn name(&self) -> &str {
        match self {
            Entry::File { name, .. } | Entry::Dir { name, .. } => name,
        }
    }
    pub fn is_dir(&self) -> bool { matches!(self, Entry::Dir { .. }) }
}

pub struct FileSystem {
    root: Entry,
}

impl FileSystem {
    pub fn new() -> Self {
        FileSystem {
            root: Entry::Dir { name: String::from("/"), children: Vec::new() },
        }
    }

    fn resolve<'a>(root: &'a mut Entry, parts: &[&str]) -> Option<&'a mut Entry> {
        if parts.is_empty() { return Some(root); }
        if let Entry::Dir { children, .. } = root {
            if let Some(child) = children.iter_mut().find(|e| e.name() == parts[0]) {
                return Self::resolve(child, &parts[1..]);
            }
        }
        None
    }

    fn split_path(path: &str) -> Vec<&str> {
        path.split('/').filter(|s| !s.is_empty()).collect()
    }

    pub fn mkdir(&mut self, path: &str) -> Result<(), &'static str> {
        let parts = Self::split_path(path);
        if parts.is_empty() { return Err("empty path"); }
        let (dir_parts, name) = parts.split_at(parts.len() - 1);
        let parent = Self::resolve(&mut self.root, dir_parts).ok_or("parent not found")?;
        if let Entry::Dir { children, .. } = parent {
            if children.iter().any(|e| e.name() == name[0]) {
                return Err("already exists");
            }
            children.push(Entry::Dir { name: String::from(name[0]), children: Vec::new() });
            Ok(())
        } else {
            Err("parent is not a directory")
        }
    }

    pub fn touch(&mut self, path: &str) -> Result<(), &'static str> {
        let parts = Self::split_path(path);
        if parts.is_empty() { return Err("empty path"); }
        let (dir_parts, name) = parts.split_at(parts.len() - 1);
        let parent = Self::resolve(&mut self.root, dir_parts).ok_or("parent not found")?;
        if let Entry::Dir { children, .. } = parent {
            if children.iter().any(|e| e.name() == name[0]) {
                return Err("already exists");
            }
            children.push(Entry::File { name: String::from(name[0]), data: Vec::new() });
            Ok(())
        } else {
            Err("parent is not a directory")
        }
    }

    pub fn write_file(&mut self, path: &str, content: &str) -> Result<(), &'static str> {
        let parts = Self::split_path(path);
        let entry = Self::resolve(&mut self.root, &parts).ok_or("not found")?;
        if let Entry::File { data, .. } = entry {
            *data = content.as_bytes().to_vec();
            Ok(())
        } else {
            Err("not a file")
        }
    }

    pub fn append_file(&mut self, path: &str, content: &str) -> Result<(), &'static str> {
        let parts = Self::split_path(path);
        let entry = Self::resolve(&mut self.root, &parts).ok_or("not found")?;
        if let Entry::File { data, .. } = entry {
            data.extend_from_slice(content.as_bytes());
            Ok(())
        } else {
            Err("not a file")
        }
    }

    pub fn read_file<'a>(&'a self, path: &str) -> Result<&'a [u8], &'static str> {
        fn find<'b>(e: &'b Entry, parts: &[&str]) -> Option<&'b Entry> {
            if parts.is_empty() { return Some(e); }
            if let Entry::Dir { children, .. } = e {
                if let Some(c) = children.iter().find(|c| c.name() == parts[0]) {
                    return find(c, &parts[1..]);
                }
            }
            None
        }
        let parts = Self::split_path(path);
        match find(&self.root, &parts) {
            Some(Entry::File { data, .. }) => Ok(data),
            Some(_) => Err("is a directory"),
            None    => Err("not found"),
        }
    }

    pub fn list<'a>(&'a self, path: &str) -> Result<Vec<&'a Entry>, &'static str> {
        fn find<'b>(e: &'b Entry, parts: &[&str]) -> Option<&'b Entry> {
            if parts.is_empty() { return Some(e); }
            if let Entry::Dir { children, .. } = e {
                if let Some(c) = children.iter().find(|c| c.name() == parts[0]) {
                    return find(c, &parts[1..]);
                }
            }
            None
        }
        let parts = Self::split_path(path);
        match if parts.is_empty() { Some(&self.root) } else { find(&self.root, &parts) } {
            Some(Entry::Dir { children, .. }) => Ok(children.iter().collect()),
            Some(_) => Err("not a directory"),
            None    => Err("not found"),
        }
    }

    pub fn delete(&mut self, path: &str) -> Result<(), &'static str> {
        let parts = Self::split_path(path);
        if parts.is_empty() { return Err("cannot delete root"); }
        let (dir_parts, name) = parts.split_at(parts.len() - 1);
        let parent = Self::resolve(&mut self.root, dir_parts).ok_or("parent not found")?;
        if let Entry::Dir { children, .. } = parent {
            let pos = children.iter().position(|e| e.name() == name[0]).ok_or("not found")?;
            children.remove(pos);
            Ok(())
        } else {
            Err("parent not a directory")
        }
    }

    pub fn exists(&self, path: &str) -> bool {
        fn find(e: &Entry, parts: &[&str]) -> bool {
            if parts.is_empty() { return true; }
            if let Entry::Dir { children, .. } = e {
                if let Some(c) = children.iter().find(|c| c.name() == parts[0]) {
                    return find(c, &parts[1..]);
                }
            }
            false
        }
        let parts = Self::split_path(path);
        find(&self.root, &parts)
    }

    /// Walk the entire tree and return `(full_path, is_dir, data)` for every
    /// node.  Directories are emitted before their children so a deserialiser
    /// can recreate the tree in a single forward pass.
    pub fn walk_all(&self) -> alloc::vec::Vec<(alloc::string::String, bool, alloc::vec::Vec<u8>)> {
        let mut out = alloc::vec::Vec::new();
        if let Entry::Dir { children, .. } = &self.root {
            for child in children { walk_entry(child, "", &mut out); }
        }
        out
    }
}

fn walk_entry(
    entry: &Entry,
    prefix: &str,
    out: &mut alloc::vec::Vec<(alloc::string::String, bool, alloc::vec::Vec<u8>)>,
) {
    use alloc::format;
    let full = format!("{}/{}", prefix, entry.name());
    match entry {
        Entry::Dir { children, .. } => {
            out.push((full.clone(), true, alloc::vec::Vec::new()));
            for child in children { walk_entry(child, &full, out); }
        }
        Entry::File { data, .. } => {
            out.push((full, false, data.clone()));
        }
    }
}
