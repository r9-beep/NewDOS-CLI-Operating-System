//! NewDOS interactive shell — renders into a TextBuffer; GUI windows display it.

use crate::{keyboard::SpecialKey, storage, time, vfs};
use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};

// ── TextBuffer ────────────────────────────────────────────────────────────────

/// Scrollback buffer that windows render line-by-line.
pub struct TextBuffer {
    pub lines:    Vec<String>,
    pub input:    String,   // current input line (not yet submitted)
    max_lines:    usize,
    current_line: String,   // line being assembled from push_str calls
}

impl TextBuffer {
    pub fn new(max_lines: usize) -> Self {
        TextBuffer {
            lines: Vec::new(),
            input: String::new(),
            max_lines,
            current_line: String::new(),
        }
    }

    /// Write a string into the buffer, splitting on newlines.
    pub fn push_str(&mut self, s: &str) {
        for ch in s.chars() {
            if ch == '\n' {
                let line = core::mem::take(&mut self.current_line);
                self.lines.push(line);
                if self.lines.len() > self.max_lines {
                    self.lines.remove(0);
                }
            } else {
                self.current_line.push(ch);
            }
        }
    }

    /// Flush any partial line.
    pub fn flush(&mut self) {
        if !self.current_line.is_empty() {
            let line = core::mem::take(&mut self.current_line);
            self.lines.push(line);
            if self.lines.len() > self.max_lines {
                self.lines.remove(0);
            }
        }
    }

    /// Last N lines visible in the terminal area.
    pub fn tail(&self, n: usize) -> &[String] {
        let len = self.lines.len();
        if len > n { &self.lines[len - n..] } else { &self.lines }
    }
}

// ── ShellState ────────────────────────────────────────────────────────────────

pub struct ShellState {
    pub output:      TextBuffer,
    pub fs:          vfs::FileSystem,
    pub cwd:         String,
    pub username:    String,
    pub device:      String,
    /// `None` = shell mode, `Some(path)` = editor mode
    pub editor_file: Option<String>,
    pub editor_buf:  String,
}

impl ShellState {
    pub fn new() -> Self {
        ShellState {
            output:      TextBuffer::new(200),
            fs:          vfs::FileSystem::new(),
            cwd:         String::from("/"),
            username:    String::from("pierre"),
            device:      String::from("NewDOS-PC"),
            editor_file: None,
            editor_buf:  String::new(),
        }
    }

    /// Like `new()` but tries to restore the VFS from the on-disk data area.
    pub fn new_from_disk() -> Self {
        let mut s = Self::new();
        if let Some(fs) = crate::disk_store::load() { s.fs = fs; }
        s
    }

    fn out(&mut self, s: &str) { self.output.push_str(s); }
    fn outln(&mut self, s: &str) { self.output.push_str(s); self.output.push_str("\n"); }

    /// Persist the in-memory VFS to disk.  Silently no-ops if no ATA drive present.
    fn disk_sync(&self) { crate::disk_store::save(&self.fs); }

    pub fn prompt_string(&self) -> String {
        format!("{}@{}:{}> ", self.username, self.device, self.cwd)
    }

    pub fn init(&mut self) {
        self.outln("╔══════════════════════════════════════════════════════╗");
        self.outln("║  NewDOS v0.1.1  —  bootloader 0.11  |  x86_64       ║");
        self.outln("║  VESA VBE / UEFI GOP  |  Desktop GUI                ║");
        self.outln("╚══════════════════════════════════════════════════════╝");
        self.outln("Type 'pierre help' for commands.");
        self.outln("");
    }

    // ── editor ────────────────────────────────────────────────────────────────

    fn editor_draw(&mut self) {
        self.outln("┌── editor ── F9=Save  F10=Exit ──────────────────────");
        let buf = self.editor_buf.clone();
        for line in buf.split('\n') {
            let s = format!("│ {}", line);
            self.outln(&s);
        }
        self.outln("└─────────────────────────────────────────────────────");
    }

    pub fn enter_editor(&mut self, path: &str) {
        let content = match self.fs.read_file(path) {
            Ok(b)  => core::str::from_utf8(b).unwrap_or("").to_string(),
            Err(_) => String::new(),
        };
        self.editor_file = Some(path.to_string());
        self.editor_buf  = content;
        self.editor_draw();
    }

    // ── handle one character of input ─────────────────────────────────────────

    pub fn process_char(&mut self, c: char) {
        if self.editor_file.is_some() {
            match c {
                '\x08' => { self.editor_buf.pop(); }
                _       => { self.editor_buf.push(c); }
            }
            return;
        }
        match c {
            '\n' | '\r' => {
                let cmd = core::mem::take(&mut self.output.input);
                let prompt = self.prompt_string();
                let echo = format!("{}{}", prompt, cmd);
                self.outln(&echo);
                self.dispatch(&cmd);
            }
            '\x08' => { self.output.input.pop(); }
            c if c.is_ascii() && !c.is_control() => { self.output.input.push(c); }
            _ => {}
        }
    }

    pub fn process_special(&mut self, k: SpecialKey) {
        if self.editor_file.is_some() {
            match k {
                SpecialKey::F9  => {
                    if let Some(ref path) = self.editor_file.clone() {
                        let buf = self.editor_buf.clone();
                        match self.fs.write_file(path, &buf) {
                            Ok(_)  => { self.outln("Saved."); self.disk_sync(); }
                            Err(e) => self.outln(e),
                        }
                    }
                }
                SpecialKey::F10 => {
                    self.editor_file = None;
                    self.editor_buf  = String::new();
                    self.outln("Editor closed.");
                }
                _ => {}
            }
        }
    }

    // ── command dispatch ──────────────────────────────────────────────────────

    fn dispatch(&mut self, raw: &str) {
        let raw = raw.trim();
        if raw.is_empty() { return; }

        let (role, rest) = if let Some(r) = raw.strip_prefix("pierre ") {
            ("pierre", r)
        } else if let Some(r) = raw.strip_prefix("suppiere ") {
            ("suppiere", r)
        } else if raw == "pierre" || raw == "suppiere" {
            (raw, "")
        } else {
            self.outln("Unknown command. Type 'pierre help'.");
            return;
        };

        let args: Vec<&str> = rest.splitn(3, ' ').collect();
        let cmd  = args.get(0).copied().unwrap_or("");
        let arg1 = args.get(1).copied().unwrap_or("");
        let arg2 = args.get(2).copied().unwrap_or("");

        match (role, cmd) {
            (_, "help") | (_, "") => {
                self.outln("NewDOS shell — commands:");
                self.outln("  pierre help / ls / dir / mkdir / touch / write / cat / del");
                self.outln("  pierre cls / mem / storage / gpt / exfat / version / banner");
                self.outln("  pierre time / tz <+N> / edit <file> / cd / user / device");
                self.outln("  pierre sync  (force filesystem flush to disk)");
                self.outln("  suppiere gfx / suppiere restart");
            }
            (_, "ls") | (_, "dir") => {
                let path = if arg1.is_empty() { self.cwd.clone() } else { arg1.to_string() };
                let listing: Vec<String> = match self.fs.list(&path) {
                    Ok(entries) => {
                        if entries.is_empty() {
                            alloc::vec!["  (empty)".to_string()]
                        } else {
                            entries.iter().map(|e| {
                                let tag = if e.is_dir() { "<DIR> " } else { "      " };
                                format!("  {}{}", tag, e.name())
                            }).collect()
                        }
                    }
                    Err(e) => alloc::vec![e.to_string()],
                };
                for line in listing { self.outln(&line); }
            }
            (_, "mkdir") => {
                let path = self.abs(arg1);
                match self.fs.mkdir(&path) {
                    Ok(_)  => { let s = format!("Created: {}", path); self.outln(&s); self.disk_sync(); }
                    Err(e) => self.outln(e),
                }
            }
            (_, "touch") => {
                let path = self.abs(arg1);
                match self.fs.touch(&path) {
                    Ok(_)  => { let s = format!("Created: {}", path); self.outln(&s); self.disk_sync(); }
                    Err(e) => self.outln(e),
                }
            }
            (_, "write") => {
                let path = self.abs(arg1);
                match self.fs.write_file(&path, arg2) {
                    Ok(_)  => { self.outln("Written."); self.disk_sync(); }
                    Err(e) => self.outln(e),
                }
            }
            (_, "cat") => {
                let path = self.abs(arg1);
                match self.fs.read_file(&path) {
                    Ok(data) => {
                        let s = core::str::from_utf8(data).unwrap_or("(binary)").to_string();
                        self.outln(&s);
                    }
                    Err(e) => self.outln(e),
                }
            }
            (_, "del") => {
                let path = self.abs(arg1);
                match self.fs.delete(&path) {
                    Ok(_)  => { self.outln("Deleted."); self.disk_sync(); }
                    Err(e) => self.outln(e),
                }
            }
            (_, "sync") => {
                self.disk_sync();
                self.outln("Filesystem synced to disk.");
            }
            (_, "cd") => {
                let path = self.abs(arg1);
                if path == "/" || self.fs.exists(&path) {
                    self.cwd = path;
                } else {
                    self.outln("No such directory.");
                }
            }
            (_, "edit") => {
                let path = self.abs(arg1);
                if !self.fs.exists(&path) { let _ = self.fs.touch(&path); }
                self.enter_editor(&path.clone());
            }
            (_, "cls") => { self.output.lines.clear(); }
            (_, "mem") => {
                let s = format!("Heap: {} KiB at 0x{:X}", crate::allocator::HEAP_SIZE / 1024, crate::allocator::HEAP_START);
                self.outln(&s);
            }
            (_, "storage") => {
                if storage::detect_ahci() { self.outln("AHCI detected."); }
                else { self.outln("No AHCI controller."); }
            }
            (_, "gpt") => {
                let s = format!("GptHeader size: {} bytes", core::mem::size_of::<storage::GptHeader>());
                self.outln(&s);
            }
            (_, "exfat") => {
                let s = format!("ExfatBootSector size: {} bytes", core::mem::size_of::<storage::ExfatBootSector>());
                self.outln(&s);
            }
            (_, "version") => { self.outln("NewDOS v0.1.1 (bootloader 0.11 / GUI)"); }
            (_, "banner")  => { self.outln("NewDOS v0.1.1 — bootloader 0.11 | x86_64 | VESA/GOP"); }
            (_, "whoami")  => { self.outln(role); }
            (_, "user")    => {
                if arg1.is_empty() {
                    let s = format!("User: {}", self.username);
                    self.outln(&s);
                } else {
                    self.username = arg1.to_string();
                    let s = format!("Username: {}", self.username);
                    self.outln(&s);
                }
            }
            (_, "device")  => {
                if arg1.is_empty() {
                    let s = format!("Device: {}", self.device);
                    self.outln(&s);
                } else {
                    self.device = arg1.to_string();
                    let s = format!("Device: {}", self.device);
                    self.outln(&s);
                }
            }
            (_, "time") => {
                let (gmt, loc) = time::formatted_times();
                let s = format!("GMT: {}  Local: {}", time::time_str(&gmt), time::time_str(&loc));
                self.outln(&s);
            }
            (_, "tz") => {
                let offset: i8 = arg1.parse().unwrap_or(0);
                time::set_timezone(offset);
                let s = format!("Timezone: UTC{:+}", offset);
                self.outln(&s);
            }
            ("suppiere", "gfx") => {
                crate::framebuffer::redraw_demo();
                self.outln("Framebuffer demo drawn.");
            }
            ("suppiere", "restart") => {
                self.outln("Restarting...");
                unsafe {
                    let mut p: x86_64::instructions::port::Port<u8> =
                        x86_64::instructions::port::Port::new(0x64);
                    p.write(0xFE);
                }
            }
            _ => {
                let s = format!("Unknown: '{}'. Try 'pierre help'.", raw);
                self.outln(&s);
            }
        }
    }

    fn abs(&self, path: &str) -> String {
        if path.starts_with('/') { path.to_string() }
        else if self.cwd == "/" { format!("/{}", path) }
        else { format!("{}/{}", self.cwd, path) }
    }
}
