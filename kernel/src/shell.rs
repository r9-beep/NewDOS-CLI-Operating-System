//! NewDOS interactive shell.
//! Role prefix: `pierre` = user, `suppiere` = admin.

use crate::{framebuffer, storage, time, vfs};
use alloc::{
    format,
    string::{String, ToString},
    vec::Vec,
};
use spin::Mutex;

// ── special key codes ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpecialKey { F1, F2, F3, F9, F10 }

// ── global input queue (written by keyboard ISR) ───────────────────────────

static INPUT_CHAR:    Mutex<Option<char>>       = Mutex::new(None);
static INPUT_SPECIAL: Mutex<Option<SpecialKey>> = Mutex::new(None);

pub fn push_char(c: char)        { *INPUT_CHAR.lock()    = Some(c); }
pub fn push_special(k: SpecialKey) { *INPUT_SPECIAL.lock() = Some(k); }

fn pop_char()    -> Option<char>       { INPUT_CHAR.lock().take() }
fn pop_special() -> Option<SpecialKey> { INPUT_SPECIAL.lock().take() }

// ── shell state ───────────────────────────────────────────────────────────────

pub struct ShellState {
    pub fs:          vfs::FileSystem,
    pub cwd:         String,
    pub username:    String,
    pub device:      String,
    pub tz_offset:   i8,
    pub line:        String,
    /// `None` = shell mode, `Some(path)` = editor mode
    pub editor_file: Option<String>,
    pub editor_buf:  String,
}

impl ShellState {
    pub fn new() -> Self {
        ShellState {
            fs:          vfs::FileSystem::new(),
            cwd:         String::from("/"),
            username:    String::from("pierre"),
            device:      String::from("NewDOS-PC"),
            tz_offset:   0,
            line:        String::new(),
            editor_file: None,
            editor_buf:  String::new(),
        }
    }

    fn print(&self, s: &str) {
        // Try framebuffer first, fall back to VGA
        if framebuffer::FB_WRITER.lock().is_some() {
            crate::fb_print!("{}", s);
        } else {
            crate::print!("{}", s);
        }
    }

    fn println(&self, s: &str) {
        self.print(s);
        self.print("\n");
    }

    pub fn prompt(&self) {
        let p = format!("{}@{}:{}> ", self.username, self.device, self.cwd);
        self.print(&p);
    }

    // ── editor ───────────────────────────────────────────────────────────────

    fn editor_draw(&self) {
        self.println("┌── editor ── F9=Save  F10=Exit ──");
        for line in self.editor_buf.split('\n') {
            self.print("│ ");
            self.println(line);
        }
        self.println("└─────────────────────────────────");
    }

    fn enter_editor(&mut self, path: &str) {
        let content = match self.fs.read_file(path) {
            Ok(b) => core::str::from_utf8(b).unwrap_or("").to_string(),
            Err(_) => String::new(),
        };
        self.editor_file = Some(path.to_string());
        self.editor_buf  = content;
        self.editor_draw();
    }

    fn editor_input(&mut self, c: char) {
        match c {
            '\x08' => { self.editor_buf.pop(); }
            _       => { self.editor_buf.push(c); }
        }
    }

    fn editor_save(&mut self) {
        if let Some(ref path) = self.editor_file.clone() {
            match self.fs.write_file(path, &self.editor_buf.clone()) {
                Ok(_)  => self.println("Saved."),
                Err(e) => self.println(e),
            }
        }
    }

    fn editor_exit(&mut self) {
        self.editor_file = None;
        self.editor_buf  = String::new();
        self.println("Editor closed.");
        self.prompt();
    }

    // ── command dispatch ──────────────────────────────────────────────────────

    fn dispatch(&mut self, raw: &str) {
        let raw = raw.trim();
        if raw.is_empty() { self.prompt(); return; }

        let (role, rest) = if let Some(r) = raw.strip_prefix("pierre ") {
            ("pierre", r)
        } else if let Some(r) = raw.strip_prefix("suppiere ") {
            ("suppiere", r)
        } else if raw == "pierre" || raw == "suppiere" {
            (raw, "")
        } else {
            self.println("Unknown command. Type 'pierre help'.");
            self.prompt();
            return;
        };

        let args: Vec<&str> = rest.splitn(3, ' ').collect();
        let cmd  = args.get(0).copied().unwrap_or("");
        let arg1 = args.get(1).copied().unwrap_or("");
        let arg2 = args.get(2).copied().unwrap_or("");

        match (role, cmd) {
            // ── help ──────────────────────────────────────────────────────────
            (_, "help") | (_, "") => {
                self.println("NewDOS shell commands");
                self.println("─────────────────────────────────────────────");
                self.println("  pierre help           this help");
                self.println("  pierre ls / dir       list directory");
                self.println("  pierre mkdir <path>   create directory");
                self.println("  pierre touch <path>   create file");
                self.println("  pierre write <f> <d>  write data to file");
                self.println("  pierre cat <path>     read file");
                self.println("  pierre del <path>     delete file/dir");
                self.println("  pierre cls            clear screen");
                self.println("  pierre mem            memory info");
                self.println("  pierre storage        storage detection");
                self.println("  pierre gpt            GPT struct info");
                self.println("  pierre exfat          exFAT struct info");
                self.println("  pierre time           show RTC time");
                self.println("  pierre tz <+/-N>      set timezone");
                self.println("  pierre edit <file>    open editor (F9=save F10=exit)");
                self.println("  pierre user <name>    set username");
                self.println("  pierre device <name>  set device name");
                self.println("  pierre whoami         show role");
                self.println("  pierre version        kernel version");
                self.println("  pierre banner         print boot banner");
                self.println("  pierre cd <path>      change directory");
                self.println("  suppiere gfx          redraw framebuffer demo");
                self.println("  suppiere restart      reboot system");
            }

            // ── filesystem ────────────────────────────────────────────────────
            (_, "ls") | (_, "dir") => {
                let path = if arg1.is_empty() { self.cwd.clone() } else { arg1.to_string() };
                match self.fs.list(&path) {
                    Ok(entries) => {
                        if entries.is_empty() {
                            self.println("(empty)");
                        } else {
                            for e in entries {
                                let tag = if e.is_dir() { "<DIR> " } else { "      " };
                                self.println(&format!("  {}{}", tag, e.name()));
                            }
                        }
                    }
                    Err(e) => self.println(e),
                }
            }

            (_, "mkdir") => {
                let path = self.abs(arg1);
                match self.fs.mkdir(&path) {
                    Ok(_)  => self.println(&format!("Created directory: {}", path)),
                    Err(e) => self.println(e),
                }
            }

            (_, "touch") => {
                let path = self.abs(arg1);
                match self.fs.touch(&path) {
                    Ok(_)  => self.println(&format!("Created file: {}", path)),
                    Err(e) => self.println(e),
                }
            }

            (_, "write") => {
                let path = self.abs(arg1);
                match self.fs.write_file(&path, arg2) {
                    Ok(_)  => self.println("Written."),
                    Err(e) => self.println(e),
                }
            }

            (_, "cat") => {
                let path = self.abs(arg1);
                match self.fs.read_file(&path) {
                    Ok(data) => {
                        match core::str::from_utf8(data) {
                            Ok(s)  => self.println(s),
                            Err(_) => self.println("(binary data)"),
                        }
                    }
                    Err(e) => self.println(e),
                }
            }

            (_, "del") => {
                let path = self.abs(arg1);
                match self.fs.delete(&path) {
                    Ok(_)  => self.println("Deleted."),
                    Err(e) => self.println(e),
                }
            }

            (_, "cd") => {
                let path = self.abs(arg1);
                if path == "/" || self.fs.exists(&path) {
                    self.cwd = path;
                } else {
                    self.println("No such directory.");
                }
            }

            // ── editor ────────────────────────────────────────────────────────
            (_, "edit") => {
                let path = self.abs(arg1);
                if !self.fs.exists(&path) {
                    let _ = self.fs.touch(&path);
                }
                self.enter_editor(&path.clone());
                return; // don't print prompt yet
            }

            // ── system info ───────────────────────────────────────────────────
            (_, "cls") => {
                if let Some(w) = framebuffer::FB_WRITER.lock().as_mut() {
                    w.clear(crate::framebuffer::Rgb::DARKBG);
                } else {
                    crate::vga::WRITER.lock().clear_screen();
                }
            }

            (_, "mem") => {
                self.println("Memory regions provided by bootloader 0.11:");
                self.println("  (Region detail requires passing BootInfo to shell)");
                let heap_used = {
                    // Approximate: allocator reports usage via linked_list_allocator
                    0usize // placeholder
                };
                let _ = heap_used;
                self.println(&format!("  Heap: {} KiB reserved at 0x{:X}", crate::allocator::HEAP_SIZE / 1024, crate::allocator::HEAP_START));
            }

            (_, "storage") => {
                if storage::detect_ahci() {
                    self.println("AHCI controller detected (SATA/SSD present).");
                } else {
                    self.println("No AHCI controller detected.");
                }
            }

            (_, "gpt") => {
                self.println("GPT header struct (no disk driver; layout reference):");
                self.println(&format!("  Signature field size: {} bytes", core::mem::size_of::<[u8;8]>()));
                self.println(&format!("  Full GptHeader size:  {} bytes", core::mem::size_of::<storage::GptHeader>()));
                self.println("  Signature: EFI PART");
            }

            (_, "exfat") => {
                self.println("exFAT boot sector struct (no disk driver; layout reference):");
                self.println(&format!("  Boot sector size: {} bytes", core::mem::size_of::<storage::ExfatBootSector>()));
            }

            (_, "version") => {
                self.println("NewDOS v0.1.1 (bootloader 0.11 / VESA+GOP framebuffer build)");
            }

            (_, "banner") => {
                self.print_banner();
            }

            (_, "whoami") => {
                self.println(role);
            }

            (_, "user") => {
                if arg1.is_empty() {
                    self.println(&format!("Current user: {}", self.username));
                } else {
                    self.username = arg1.to_string();
                    self.println(&format!("Username set to: {}", self.username));
                }
            }

            (_, "device") => {
                if arg1.is_empty() {
                    self.println(&format!("Device: {}", self.device));
                } else {
                    self.device = arg1.to_string();
                    self.println(&format!("Device name set to: {}", self.device));
                }
            }

            // ── time ──────────────────────────────────────────────────────────
            (_, "time") => {
                let (gmt, loc) = time::formatted_times();
                self.println(&format!("GMT:   {}", time::time_str(&gmt)));
                self.println(&format!("Local: {}", time::time_str(&loc)));
            }

            (_, "tz") => {
                if arg1.is_empty() {
                    self.println(&format!("Timezone: UTC{:+}", time::get_timezone()));
                } else {
                    let offset: i8 = arg1.parse().unwrap_or(0);
                    time::set_timezone(offset);
                    self.tz_offset = offset;
                    self.println(&format!("Timezone set to UTC{:+}", offset));
                }
            }

            // ── admin (suppiere) ──────────────────────────────────────────────
            ("suppiere", "gfx") => {
                framebuffer::redraw_demo();
                self.println("Framebuffer demo redrawn.");
            }

            ("suppiere", "restart") => {
                self.println("Restarting...");
                unsafe {
                    let mut port: x86_64::instructions::port::Port<u8> =
                        x86_64::instructions::port::Port::new(0x64);
                    port.write(0xFE);
                }
            }

            // ── unknown ───────────────────────────────────────────────────────
            _ => {
                self.println(&format!("Unknown command: '{}'. Try 'pierre help'.", raw));
            }
        }

        self.prompt();
    }

    fn abs(&self, path: &str) -> String {
        if path.starts_with('/') {
            path.to_string()
        } else if self.cwd == "/" {
            format!("/{}", path)
        } else {
            format!("{}/{}", self.cwd, path)
        }
    }

    fn print_banner(&self) {
        self.println("╔══════════════════════════════════════════════════════╗");
        self.println("║  NewDOS v0.1.1  —  bootloader 0.11  |  x86_64       ║");
        self.println("║  VESA VBE / UEFI GOP framebuffer  (HDMI / DP / VGA) ║");
        self.println("║  Type 'pierre help' for commands.                   ║");
        self.println("╚══════════════════════════════════════════════════════╝");
    }

    // ── main tick (called from kernel main loop) ──────────────────────────────

    pub fn tick(&mut self) {
        // Editor mode
        if self.editor_file.is_some() {
            if let Some(k) = pop_special() {
                match k {
                    SpecialKey::F9  => self.editor_save(),
                    SpecialKey::F10 => self.editor_exit(),
                    _ => {}
                }
            }
            if let Some(c) = pop_char() {
                self.editor_input(c);
            }
            return;
        }

        // Normal shell mode
        if let Some(c) = pop_char() {
            match c {
                '\n' | '\r' => {
                    self.print("\n");
                    let cmd = core::mem::take(&mut self.line);
                    self.dispatch(&cmd);
                }
                '\x08' => {
                    // backspace
                    if !self.line.is_empty() {
                        self.line.pop();
                        self.print("\x08 \x08");
                    }
                }
                c if c.is_ascii() && !c.is_control() => {
                    self.line.push(c);
                    let s = format!("{}", c);
                    self.print(&s);
                }
                _ => {}
            }
        }
    }

    pub fn init(&mut self) {
        self.print_banner();
        self.println("");
        self.println("Video: VESA VBE (BIOS) / UEFI GOP -> HDMI / DisplayPort / VGA");
        self.println("Bootloader: 0.11 |  Arch: x86_64");
        self.println("");
        self.prompt();
    }
}
