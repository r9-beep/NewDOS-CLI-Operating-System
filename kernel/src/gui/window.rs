//! Window struct — title bar, content area, dragging, close button.

use crate::framebuffer::{FbWriter, Rgb};
use crate::keyboard::SpecialKey;
use crate::shell::ShellState;
use alloc::string::{String, ToString};

pub const TITLE_H:    i32 = 22;
pub const BORDER:     i32 = 2;
pub const MIN_W:      i32 = 200;
pub const MIN_H:      i32 = 120;

// ── Window content ────────────────────────────────────────────────────────────

pub enum WindowContent {
    Terminal(ShellState),
    About,
    FileManager,
}

// ── Hit zone ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HitZone { TitleBar, CloseButton, Content, Border, None }

// ── Window ────────────────────────────────────────────────────────────────────

pub struct Window {
    pub x:       i32,
    pub y:       i32,
    pub w:       i32,
    pub h:       i32,
    pub title:   &'static str,
    pub content: WindowContent,
    pub closed:  bool,
}

impl Window {
    pub fn new(x: i32, y: i32, w: i32, h: i32, title: &'static str, content: WindowContent) -> Self {
        Window { x, y, w, h, title, content, closed: false }
    }

    // ── Geometry helpers ──────────────────────────────────────────────────────

    pub fn title_bar_rect(&self) -> (i32, i32, i32, i32) {
        (self.x, self.y, self.w, TITLE_H)
    }

    pub fn close_btn_rect(&self) -> (i32, i32, i32, i32) {
        let bx = self.x + self.w - TITLE_H;
        (bx, self.y, TITLE_H, TITLE_H)
    }

    pub fn content_rect(&self) -> (i32, i32, i32, i32) {
        let cx = self.x + BORDER;
        let cy = self.y + TITLE_H;
        let cw = self.w - BORDER * 2;
        let ch = self.h - TITLE_H - BORDER;
        (cx, cy, cw.max(0), ch.max(0))
    }

    pub fn hit_test(&self, px: i32, py: i32) -> HitZone {
        if px < self.x || px >= self.x + self.w || py < self.y || py >= self.y + self.h {
            return HitZone::None;
        }
        let (bx, by, bw, bh) = self.close_btn_rect();
        if px >= bx && px < bx + bw && py >= by && py < by + bh {
            return HitZone::CloseButton;
        }
        if py < self.y + TITLE_H { return HitZone::TitleBar; }
        HitZone::Content
    }

    // ── Input ─────────────────────────────────────────────────────────────────

    pub fn process_char(&mut self, c: char) {
        if let WindowContent::Terminal(ref mut sh) = self.content {
            sh.process_char(c);
        }
    }

    pub fn process_special(&mut self, k: SpecialKey) {
        if let WindowContent::Terminal(ref mut sh) = self.content {
            sh.process_special(k);
        }
    }

    // ── Drawing ───────────────────────────────────────────────────────────────

    pub fn draw(&self, fb: &mut FbWriter, focused: bool) {
        // Drop shadow
        fb.fill_rect(
            (self.x + 5) as usize, (self.y + 5) as usize,
            self.w as usize, self.h as usize,
            Rgb { r: 0, g: 0, b: 0 },
        );

        // Window border
        let border_col = if focused { Rgb::ACCENT } else { Rgb { r: 60, g: 60, b: 90 } };
        fb.fill_rect(self.x as usize, self.y as usize, self.w as usize, self.h as usize, border_col);

        // Title bar
        let tb_bg = if focused {
            Rgb { r: 25, g: 35, b: 70 }
        } else {
            Rgb { r: 20, g: 20, b: 40 }
        };
        fb.fill_rect(
            (self.x + BORDER) as usize, (self.y + BORDER) as usize,
            (self.w - BORDER * 2) as usize, (TITLE_H - BORDER) as usize,
            tb_bg,
        );

        // Title text
        fb.draw_str(
            (self.x + 8) as usize, (self.y + 7) as usize,
            self.title,
            if focused { Rgb::WHITE } else { Rgb::GRAY },
            tb_bg,
        );

        // Close button
        let (bx, by, bw, bh) = self.close_btn_rect();
        fb.fill_rect(bx as usize, by as usize, bw as usize, bh as usize, Rgb::RED);
        let mid_x = (bx + bw / 2 - 4) as usize;
        let mid_y = (by + bh / 2 - 4) as usize;
        fb.draw_str(mid_x, mid_y, "X", Rgb::WHITE, Rgb::RED);

        // Content area background
        let (cx, cy, cw, ch) = self.content_rect();
        let content_bg = Rgb { r: 13, g: 13, b: 20 };
        fb.fill_rect(cx as usize, cy as usize, cw as usize, ch as usize, content_bg);

        // Content
        match &self.content {
            WindowContent::Terminal(sh) => self.draw_terminal(fb, cx, cy, cw, ch, sh),
            WindowContent::About        => self.draw_about(fb, cx, cy, cw, ch),
            WindowContent::FileManager  => self.draw_filemanager(fb, cx, cy, cw, ch),
        }
    }

    // ── Terminal renderer ─────────────────────────────────────────────────────

    fn draw_terminal(&self, fb: &mut FbWriter, cx: i32, cy: i32, cw: i32, ch: i32, sh: &ShellState) {
        let line_h  = 10usize;
        let pad     = 6usize;
        let max_vis = (ch as usize).saturating_sub(pad * 2 + line_h) / line_h;

        // Scrollback lines
        let lines = sh.output.tail(max_vis);
        for (i, line) in lines.iter().enumerate() {
            let lx = (cx as usize) + pad;
            let ly = (cy as usize) + pad + i * line_h;
            if ly + line_h > (cy + ch) as usize { break; }
            // Colour-code lines that start with known prefixes
            let fg = if line.contains("error") || line.contains("PANIC") {
                Rgb::RED
            } else if line.contains("pierre@") || line.contains("suppiere@") {
                Rgb::GREEN
            } else if line.starts_with("╔") || line.starts_with("║") || line.starts_with("╚") {
                Rgb::ACCENT
            } else {
                Rgb::WHITE
            };
            let display = if line.len() * 8 > cw as usize { &line[..cw as usize / 8] } else { line };
            fb.draw_str(lx, ly, display, fg, Rgb { r: 13, g: 13, b: 20 });
        }

        // Input line at bottom
        let prompt = sh.prompt_string();
        let input  = &sh.output.input;
        let full   = alloc::format!("{}{}_", prompt, input);
        let iy = (cy + ch - line_h as i32 - pad as i32) as usize;
        let trim_len = (cw as usize / 8).min(full.len());
        fb.draw_str((cx as usize) + pad, iy, &full[..trim_len], Rgb::GREEN,
                    Rgb { r: 13, g: 13, b: 20 });
    }

    // ── About renderer ────────────────────────────────────────────────────────

    fn draw_about(&self, fb: &mut FbWriter, cx: i32, cy: i32, _cw: i32, _ch: i32) {
        let bg = Rgb { r: 13, g: 13, b: 20 };
        let lines: &[(&str, Rgb)] = &[
            ("NewDOS v0.1.1",                          Rgb::ACCENT),
            ("",                                        Rgb::WHITE),
            ("Bootloader: 0.11 (BIOS + UEFI)",         Rgb::WHITE),
            ("Graphics:   VESA VBE / UEFI GOP",         Rgb::GREEN),
            ("Video out:  HDMI / DisplayPort / VGA",    Rgb::GREEN),
            ("Arch:       x86_64 bare metal (Rust)",    Rgb::CYAN),
            ("Keyboard:   PS/2 IRQ1",                   Rgb::YELLOW),
            ("Mouse:      PS/2 IRQ12",                  Rgb::YELLOW),
            ("Memory:     PMM + linked_list heap",      Rgb::MAGENTA),
            ("",                                        Rgb::WHITE),
            ("(c) 2026 r9-beep  BSD 2-Clause",          Rgb::GRAY),
        ];
        for (i, (text, col)) in lines.iter().enumerate() {
            fb.draw_str((cx + 12) as usize, (cy + 12 + i as i32 * 14) as usize, text, *col, bg);
        }
    }

    // ── File manager renderer ─────────────────────────────────────────────────

    fn draw_filemanager(&self, fb: &mut FbWriter, cx: i32, cy: i32, cw: i32, ch: i32) {
        let bg = Rgb { r: 13, g: 13, b: 20 };
        fb.draw_str((cx + 8) as usize, (cy + 8) as usize, "/ (root)", Rgb::YELLOW, bg);
        fb.draw_str((cx + 8) as usize, (cy + 22) as usize,
                    "No disk driver — VFS is in-memory only.", Rgb::GRAY, bg);
        fb.draw_str((cx + 8) as usize, (cy + 36) as usize,
                    "Use Terminal: pierre ls / mkdir / touch", Rgb::WHITE, bg);
        let _ = (cw, ch);
    }
}
