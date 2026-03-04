//! Full-screen CLI terminal mode.
//!
//! Renders the shell's TextBuffer directly onto the entire framebuffer.
//! No mouse, no windows — just a classic scrolling terminal.

use crate::{
    framebuffer::{Rgb, FB_WRITER},
    keyboard::{self, SpecialKey},
    shell::ShellState,
};
use alloc::format;

const BG:      Rgb = Rgb { r: 13, g: 13, b: 20 };
const LINE_H:  usize = 10;
const PAD:     usize = 6;

pub struct CliMode {
    pub shell: ShellState,
    dirty:     bool,
}

impl CliMode {
    pub fn new() -> Self {
        let mut sh = ShellState::new();
        sh.init();
        let mut m = CliMode { shell: sh, dirty: true };
        m
    }

    pub fn tick(&mut self) {
        if let Some(c) = keyboard::pop_char() {
            self.shell.process_char(c);
            self.dirty = true;
        }
        if let Some(k) = keyboard::pop_special() {
            self.shell.process_special(k);
            self.dirty = true;
        }
        if self.dirty {
            self.render();
            self.dirty = false;
        }
    }

    fn render(&self) {
        let mut guard = FB_WRITER.lock();
        let fb = match guard.as_mut() { Some(f) => f, None => return };

        let sw   = fb.width();
        let sh   = fb.height();

        fb.fill_rect(0, 0, sw, sh, BG);
        // Accent top bar
        fb.fill_rect(0, 0, sw, 2, Rgb::ACCENT);
        // Status line at very bottom
        let bar_y = sh.saturating_sub(14);
        fb.fill_rect(0, bar_y, sw, 14, Rgb { r: 10, g: 10, b: 38 });
        fb.fill_rect(0, bar_y, sw, 1, Rgb::GRAY);
        fb.draw_str(PAD, bar_y + 3,
            "NewDOS CLI  |  pierre help  |  F2=GUI mode",
            Rgb::GRAY, Rgb { r: 10, g: 10, b: 38 });

        // Input line
        let prompt = self.shell.prompt_string();
        let input  = &self.shell.output.input;
        let full   = format!("{}{}_", prompt, input);
        let input_y = bar_y.saturating_sub(LINE_H + PAD);
        fb.fill_rect(0, input_y.saturating_sub(2), sw, 1, Rgb { r: 40, g: 40, b: 60 });
        let max_chars = (sw.saturating_sub(PAD * 2)) / 8;
        let trimmed = if full.len() > max_chars {
            &full[full.len().saturating_sub(max_chars)..]
        } else {
            full.as_str()
        };
        fb.draw_str(PAD, input_y, trimmed, Rgb::GREEN, BG);

        // Scrollback lines (fill remaining height)
        let available_h = input_y.saturating_sub(PAD + 2);
        let max_lines   = available_h / LINE_H;
        let lines       = self.shell.output.tail(max_lines);
        for (i, line) in lines.iter().enumerate() {
            let y = PAD + 2 + i * LINE_H;
            if y + LINE_H > input_y { break; }
            let fg = colour_for_line(line);
            let max_ch  = (sw.saturating_sub(PAD * 2)) / 8;
            let display = if line.len() > max_ch { &line[..max_ch] } else { line };
            fb.draw_str(PAD, y, display, fg, BG);
        }
    }
}

fn colour_for_line(line: &str) -> Rgb {
    if line.contains("pierre@") || line.contains("suppiere@") {
        Rgb::GREEN
    } else if line.starts_with("╔") || line.starts_with("║") || line.starts_with("╚") {
        Rgb::ACCENT
    } else if line.contains("error") || line.to_lowercase().contains("panic") {
        Rgb::RED
    } else {
        Rgb::WHITE
    }
}
