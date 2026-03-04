//! Desktop background, taskbar, and desktop icon grid.

use crate::framebuffer::{FbWriter, Rgb};
use crate::time;

pub const TASKBAR_H: usize = 32;

// ── Background ────────────────────────────────────────────────────────────────

pub fn draw_background(fb: &mut FbWriter) {
    let sw = fb.width();
    let sh = fb.height();
    // Two-tone vertical gradient: deep navy top → slightly lighter bottom
    for y in 0..sh.saturating_sub(TASKBAR_H) {
        let t  = y * 255 / sh.max(1);
        let bg = Rgb { r: (10 + t / 16) as u8, g: (12 + t / 14) as u8, b: (28 + t / 6) as u8 };
        for x in 0..sw {
            fb.draw_pixel(x, y, bg);
        }
    }
}

// ── Taskbar ───────────────────────────────────────────────────────────────────

pub fn draw_taskbar(fb: &mut FbWriter, open_windows: &[&str]) {
    let sw = fb.width();
    let sh = fb.height();
    let ty = sh - TASKBAR_H;

    // Background
    fb.fill_rect(0, ty, sw, TASKBAR_H, Rgb { r: 18, g: 18, b: 38 });
    // Top border accent
    fb.fill_rect(0, ty, sw, 1, Rgb::ACCENT);

    // NewDOS logo / start
    fb.fill_rect(4, ty + 4, 24, 24, Rgb::ACCENT);
    fb.draw_str(8, ty + 10, "N", Rgb::BLACK, Rgb::ACCENT);
    fb.draw_str(36, ty + 11, "NewDOS", Rgb::ACCENT, Rgb { r: 18, g: 18, b: 38 });

    // Open window buttons
    let mut bx = 110usize;
    for &title in open_windows {
        let label_len = title.len().min(12);
        let bw = label_len * 8 + 16;
        fb.fill_rect(bx, ty + 5, bw, 22, Rgb { r: 35, g: 35, b: 65 });
        fb.draw_rect_outline(bx, ty + 5, bw, 22, Rgb { r: 80, g: 80, b: 120 });
        fb.draw_str(bx + 8, ty + 12, &title[..label_len], Rgb::WHITE, Rgb { r: 35, g: 35, b: 65 });
        bx += bw + 6;
    }

    // Clock (right side)
    let (gmt, _) = time::formatted_times();
    let time_s = time::time_str(&gmt);
    let time_short = if time_s.len() >= 5 { &time_s[..5] } else { time_s };
    let cx = sw.saturating_sub(time_short.len() * 8 + 12);
    fb.draw_str(cx, ty + 12, time_short, Rgb::WHITE, Rgb { r: 18, g: 18, b: 38 });
}

// ── Desktop icons ─────────────────────────────────────────────────────────────

pub struct Icon {
    pub x:     usize,
    pub y:     usize,
    pub label: &'static str,
    pub kind:  IconKind,
}

#[derive(Clone, Copy, PartialEq)]
pub enum IconKind { Terminal, FileManager, About, Settings }

pub const ICONS: &[Icon] = &[
    Icon { x: 24, y: 24, label: "Terminal",   kind: IconKind::Terminal    },
    Icon { x: 24, y: 96, label: "Files",       kind: IconKind::FileManager },
    Icon { x: 24, y: 168, label: "About",      kind: IconKind::About       },
    Icon { x: 24, y: 240, label: "Settings",   kind: IconKind::Settings    },
];

const ICON_SIZE: usize = 40;

pub fn draw_icons(fb: &mut FbWriter) {
    for icon in ICONS {
        draw_icon(fb, icon);
    }
}

fn draw_icon(fb: &mut FbWriter, icon: &Icon) {
    let x = icon.x;
    let y = icon.y;

    // Shadow
    fb.fill_rect(x + 3, y + 3, ICON_SIZE, ICON_SIZE, Rgb { r: 0, g: 0, b: 0 });
    // Icon bg
    let (bg, accent) = match icon.kind {
        IconKind::Terminal    => (Rgb { r: 20, g: 20, b: 20 }, Rgb::GREEN),
        IconKind::FileManager => (Rgb { r: 30, g: 20, b: 0  }, Rgb::YELLOW),
        IconKind::About       => (Rgb { r: 0,  g: 20, b: 40 }, Rgb::CYAN),
        IconKind::Settings    => (Rgb { r: 20, g: 10, b: 30 }, Rgb::MAGENTA),
    };
    fb.fill_rect(x, y, ICON_SIZE, ICON_SIZE, bg);
    fb.draw_rect_outline(x, y, ICON_SIZE, ICON_SIZE, accent);

    // Icon glyph
    match icon.kind {
        IconKind::Terminal => {
            fb.draw_str(x + 4,  y + 10, ">_", accent, bg);
        }
        IconKind::FileManager => {
            // Folder shape
            fb.fill_rect(x + 6,  y + 10, 28, 20, Rgb::YELLOW);
            fb.fill_rect(x + 6,  y + 8,  12, 4,  Rgb::YELLOW);
            fb.fill_rect(x + 7,  y + 11, 26, 18, Rgb { r: 200, g: 160, b: 0 });
        }
        IconKind::About => {
            fb.draw_str(x + 14, y + 9,  "i", accent, bg);
            fb.fill_rect(x + 17, y + 18, 6,  12, accent);
        }
        IconKind::Settings => {
            // Simple gear outline
            fb.fill_rect(x + 14, y + 8,  12, 24, accent);
            fb.fill_rect(x + 8,  y + 14, 24, 12, accent);
            fb.fill_rect(x + 14, y + 14, 12, 12, bg);
            fb.fill_rect(x + 17, y + 17, 6,  6,  accent);
        }
    }

    // Label below icon
    let lx = x.saturating_sub(icon.label.len() * 4).max(2);
    fb.draw_str(lx, y + ICON_SIZE + 4, icon.label, Rgb::WHITE,
                Rgb { r: 0, g: 0, b: 0 });
}

/// Returns which icon (if any) contains pixel (px, py).
pub fn hit_icon(px: i32, py: i32) -> Option<IconKind> {
    for icon in ICONS {
        let x = icon.x as i32;
        let y = icon.y as i32;
        let s = ICON_SIZE as i32;
        if px >= x && px < x + s && py >= y && py < y + s {
            return Some(icon.kind);
        }
    }
    None
}
