//! NewDOS desktop GUI — desktop, taskbar, draggable windows, mouse cursor.

pub mod cursor;
pub mod desktop;
pub mod window;

use crate::{framebuffer, keyboard, mouse, shell, time};
use alloc::vec::Vec;
use window::{HitZone, Window, WindowContent};

// ── Gui ───────────────────────────────────────────────────────────────────────

pub struct Gui {
    pub windows:   Vec<Window>,
    pub focused:   Option<usize>,
    drag:          Option<DragState>,
    prev_buttons:  u8,
    screen_w:      usize,
    screen_h:      usize,
}

struct DragState {
    idx:      usize,
    offset_x: i32,
    offset_y: i32,
}

impl Gui {
    pub fn new(screen_w: usize, screen_h: usize) -> Self {
        mouse::set_bounds(screen_w, screen_h);

        let mut sh = shell::ShellState::new_from_disk();
        sh.init();

        let mut gui = Gui {
            windows:      Vec::new(),
            focused:      None,
            drag:         None,
            prev_buttons: 0,
            screen_w,
            screen_h,
        };

        // Default windows
        let term_w = (screen_w as i32 * 6 / 10).max(480);
        let term_h = (screen_h as i32 * 6 / 10).max(300);
        gui.windows.push(Window::new(
            80, 60, term_w, term_h,
            "Terminal",
            WindowContent::Terminal(sh),
        ));

        let about_w = 320i32;
        let about_h = 220i32;
        gui.windows.push(Window::new(
            screen_w as i32 - about_w - 40,
            60,
            about_w,
            about_h,
            "About NewDOS",
            WindowContent::About,
        ));

        gui.focused = Some(0);
        gui
    }

    // ── Main tick ─────────────────────────────────────────────────────────────

    pub fn tick(&mut self) {
        let (mx, my, buttons) = {
            let s = mouse::STATE.lock();
            (s.x, s.y, s.buttons)
        };
        mouse::STATE.lock().updated = false;

        let lbtn_now  = buttons & 1 != 0;
        let lbtn_prev = self.prev_buttons & 1 != 0;
        let lbtn_down = lbtn_now && !lbtn_prev;   // leading edge
        let lbtn_up   = !lbtn_now && lbtn_prev;   // trailing edge
        self.prev_buttons = buttons;

        // ── Mouse drag ────────────────────────────────────────────────────────
        if lbtn_up {
            self.drag = None;
        }
        if let Some(ref drag) = self.drag {
            let idx = drag.idx;
            let ox  = drag.offset_x;
            let oy  = drag.offset_y;
            let sw  = self.screen_w as i32;
            let sh  = self.screen_h as i32 - desktop::TASKBAR_H as i32;
            let win = &mut self.windows[idx];
            win.x = (mx - ox).clamp(-(win.w - 40), sw - 20);
            win.y = (my - oy).clamp(0, sh - window::TITLE_H);
        }

        // ── Mouse click ───────────────────────────────────────────────────────
        if lbtn_down {
            // Check from front (last) to back
            let mut hit_idx = None;
            for i in (0..self.windows.len()).rev() {
                let zone = self.windows[i].hit_test(mx, my);
                if zone != HitZone::None {
                    hit_idx = Some((i, zone));
                    break;
                }
            }

            if let Some((idx, zone)) = hit_idx {
                // Bring to front
                self.focused = Some(idx);

                match zone {
                    HitZone::CloseButton => {
                        self.windows[idx].closed = true;
                    }
                    HitZone::TitleBar => {
                        let win = &self.windows[idx];
                        self.drag = Some(DragState {
                            idx,
                            offset_x: mx - win.x,
                            offset_y: my - win.y,
                        });
                    }
                    HitZone::Content => {
                        // Could handle content clicks here
                    }
                    _ => {}
                }
            } else {
                // Click on desktop
                let icon = desktop::hit_icon(mx, my);
                if let Some(kind) = icon {
                    self.open_icon(kind);
                }
            }
        }

        // Remove closed windows
        self.windows.retain(|w| !w.closed);
        if let Some(f) = self.focused {
            if f >= self.windows.len() {
                self.focused = self.windows.len().checked_sub(1);
            }
        }

        // ── Keyboard → focused window ─────────────────────────────────────────
        if let Some(c) = keyboard::pop_char() {
            if let Some(idx) = self.focused {
                if let Some(win) = self.windows.get_mut(idx) {
                    win.process_char(c);
                }
            }
        }
        if let Some(k) = keyboard::pop_special() {
            if let Some(idx) = self.focused {
                if let Some(win) = self.windows.get_mut(idx) {
                    win.process_special(k);
                }
            }
        }

        // ── Draw everything ───────────────────────────────────────────────────
        self.draw(mx, my);
    }

    // ── Rendering ─────────────────────────────────────────────────────────────

    fn draw(&self, mx: i32, my: i32) {
        let mut guard = framebuffer::FB_WRITER.lock();
        if let Some(fb) = guard.as_mut() {
            // Desktop background
            desktop::draw_background(fb);

            // Desktop icons
            desktop::draw_icons(fb);

            // Windows (back to front; focused last)
            let focused = self.focused;
            for (i, win) in self.windows.iter().enumerate() {
                if focused != Some(i) {
                    win.draw(fb, false);
                }
            }
            if let Some(idx) = focused {
                if let Some(win) = self.windows.get(idx) {
                    win.draw(fb, true);
                }
            }

            // Taskbar
            let titles: Vec<&str> = self.windows.iter().map(|w| w.title).collect();
            desktop::draw_taskbar(fb, &titles);

            // Cursor on top
            cursor::draw(fb, mx, my);
        }
    }

    // ── Open window from icon click ───────────────────────────────────────────

    fn open_icon(&mut self, kind: desktop::IconKind) {
        // If a window of this type already exists, just focus it
        for (i, win) in self.windows.iter().enumerate() {
            let matches = match (&win.content, kind) {
                (WindowContent::Terminal(_), desktop::IconKind::Terminal)     => true,
                (WindowContent::About,       desktop::IconKind::About)        => true,
                (WindowContent::FileManager, desktop::IconKind::FileManager)  => true,
                _ => false,
            };
            if matches {
                self.focused = Some(i);
                return;
            }
        }

        // Otherwise open a new window
        let sw = self.screen_w as i32;
        let sh = self.screen_h as i32;
        let off = (self.windows.len() as i32) * 30;
        let (title, content, w, h) = match kind {
            desktop::IconKind::Terminal => {
                let mut sh_state = shell::ShellState::new_from_disk();
                sh_state.init();
                ("Terminal", WindowContent::Terminal(sh_state),
                 sw * 6 / 10, sh * 6 / 10)
            }
            desktop::IconKind::About => {
                ("About NewDOS", WindowContent::About, 320, 220)
            }
            desktop::IconKind::FileManager => {
                ("File Manager", WindowContent::FileManager, 360, 280)
            }
            desktop::IconKind::Settings => {
                // Settings: reuse About for now
                ("Settings", WindowContent::About, 320, 220)
            }
        };
        let nx = (100 + off).clamp(0, sw - w - 20);
        let ny = (60  + off).clamp(0, sh - h - 60);
        self.windows.push(Window::new(nx, ny, w, h, title, content));
        self.focused = Some(self.windows.len() - 1);
    }
}
