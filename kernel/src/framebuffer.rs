//! VESA / UEFI GOP framebuffer graphics and digital video output.
//!
//! The bootloader (0.11) automatically sets up a linear pixel framebuffer using
//! either VESA VBE (BIOS) or UEFI GOP.  We receive a pointer to that buffer in
//! `BootInfo::framebuffer` and write RGB pixels directly into it.  The GPU then
//! outputs the framebuffer on every connected digital (HDMI / DisplayPort / DVI)
//! and analogue (VGA) display.

use bootloader_api::info::{FrameBuffer, FrameBufferInfo, PixelFormat};
use core::fmt;
use spin::Mutex;

// ── embedded 8×8 bitmap font (ASCII 0x20-0x7F) ──────────────────────────────

static FONT8X8: [[u8; 8]; 96] = include_font();

const fn include_font() -> [[u8; 8]; 96] {
    let mut f = [[0u8; 8]; 96];
    // space
    // '!'
    f[1]  = [0x18,0x18,0x18,0x18,0x18,0x00,0x18,0x00];
    f[2]  = [0x66,0x66,0x66,0x00,0x00,0x00,0x00,0x00]; // '"'
    f[3]  = [0x66,0x66,0xFF,0x66,0xFF,0x66,0x66,0x00]; // '#'
    f[4]  = [0x18,0x7E,0x06,0x3E,0x60,0x7E,0x18,0x00]; // '$'
    f[5]  = [0x46,0x66,0x30,0x18,0x0C,0x66,0x62,0x00]; // '%'
    f[6]  = [0x3C,0x66,0x3C,0x1C,0xE6,0x66,0xFC,0x00]; // '&'
    f[7]  = [0x18,0x18,0x18,0x00,0x00,0x00,0x00,0x00]; // '\''
    f[8]  = [0x30,0x18,0x0C,0x0C,0x0C,0x18,0x30,0x00]; // '('
    f[9]  = [0x0C,0x18,0x30,0x30,0x30,0x18,0x0C,0x00]; // ')'
    f[10] = [0x00,0x66,0x3C,0xFF,0x3C,0x66,0x00,0x00]; // '*'
    f[11] = [0x00,0x18,0x18,0x7E,0x18,0x18,0x00,0x00]; // '+'
    f[12] = [0x00,0x00,0x00,0x00,0x00,0x18,0x18,0x0C]; // ','
    f[13] = [0x00,0x00,0x00,0x7E,0x00,0x00,0x00,0x00]; // '-'
    f[14] = [0x00,0x00,0x00,0x00,0x00,0x18,0x18,0x00]; // '.'
    f[15] = [0x40,0x60,0x30,0x18,0x0C,0x06,0x02,0x00]; // '/'
    // 0-9
    f[16] = [0x3C,0x66,0x76,0x7E,0x6E,0x66,0x3C,0x00];
    f[17] = [0x18,0x1C,0x18,0x18,0x18,0x18,0x7E,0x00];
    f[18] = [0x3C,0x66,0x60,0x30,0x18,0x0C,0x7E,0x00];
    f[19] = [0x3C,0x66,0x60,0x38,0x60,0x66,0x3C,0x00];
    f[20] = [0x30,0x38,0x3C,0x36,0x7E,0x30,0x30,0x00];
    f[21] = [0x7E,0x06,0x3E,0x60,0x60,0x66,0x3C,0x00];
    f[22] = [0x38,0x0C,0x06,0x3E,0x66,0x66,0x3C,0x00];
    f[23] = [0x7E,0x60,0x30,0x18,0x0C,0x0C,0x0C,0x00];
    f[24] = [0x3C,0x66,0x66,0x3C,0x66,0x66,0x3C,0x00];
    f[25] = [0x3C,0x66,0x66,0x7C,0x60,0x30,0x1C,0x00];
    f[26] = [0x00,0x18,0x18,0x00,0x00,0x18,0x18,0x00]; // ':'
    f[27] = [0x00,0x18,0x18,0x00,0x00,0x18,0x18,0x0C]; // ';'
    f[28] = [0x30,0x18,0x0C,0x06,0x0C,0x18,0x30,0x00]; // '<'
    f[29] = [0x00,0x00,0x7E,0x00,0x7E,0x00,0x00,0x00]; // '='
    f[30] = [0x06,0x0C,0x18,0x30,0x18,0x0C,0x06,0x00]; // '>'
    f[31] = [0x3C,0x66,0x60,0x30,0x18,0x00,0x18,0x00]; // '?'
    f[32] = [0x3E,0x63,0x6F,0x69,0x6F,0x03,0x3E,0x00]; // '@'
    // A-Z
    f[33] = [0x18,0x3C,0x66,0x7E,0x66,0x66,0x66,0x00];
    f[34] = [0x3E,0x66,0x66,0x3E,0x66,0x66,0x3E,0x00];
    f[35] = [0x3C,0x66,0x06,0x06,0x06,0x66,0x3C,0x00];
    f[36] = [0x1E,0x36,0x66,0x66,0x66,0x36,0x1E,0x00];
    f[37] = [0x7E,0x06,0x06,0x3E,0x06,0x06,0x7E,0x00];
    f[38] = [0x7E,0x06,0x06,0x3E,0x06,0x06,0x06,0x00];
    f[39] = [0x3C,0x66,0x06,0x76,0x66,0x66,0x3C,0x00];
    f[40] = [0x66,0x66,0x66,0x7E,0x66,0x66,0x66,0x00];
    f[41] = [0x3C,0x18,0x18,0x18,0x18,0x18,0x3C,0x00];
    f[42] = [0x60,0x60,0x60,0x60,0x60,0x66,0x3C,0x00];
    f[43] = [0x66,0x36,0x1E,0x0E,0x1E,0x36,0x66,0x00];
    f[44] = [0x06,0x06,0x06,0x06,0x06,0x06,0x7E,0x00];
    f[45] = [0x63,0x77,0x7F,0x6B,0x63,0x63,0x63,0x00];
    f[46] = [0x66,0x6E,0x7E,0x76,0x66,0x66,0x66,0x00];
    f[47] = [0x3C,0x66,0x66,0x66,0x66,0x66,0x3C,0x00];
    f[48] = [0x3E,0x66,0x66,0x3E,0x06,0x06,0x06,0x00];
    f[49] = [0x3C,0x66,0x66,0x66,0x6E,0x3C,0x68,0x00];
    f[50] = [0x3E,0x66,0x66,0x3E,0x1E,0x36,0x66,0x00];
    f[51] = [0x3C,0x66,0x06,0x3C,0x60,0x66,0x3C,0x00];
    f[52] = [0x7E,0x18,0x18,0x18,0x18,0x18,0x18,0x00];
    f[53] = [0x66,0x66,0x66,0x66,0x66,0x66,0x3C,0x00];
    f[54] = [0x66,0x66,0x66,0x66,0x66,0x3C,0x18,0x00];
    f[55] = [0x63,0x63,0x63,0x6B,0x7F,0x77,0x63,0x00];
    f[56] = [0x66,0x66,0x3C,0x18,0x3C,0x66,0x66,0x00];
    f[57] = [0x66,0x66,0x66,0x3C,0x18,0x18,0x18,0x00];
    f[58] = [0x7E,0x60,0x30,0x18,0x0C,0x06,0x7E,0x00];
    f[59] = [0x3C,0x0C,0x0C,0x0C,0x0C,0x0C,0x3C,0x00]; // '['
    f[60] = [0x02,0x06,0x0C,0x18,0x30,0x60,0x40,0x00]; // '\'
    f[61] = [0x3C,0x30,0x30,0x30,0x30,0x30,0x3C,0x00]; // ']'
    f[62] = [0x08,0x1C,0x36,0x63,0x00,0x00,0x00,0x00]; // '^'
    f[63] = [0x00,0x00,0x00,0x00,0x00,0x00,0xFF,0x00]; // '_'
    f[64] = [0x18,0x18,0x30,0x00,0x00,0x00,0x00,0x00]; // '`'
    // a-z
    f[65] = [0x00,0x00,0x3C,0x60,0x7C,0x66,0x7C,0x00];
    f[66] = [0x06,0x06,0x3E,0x66,0x66,0x66,0x3E,0x00];
    f[67] = [0x00,0x00,0x3C,0x06,0x06,0x06,0x3C,0x00];
    f[68] = [0x60,0x60,0x7C,0x66,0x66,0x66,0x7C,0x00];
    f[69] = [0x00,0x00,0x3C,0x66,0x7E,0x06,0x3C,0x00];
    f[70] = [0x38,0x0C,0x0C,0x1E,0x0C,0x0C,0x0C,0x00];
    f[71] = [0x00,0x00,0x7C,0x66,0x66,0x7C,0x60,0x3C];
    f[72] = [0x06,0x06,0x3E,0x66,0x66,0x66,0x66,0x00];
    f[73] = [0x18,0x00,0x1C,0x18,0x18,0x18,0x3C,0x00];
    f[74] = [0x30,0x00,0x30,0x30,0x30,0x30,0x36,0x1C];
    f[75] = [0x06,0x06,0x36,0x1E,0x1E,0x36,0x66,0x00];
    f[76] = [0x1C,0x18,0x18,0x18,0x18,0x18,0x3C,0x00];
    f[77] = [0x00,0x00,0x37,0x7F,0x6B,0x63,0x63,0x00];
    f[78] = [0x00,0x00,0x3E,0x66,0x66,0x66,0x66,0x00];
    f[79] = [0x00,0x00,0x3C,0x66,0x66,0x66,0x3C,0x00];
    f[80] = [0x00,0x00,0x3E,0x66,0x66,0x3E,0x06,0x06];
    f[81] = [0x00,0x00,0x7C,0x66,0x66,0x7C,0x60,0x60];
    f[82] = [0x00,0x00,0x3E,0x66,0x06,0x06,0x06,0x00];
    f[83] = [0x00,0x00,0x3C,0x06,0x3C,0x60,0x3C,0x00];
    f[84] = [0x0C,0x0C,0x3E,0x0C,0x0C,0x0C,0x38,0x00];
    f[85] = [0x00,0x00,0x66,0x66,0x66,0x66,0x7C,0x00];
    f[86] = [0x00,0x00,0x66,0x66,0x66,0x3C,0x18,0x00];
    f[87] = [0x00,0x00,0x63,0x6B,0x7F,0x3E,0x36,0x00];
    f[88] = [0x00,0x00,0x66,0x3C,0x18,0x3C,0x66,0x00];
    f[89] = [0x00,0x00,0x66,0x66,0x66,0x7C,0x60,0x3C];
    f[90] = [0x00,0x00,0x7E,0x30,0x18,0x0C,0x7E,0x00];
    f[91] = [0x38,0x0C,0x0C,0x06,0x0C,0x0C,0x38,0x00]; // '{'
    f[92] = [0x18,0x18,0x18,0x00,0x18,0x18,0x18,0x00]; // '|'
    f[93] = [0x0E,0x18,0x18,0x30,0x18,0x18,0x0E,0x00]; // '}'
    f[94] = [0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00]; // '~' placeholder
    f[95] = [0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF]; // DEL / block
    f
}

// ── Colour helpers ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct Rgb {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Rgb {
    pub const BLACK:   Rgb = Rgb { r: 0,   g: 0,   b: 0   };
    pub const WHITE:   Rgb = Rgb { r: 255, g: 255, b: 255 };
    pub const RED:     Rgb = Rgb { r: 220, g: 50,  b: 47  };
    pub const GREEN:   Rgb = Rgb { r: 50,  g: 220, b: 80  };
    pub const BLUE:    Rgb = Rgb { r: 38,  g: 139, b: 210 };
    pub const CYAN:    Rgb = Rgb { r: 42,  g: 161, b: 152 };
    pub const YELLOW:  Rgb = Rgb { r: 181, g: 137, b: 0   };
    pub const MAGENTA: Rgb = Rgb { r: 211, g: 54,  b: 130 };
    pub const ORANGE:  Rgb = Rgb { r: 255, g: 140, b: 0   };
    pub const GRAY:    Rgb = Rgb { r: 50,  g: 50,  b: 60  };
    pub const DARKBG:  Rgb = Rgb { r: 20,  g: 20,  b: 30  };
    pub const ACCENT:  Rgb = Rgb { r: 0,   g: 180, b: 255 };
}

// ── Core writer ───────────────────────────────────────────────────────────────

pub struct FbWriter {
    buf:  &'static mut [u8],
    info: FrameBufferInfo,
    /// Cursor for fmt::Write (text column / row, in 8-pixel cells)
    col:  usize,
    row:  usize,
    fg:   Rgb,
    bg:   Rgb,
}

impl FbWriter {
    pub fn new(fb: &'static mut FrameBuffer) -> Self {
        let info = fb.info().clone();
        let buf  = fb.buffer_mut();
        Self { buf, info, col: 0, row: 0, fg: Rgb::WHITE, bg: Rgb::DARKBG }
    }

    // ── low-level pixel ─────────────────────────────────────────────────────

    #[inline]
    pub fn draw_pixel(&mut self, x: usize, y: usize, c: Rgb) {
        if x >= self.info.width || y >= self.info.height {
            return;
        }
        let off = (y * self.info.stride + x) * self.info.bytes_per_pixel;
        match self.info.pixel_format {
            PixelFormat::Rgb => {
                self.buf[off]     = c.r;
                self.buf[off + 1] = c.g;
                self.buf[off + 2] = c.b;
            }
            PixelFormat::Bgr => {
                self.buf[off]     = c.b;
                self.buf[off + 1] = c.g;
                self.buf[off + 2] = c.r;
            }
            PixelFormat::U8 => {
                self.buf[off] = ((c.r as u16 + c.g as u16 + c.b as u16) / 3) as u8;
            }
            _ => {
                self.buf[off]     = c.b;
                self.buf[off + 1] = c.g;
                self.buf[off + 2] = c.r;
            }
        }
    }

    // ── drawing primitives ──────────────────────────────────────────────────

    pub fn fill_rect(&mut self, x: usize, y: usize, w: usize, h: usize, c: Rgb) {
        for dy in 0..h {
            for dx in 0..w {
                self.draw_pixel(x + dx, y + dy, c);
            }
        }
    }

    pub fn draw_rect_outline(&mut self, x: usize, y: usize, w: usize, h: usize, c: Rgb) {
        for dx in 0..w {
            self.draw_pixel(x + dx, y,         c);
            self.draw_pixel(x + dx, y + h - 1, c);
        }
        for dy in 0..h {
            self.draw_pixel(x,         y + dy, c);
            self.draw_pixel(x + w - 1, y + dy, c);
        }
    }

    pub fn draw_line(&mut self, x0: i32, y0: i32, x1: i32, y1: i32, c: Rgb) {
        // Bresenham
        let (mut x0, mut y0) = (x0, y0);
        let dx = (x1 - x0).abs();
        let dy = -(y1 - y0).abs();
        let sx: i32 = if x0 < x1 { 1 } else { -1 };
        let sy: i32 = if y0 < y1 { 1 } else { -1 };
        let mut err = dx + dy;
        loop {
            self.draw_pixel(x0 as usize, y0 as usize, c);
            if x0 == x1 && y0 == y1 { break; }
            let e2 = 2 * err;
            if e2 >= dy { err += dy; x0 += sx; }
            if e2 <= dx { err += dx; y0 += sy; }
        }
    }

    /// Draw a filled circle using midpoint algorithm.
    pub fn fill_circle(&mut self, cx: usize, cy: usize, r: usize, c: Rgb) {
        let r = r as i32;
        let cx = cx as i32;
        let cy = cy as i32;
        for dy in -r..=r {
            let sq = (r * r - dy * dy).max(0);
            // Integer square root via Newton's method (no_std safe)
            let half = {
                let mut x = sq;
                if x > 0 {
                    let mut y = (x + 1) / 2;
                    while y < x { x = y; y = (x + sq / x) / 2; }
                }
                x
            };
            for dx in -half..=half {
                self.draw_pixel((cx + dx) as usize, (cy + dy) as usize, c);
            }
        }
    }

    // ── 8×8 bitmap glyph rendering ──────────────────────────────────────────

    pub fn draw_char(&mut self, x: usize, y: usize, ch: char, fg: Rgb, bg: Rgb) {
        let idx = ch as usize;
        let glyph = if idx >= 0x20 && idx < 0x80 {
            FONT8X8[idx - 0x20]
        } else {
            FONT8X8[0] // space for unknown
        };
        for row in 0..8usize {
            for col in 0..8usize {
                let bit = (glyph[row] >> col) & 1;  // font is LSB-first (leftmost pixel = bit 0)
                self.draw_pixel(x + col, y + row, if bit != 0 { fg } else { bg });
            }
        }
    }

    pub fn draw_str(&mut self, x: usize, y: usize, s: &str, fg: Rgb, bg: Rgb) {
        let mut cur_x = x;
        for ch in s.chars() {
            self.draw_char(cur_x, y, ch, fg, bg);
            cur_x += 8;
        }
    }

    pub fn draw_str_scaled(
        &mut self, x: usize, y: usize, s: &str, scale: usize, fg: Rgb, bg: Rgb,
    ) {
        let mut cur_x = x;
        for ch in s.chars() {
            let idx = ch as usize;
            let glyph = if idx >= 0x20 && idx < 0x80 {
                FONT8X8[idx - 0x20]
            } else {
                FONT8X8[0]
            };
            for row in 0..8usize {
                for col in 0..8usize {
                    let bit = (glyph[row] >> col) & 1;  // LSB-first
                    let c = if bit != 0 { fg } else { bg };
                    self.fill_rect(cur_x + col * scale, y + row * scale, scale, scale, c);
                }
            }
            cur_x += 8 * scale;
        }
    }

    // ── screen info ─────────────────────────────────────────────────────────

    pub fn width(&self)  -> usize { self.info.width  }
    pub fn height(&self) -> usize { self.info.height }

    pub fn clear(&mut self, c: Rgb) {
        self.fill_rect(0, 0, self.info.width, self.info.height, c);
        self.col = 0;
        self.row = 0;
    }

    pub fn set_colors(&mut self, fg: Rgb, bg: Rgb) {
        self.fg = fg;
        self.bg = bg;
    }

    // ── scrolling text cursor ────────────────────────────────────────────────

    fn cols(&self) -> usize { self.info.width  / 8 }
    fn rows(&self) -> usize { self.info.height / 8 }

    fn scroll_up(&mut self) {
        let row_bytes = 8 * self.info.stride * self.info.bytes_per_pixel;
        let total = self.info.stride * self.info.height * self.info.bytes_per_pixel;
        self.buf.copy_within(row_bytes..total, 0);
        let clear_start = total - row_bytes;
        let bg = self.bg;
        // clear last row
        for i in (clear_start..total).step_by(self.info.bytes_per_pixel) {
            match self.info.pixel_format {
                PixelFormat::Rgb => {
                    self.buf[i]     = bg.r;
                    self.buf[i + 1] = bg.g;
                    self.buf[i + 2] = bg.b;
                }
                _ => {
                    self.buf[i]     = bg.b;
                    self.buf[i + 1] = bg.g;
                    self.buf[i + 2] = bg.r;
                }
            }
        }
    }

    fn advance_cursor(&mut self) {
        self.col += 1;
        if self.col >= self.cols() {
            self.col = 0;
            self.row += 1;
        }
        if self.row >= self.rows() {
            self.scroll_up();
            self.row = self.rows() - 1;
        }
    }

    pub fn write_byte(&mut self, byte: u8) {
        let fg = self.fg;
        let bg = self.bg;
        match byte {
            b'\n' => {
                self.col = 0;
                self.row += 1;
                if self.row >= self.rows() {
                    self.scroll_up();
                    self.row = self.rows() - 1;
                }
            }
            b'\r' => { self.col = 0; }
            b'\x08' => { // backspace
                if self.col > 0 { self.col -= 1; }
                self.draw_char(self.col * 8, self.row * 8, ' ', fg, bg);
            }
            byte => {
                let x = self.col * 8;
                let y = self.row * 8;
                self.draw_char(x, y, byte as char, fg, bg);
                self.advance_cursor();
            }
        }
    }

    pub fn write_str_buf(&mut self, s: &str) {
        for byte in s.bytes() {
            self.write_byte(byte);
        }
    }
}

impl fmt::Write for FbWriter {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.write_str_buf(s);
        Ok(())
    }
}

// ── global writer ─────────────────────────────────────────────────────────────

pub static FB_WRITER: Mutex<Option<FbWriter>> = Mutex::new(None);

pub fn init(fb: &'static mut FrameBuffer) {
    let mut writer = FbWriter::new(fb);
    writer.clear(Rgb::DARKBG);
    *FB_WRITER.lock() = Some(writer);
}

/// Draw the NewDOS boot splash and status bar.
pub fn draw_splash() {
    let mut guard = FB_WRITER.lock();
    if let Some(w) = guard.as_mut() {
        let sw = w.width();
        let sh = w.height();

        // Background gradient-like fill (two-tone)
        w.fill_rect(0, 0, sw, sh / 2, Rgb { r: 15, g: 15, b: 25 });
        w.fill_rect(0, sh / 2, sw, sh / 2, Rgb::DARKBG);

        // Top accent bar
        w.fill_rect(0, 0, sw, 4, Rgb::ACCENT);

        // Large "NewDOS" title (3× scale)
        let title = "NewDOS";
        let tx = (sw / 2).saturating_sub(title.len() * 12);
        let ty = sh / 2 - 50;
        w.draw_str_scaled(tx, ty, title, 3, Rgb::ACCENT, Rgb { r: 15, g: 15, b: 25 });

        // Version subtitle
        let ver = "v0.1.1  |  bootloader 0.11  |  VESA / UEFI GOP";
        let vx = (sw / 2).saturating_sub(ver.len() * 4);
        w.draw_str(vx, ty + 30, ver, Rgb::CYAN, Rgb { r: 15, g: 15, b: 25 });

        // Divider
        w.draw_line(40, (ty + 42) as i32, (sw - 40) as i32, (ty + 42) as i32, Rgb::GRAY);

        // Feature bullets
        let bullets: &[(&str, Rgb)] = &[
            ("VESA VBE / UEFI GOP framebuffer  (HDMI, DisplayPort, VGA)", Rgb::GREEN),
            ("PS/2 keyboard + mouse            (IRQ1 / IRQ12)",            Rgb::CYAN),
            ("In-memory VFS                   (mkdir/touch/write/cat)",   Rgb::YELLOW),
            ("Physical memory map             (from bootloader 0.11)",    Rgb::MAGENTA),
            ("GDT / TSS  (ring 0 / ring 3 segments)",                     Rgb::ORANGE),
        ];
        for (i, (txt, col)) in bullets.iter().enumerate() {
            let by = ty + 55 + i * 14;
            w.draw_str(tx.saturating_sub(8), by, "* ", Rgb::WHITE, Rgb::DARKBG);
            w.draw_str(tx + 8, by, txt, *col, Rgb::DARKBG);
        }

        // Bottom status bar
        let bar_y = sh - 18;
        w.fill_rect(0, bar_y, sw, 18, Rgb { r: 10, g: 10, b: 40 });
        w.fill_rect(0, bar_y, sw, 1,  Rgb::ACCENT);
        w.draw_str(8, bar_y + 5, "NewDOS 0.1.1", Rgb::ACCENT, Rgb { r: 10, g: 10, b: 40 });

        let right = "Video: VESA/GOP  |  Serial: COM1";
        let rx = sw.saturating_sub(right.len() * 8 + 8);
        w.draw_str(rx, bar_y + 5, right, Rgb::GRAY, Rgb { r: 10, g: 10, b: 40 });
    }
}

/// Re-draw the framebuffer demo (called by `suppiere gfx`).
pub fn redraw_demo() {
    let mut guard = FB_WRITER.lock();
    if let Some(w) = guard.as_mut() {
        let sw = w.width();
        let sh = w.height();
        w.clear(Rgb::DARKBG);

        w.fill_rect(0, 0, sw, 4, Rgb::ACCENT);

        // Colour palette swatches
        let swatches: &[(Rgb, &str)] = &[
            (Rgb::RED,     "Red"),
            (Rgb::GREEN,   "Grn"),
            (Rgb::BLUE,    "Blu"),
            (Rgb::CYAN,    "Cyn"),
            (Rgb::YELLOW,  "Yel"),
            (Rgb::MAGENTA, "Mag"),
            (Rgb::ORANGE,  "Org"),
            (Rgb::WHITE,   "Wht"),
        ];
        for (i, (c, lbl)) in swatches.iter().enumerate() {
            let sx = 20 + i * 70;
            w.fill_rect(sx, 20, 60, 40, *c);
            w.draw_rect_outline(sx, 20, 60, 40, Rgb::WHITE);
            w.draw_str(sx + 14, 66, lbl, Rgb::WHITE, Rgb::DARKBG);
        }

        // Gradient bar
        w.draw_str(20, 90, "Horizontal gradient (R->B):", Rgb::WHITE, Rgb::DARKBG);
        for x in 0..sw.min(600) {
            let r = (255 * x / 600) as u8;
            let b = (255 - r) as u8;
            w.fill_rect(20 + x, 105, 1, 20, Rgb { r, g: 60, b });
        }

        // Concentric circles
        w.draw_str(20, 135, "Circles + Bresenham lines:", Rgb::WHITE, Rgb::DARKBG);
        let cx = sw / 2;
        let cy = 200;
        for (r, c) in &[(40usize,Rgb::RED),(30,Rgb::GREEN),(20,Rgb::BLUE),(10,Rgb::YELLOW)] {
            w.fill_circle(cx, cy, *r, *c);
        }
        w.draw_line(20, 230, (sw - 20) as i32, 170, Rgb::CYAN);
        w.draw_line(20, 170, (sw - 20) as i32, 230, Rgb::MAGENTA);

        // Text rendering sample
        w.draw_str(20, 260, "Font rendering: ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz", Rgb::WHITE, Rgb::DARKBG);
        w.draw_str(20, 276, "0123456789  !@#$%^&*()  <>{}[]|  VESA framebuffer output", Rgb::CYAN, Rgb::DARKBG);

        // Digital video out notice
        let notice = "Digital video out: pixels sent via VESA VBE (BIOS) or UEFI GOP -> HDMI/DP/VGA";
        w.draw_str(20, 296, notice, Rgb::GREEN, Rgb::DARKBG);

        // Bottom bar
        let bar_y = sh - 18;
        w.fill_rect(0, bar_y, sw, 18, Rgb { r: 10, g: 10, b: 40 });
        w.fill_rect(0, bar_y, sw, 1, Rgb::ACCENT);
        w.draw_str(8, bar_y + 5, "suppiere gfx -- framebuffer demo", Rgb::ACCENT, Rgb { r: 10, g: 10, b: 40 });
    }
}

/// Draw the interactive boot-mode selection menu.
pub fn draw_boot_menu() {
    let mut guard = FB_WRITER.lock();
    let fb = match guard.as_mut() { Some(f) => f, None => return };

    let sw = fb.width();
    let sh = fb.height();

    // Dark background with faint gradient
    for y in 0..sh {
        let t  = y * 30 / sh.max(1);
        let bg = Rgb { r: (10 + t) as u8, g: (10 + t) as u8, b: (25 + t * 2) as u8 };
        fb.fill_rect(0, y, sw, 1, bg);
    }

    // Accent bar
    fb.fill_rect(0, 0, sw, 4, Rgb::ACCENT);

    // Centre box
    let bw = 520usize;
    let bh = 260usize;
    let bx = (sw / 2).saturating_sub(bw / 2);
    let by = (sh / 2).saturating_sub(bh / 2);

    fb.fill_rect(bx, by, bw, bh, Rgb { r: 16, g: 16, b: 32 });
    fb.draw_rect_outline(bx, by, bw, bh, Rgb::ACCENT);

    // Title
    let title = "NewDOS v0.1.1";
    fb.draw_str_scaled(
        (sw / 2).saturating_sub(title.len() * 12),
        by + 20,
        title, 3, Rgb::ACCENT, Rgb { r: 16, g: 16, b: 32 },
    );

    // Subtitle
    let sub = "Select boot mode:";
    fb.draw_str(
        (sw / 2).saturating_sub(sub.len() * 4),
        by + 74,
        sub, Rgb::WHITE, Rgb { r: 16, g: 16, b: 32 },
    );

    // Divider
    fb.fill_rect(bx + 20, by + 88, bw - 40, 1, Rgb { r: 50, g: 50, b: 80 });

    // Option 1 — GUI
    let opt1_y = by + 100;
    fb.fill_rect(bx + 20, opt1_y, bw - 40, 44, Rgb { r: 22, g: 22, b: 48 });
    fb.draw_rect_outline(bx + 20, opt1_y, bw - 40, 44, Rgb::ACCENT);
    fb.draw_str(bx + 36, opt1_y + 8,  "[1]  Desktop  — GUI, mouse, draggable windows",
                Rgb::ACCENT, Rgb { r: 22, g: 22, b: 48 });
    fb.draw_str(bx + 36, opt1_y + 22, "     Mouse-driven, taskbar, multiple apps",
                Rgb { r: 140, g: 180, b: 220 }, Rgb { r: 22, g: 22, b: 48 });

    // Option 2 — CLI
    let opt2_y = opt1_y + 54;
    fb.fill_rect(bx + 20, opt2_y, bw - 40, 44, Rgb { r: 16, g: 22, b: 16 });
    fb.draw_rect_outline(bx + 20, opt2_y, bw - 40, 44, Rgb::GREEN);
    fb.draw_str(bx + 36, opt2_y + 8,  "[2]  Terminal — CLI, full-screen shell",
                Rgb::GREEN, Rgb { r: 16, g: 22, b: 16 });
    fb.draw_str(bx + 36, opt2_y + 22, "     Classic text console, keyboard-only",
                Rgb { r: 100, g: 200, b: 100 }, Rgb { r: 16, g: 22, b: 16 });

    // Hint
    fb.draw_str(
        (sw / 2).saturating_sub(19 * 4),
        by + bh - 24,
        "Press 1 or 2 to continue",
        Rgb { r: 100, g: 100, b: 140 }, Rgb { r: 16, g: 16, b: 32 },
    );
}

// ── fmt::Write macro shims ────────────────────────────────────────────────────

#[macro_export]
macro_rules! fb_print {
    ($($arg:tt)*) => {
        $crate::framebuffer::_fb_print(format_args!($($arg)*))
    };
}

#[macro_export]
macro_rules! fb_println {
    ()            => ($crate::fb_print!("\n"));
    ($($arg:tt)*) => ($crate::fb_print!("{}\n", format_args!($($arg)*)));
}

#[doc(hidden)]
pub fn _fb_print(args: fmt::Arguments) {
    use core::fmt::Write;
    use x86_64::instructions::interrupts;
    interrupts::without_interrupts(|| {
        if let Some(w) = FB_WRITER.lock().as_mut() {
            let _ = w.write_fmt(args);
        }
    });
}
