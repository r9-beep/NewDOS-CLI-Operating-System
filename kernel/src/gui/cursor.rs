//! Arrow mouse cursor — drawn on top of every frame.

use crate::framebuffer::{FbWriter, Rgb};

pub const W: usize = 12;
pub const H: usize = 19;

// 0 = transparent, 1 = black outline, 2 = white fill
const SPRITE: [[u8; W]; H] = [
    [1,0,0,0,0,0,0,0,0,0,0,0],
    [1,1,0,0,0,0,0,0,0,0,0,0],
    [1,2,1,0,0,0,0,0,0,0,0,0],
    [1,2,2,1,0,0,0,0,0,0,0,0],
    [1,2,2,2,1,0,0,0,0,0,0,0],
    [1,2,2,2,2,1,0,0,0,0,0,0],
    [1,2,2,2,2,2,1,0,0,0,0,0],
    [1,2,2,2,2,2,2,1,0,0,0,0],
    [1,2,2,2,2,2,2,2,1,0,0,0],
    [1,2,2,2,2,2,1,1,1,0,0,0],
    [1,2,2,2,1,2,2,1,0,0,0,0],
    [1,2,2,1,0,1,2,2,1,0,0,0],
    [1,2,1,0,0,0,1,2,2,1,0,0],
    [1,1,0,0,0,0,0,1,2,1,0,0],
    [1,0,0,0,0,0,0,0,1,1,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0],
];

pub fn draw(fb: &mut FbWriter, x: i32, y: i32) {
    for row in 0..H {
        for col in 0..W {
            let px = x + col as i32;
            let py = y + row as i32;
            if px < 0 || py < 0 { continue; }
            match SPRITE[row][col] {
                1 => fb.draw_pixel(px as usize, py as usize, Rgb::BLACK),
                2 => fb.draw_pixel(px as usize, py as usize, Rgb::WHITE),
                _ => {}
            }
        }
    }
}
