use spin::Mutex;
use x86_64::instructions::port::Port;

// ── Global mouse state ────────────────────────────────────────────────────────

pub struct MouseState {
    pub x:       i32,
    pub y:       i32,
    pub buttons: u8,
    pub updated: bool,
    width:       i32,
    height:      i32,
}

pub static STATE: Mutex<MouseState> = Mutex::new(MouseState {
    x: 512, y: 384, buttons: 0, updated: false, width: 1024, height: 768,
});

pub fn set_bounds(w: usize, h: usize) {
    let mut s = STATE.lock();
    s.width  = w as i32;
    s.height = h as i32;
    s.x      = w as i32 / 2;
    s.y      = h as i32 / 2;
}

// ── PS/2 packet assembly ──────────────────────────────────────────────────────

static mut CYCLE: u8      = 0;
static mut BYTES: [u8; 3] = [0; 3];  // u8 — sign handled via flags bits 4/5

pub fn init() {
    let mut cmd:  Port<u8> = Port::new(0x64);
    let mut data: Port<u8> = Port::new(0x60);
    unsafe {
        // Enable aux (mouse) port
        wait_wr(&mut cmd); cmd.write(0xA8);

        // Read current controller config byte
        wait_wr(&mut cmd); cmd.write(0x20);
        wait_rd(&mut cmd);                      // ← wait for data to arrive
        let mut cfg = data.read();
        cfg |= 0x03;                            // enable IRQ1 (keyboard) and IRQ12 (mouse)
        cfg &= !0x20;                           // enable mouse clock (clear "disable" bit)

        // Write updated config back
        wait_wr(&mut cmd); cmd.write(0x60);
        wait_wr(&mut cmd); data.write(cfg);     // ← status at cmd, not data

        // Tell mouse to start streaming
        mouse_write(&mut cmd, &mut data, 0xF4);
        wait_rd(&mut cmd); data.read();         // read ack
    }
}

/// Wait until the controller output buffer has data (status bit 0 = 1).
fn wait_rd(cmd: &mut Port<u8>) {
    for _ in 0..100_000u32 {
        if unsafe { cmd.read() } & 0x01 != 0 { return; }
    }
}

/// Wait until the controller input buffer is empty (status bit 1 = 0).
fn wait_wr(cmd: &mut Port<u8>) {
    for _ in 0..100_000u32 {
        if unsafe { cmd.read() } & 0x02 == 0 { return; }
    }
}

/// Route a byte to the PS/2 mouse via the keyboard controller.
fn mouse_write(cmd: &mut Port<u8>, data: &mut Port<u8>, byte: u8) {
    unsafe {
        wait_wr(cmd); cmd.write(0xD4);  // tell KBC to send next byte to mouse
        wait_wr(cmd); data.write(byte); // status reg is at cmd (0x64), not data (0x60)
    }
}

pub fn handle_packet() {
    let mut data_port: Port<u8> = Port::new(0x60);
    unsafe {
        let byte = data_port.read();
        match CYCLE {
            0 => { BYTES[0] = byte; CYCLE = 1; }
            1 => { BYTES[1] = byte; CYCLE = 2; }
            2 => {
                BYTES[2] = byte;
                CYCLE    = 0;
                let flags = BYTES[0];
                if flags & 0x08 == 0 { return; }   // sanity — bit 3 must be set
                if flags & 0xC0 != 0 { return; }   // overflow — discard

                // PS/2 sends a 9-bit signed value: bits 4/5 of flags are the sign bits.
                let dx = if flags & 0x10 != 0 { (BYTES[1] as i32) - 256 }
                         else                 {  BYTES[1] as i32         };
                let dy = if flags & 0x20 != 0 { (BYTES[2] as i32) - 256 }
                         else                 {  BYTES[2] as i32         };
                let dy = -dy;   // framebuffer Y grows downward, mouse Y grows upward

                let btns = flags & 0x07;
                let mut s = STATE.lock();
                s.x       = (s.x + dx).clamp(0, s.width  - 1);
                s.y       = (s.y + dy).clamp(0, s.height - 1);
                s.buttons = btns;
                s.updated = true;
            }
            _ => { CYCLE = 0; }
        }
    }
}
