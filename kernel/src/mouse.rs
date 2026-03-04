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
static mut BYTES: [i8; 3] = [0; 3];

pub fn init() {
    let mut cmd:  Port<u8> = Port::new(0x64);
    let mut data: Port<u8> = Port::new(0x60);
    unsafe {
        cmd.write(0xA8);
        wait_out(&mut cmd);
        cmd.write(0x20);
        let mut status = data.read();
        status |= 0x02;
        status &= !0x20;
        wait_out(&mut cmd);  cmd.write(0x60);
        wait_out(&mut data); data.write(status);
        mouse_write(&mut cmd, &mut data, 0xF4);
        data.read(); // ack
    }
}

fn wait_out(cmd: &mut Port<u8>) {
    for _ in 0..10_000u32 {
        if unsafe { cmd.read() } & 2 == 0 { return; }
    }
}

fn mouse_write(cmd: &mut Port<u8>, data: &mut Port<u8>, byte: u8) {
    unsafe {
        wait_out(cmd); cmd.write(0xD4);
        wait_out(data); data.write(byte);
    }
}

pub fn handle_packet() {
    let mut data_port: Port<u8> = Port::new(0x60);
    unsafe {
        let byte = data_port.read() as i8;
        match CYCLE {
            0 => { BYTES[0] = byte; CYCLE = 1; }
            1 => { BYTES[1] = byte; CYCLE = 2; }
            2 => {
                BYTES[2] = byte;
                CYCLE    = 0;
                let flags = BYTES[0] as u8;
                if flags & 0x08 == 0 { return; } // sanity
                if flags & 0xC0 != 0 { return; } // overflow
                let dx   = BYTES[1] as i32;
                let dy   = -(BYTES[2] as i32);   // Y inverted
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
