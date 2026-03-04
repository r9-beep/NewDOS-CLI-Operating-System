use x86_64::instructions::port::Port;

static mut MOUSE_CYCLE: u8  = 0;
static mut MOUSE_BYTE:  [i8; 3] = [0; 3];

pub fn init() {
    let mut cmd_port:  Port<u8> = Port::new(0x64);
    let mut data_port: Port<u8> = Port::new(0x60);

    unsafe {
        // Enable auxiliary (mouse) port
        cmd_port.write(0xA8);
        wait_write(&mut cmd_port);
        // Enable IRQ12
        cmd_port.write(0x20);
        let mut status = data_port.read();
        status |= 0x02;
        status &= !0x20;
        wait_write(&mut cmd_port);
        cmd_port.write(0x60);
        wait_write(&mut data_port);
        data_port.write(status);
        // Send mouse enable streaming command
        mouse_write(&mut cmd_port, &mut data_port, 0xF4);
        data_port.read(); // ack
    }
}

fn wait_write(cmd: &mut Port<u8>) {
    for _ in 0..10_000 {
        let status: u8 = unsafe { cmd.read() };
        if status & 2 == 0 { return; }
    }
}

fn mouse_write(cmd: &mut Port<u8>, data: &mut Port<u8>, byte: u8) {
    unsafe {
        wait_write(cmd);
        cmd.write(0xD4);
        wait_write(cmd);
        data.write(byte);
    }
}

pub fn handle_packet() {
    let mut data_port: Port<u8> = Port::new(0x60);
    unsafe {
        let byte = data_port.read() as i8;
        match MOUSE_CYCLE {
            0 => { MOUSE_BYTE[0] = byte; MOUSE_CYCLE = 1; }
            1 => { MOUSE_BYTE[1] = byte; MOUSE_CYCLE = 2; }
            2 => {
                MOUSE_BYTE[2] = byte;
                MOUSE_CYCLE   = 0;
                // Could process dx/dy here
                let _dx = MOUSE_BYTE[1];
                let _dy = MOUSE_BYTE[2];
                let _buttons = MOUSE_BYTE[0];
            }
            _ => { MOUSE_CYCLE = 0; }
        }
    }
}
