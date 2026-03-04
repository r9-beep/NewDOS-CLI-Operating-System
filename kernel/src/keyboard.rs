use crate::shell;
use pc_keyboard::{layouts, DecodedKey, HandleControl, KeyCode, Keyboard, ScancodeSet1};
use spin::Mutex;
use x86_64::instructions::port::Port;

static KEYBOARD: Mutex<Keyboard<layouts::Us104Key, ScancodeSet1>> = Mutex::new(
    Keyboard::new(ScancodeSet1::new(), layouts::Us104Key, HandleControl::Ignore),
);

pub fn handle_scancode() {
    let mut port = Port::<u8>::new(0x60);
    let scancode: u8 = unsafe { port.read() };

    let mut kb = KEYBOARD.lock();
    if let Ok(Some(key_event)) = kb.add_byte(scancode) {
        if let Some(key) = kb.process_keyevent(key_event) {
            match key {
                DecodedKey::Unicode(c) => shell::push_char(c),
                DecodedKey::RawKey(k)  => handle_raw(k),
            }
        }
    }
}

fn handle_raw(k: KeyCode) {
    match k {
        KeyCode::ArrowUp    => shell::push_char('\x1b'), // ESC-based nav stub
        KeyCode::ArrowDown  => {}
        KeyCode::ArrowLeft  => {}
        KeyCode::ArrowRight => {}
        KeyCode::F1  => shell::push_special(shell::SpecialKey::F1),
        KeyCode::F2  => shell::push_special(shell::SpecialKey::F2),
        KeyCode::F3  => shell::push_special(shell::SpecialKey::F3),
        KeyCode::F9  => shell::push_special(shell::SpecialKey::F9),
        KeyCode::F10 => shell::push_special(shell::SpecialKey::F10),
        _ => {}
    }
}
