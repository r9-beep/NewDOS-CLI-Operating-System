use pc_keyboard::{layouts, DecodedKey, HandleControl, KeyCode, Keyboard, ScancodeSet1};
use spin::Mutex;
use x86_64::instructions::port::Port;

// ── SpecialKey ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpecialKey { F1, F2, F3, F9, F10, Escape, Up, Down, Left, Right, Delete }

// ── Input queues (single-slot; last wins) ─────────────────────────────────────

static CHAR_Q:    Mutex<Option<char>>       = Mutex::new(None);
static SPECIAL_Q: Mutex<Option<SpecialKey>> = Mutex::new(None);

pub fn pop_char()    -> Option<char>       { CHAR_Q.lock().take() }
pub fn pop_special() -> Option<SpecialKey> { SPECIAL_Q.lock().take() }

// ── PS/2 decoder ──────────────────────────────────────────────────────────────

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
                DecodedKey::Unicode(c) => { *CHAR_Q.lock() = Some(c); }
                DecodedKey::RawKey(k)  => handle_raw(k),
            }
        }
    }
}

fn handle_raw(k: KeyCode) {
    let sk = match k {
        KeyCode::Escape     => Some(SpecialKey::Escape),
        KeyCode::ArrowUp    => Some(SpecialKey::Up),
        KeyCode::ArrowDown  => Some(SpecialKey::Down),
        KeyCode::ArrowLeft  => Some(SpecialKey::Left),
        KeyCode::ArrowRight => Some(SpecialKey::Right),
        KeyCode::Delete     => Some(SpecialKey::Delete),
        KeyCode::F1         => Some(SpecialKey::F1),
        KeyCode::F2         => Some(SpecialKey::F2),
        KeyCode::F3         => Some(SpecialKey::F3),
        KeyCode::F9         => Some(SpecialKey::F9),
        KeyCode::F10        => Some(SpecialKey::F10),
        _ => None,
    };
    if let Some(sk) = sk { *SPECIAL_Q.lock() = Some(sk); }
}
