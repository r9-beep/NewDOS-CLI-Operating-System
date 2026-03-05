//! ATA PIO 28-bit LBA driver — primary IDE channel (ports 0x1F0–0x1F7).
//!
//! Works with IDE/PATA drives and AHCI controllers in IDE-compatibility mode.
//! Silently no-ops when no drive is present (floating bus → status == 0xFF).

use x86_64::instructions::port::Port;

// ── Port addresses ─────────────────────────────────────────────────────────────

const DATA:     u16 = 0x1F0;   // 16-bit data register
const SECCOUNT: u16 = 0x1F2;   // sector count
const LBA0:     u16 = 0x1F3;   // LBA bits  0– 7
const LBA1:     u16 = 0x1F4;   // LBA bits  8–15
const LBA2:     u16 = 0x1F5;   // LBA bits 16–23
const DRIVE:    u16 = 0x1F6;   // drive / head (LBA bits 24–27, LBA flag, drive select)
const CMD:      u16 = 0x1F7;   // write = command;  read = status
const ALT_STAT: u16 = 0x3F6;   // alternate status (read-only here)

// ── ATA commands ───────────────────────────────────────────────────────────────

const CMD_READ:  u8 = 0x20;
const CMD_WRITE: u8 = 0x30;
const CMD_FLUSH: u8 = 0xE7;   // flush write cache to media

// ── Status bits ───────────────────────────────────────────────────────────────

const ST_ERR: u8 = 0x01;
const ST_DRQ: u8 = 0x08;
const ST_BSY: u8 = 0x80;

// ── Internal helpers ──────────────────────────────────────────────────────────

#[inline]
fn rd_status() -> u8 { unsafe { Port::<u8>::new(CMD).read() } }

/// 400 ns delay via four reads of the alternate-status register (~100 ns each).
fn delay400ns() {
    for _ in 0..4 { unsafe { Port::<u8>::new(ALT_STAT).read(); } }
}

fn wait_not_busy() -> bool {
    for _ in 0..500_000u32 {
        let s = rd_status();
        if s == 0xFF { return false; }   // floating bus → no drive attached
        if s & ST_BSY == 0 { return true; }
    }
    false
}

fn wait_drq() -> bool {
    for _ in 0..500_000u32 {
        let s = rd_status();
        if s & ST_ERR != 0 { return false; }
        if s & ST_DRQ != 0 { return true; }
    }
    false
}

/// Select master drive and set up LBA28 address + sector count.
fn setup_lba28(lba: u32, count: u8) {
    unsafe {
        // 0xE0 = 1110_xxxx: LBA mode (bit 6), master drive (bit 4 = 0)
        Port::<u8>::new(DRIVE).write(0xE0 | ((lba >> 24) & 0x0F) as u8);
        delay400ns();
        Port::<u8>::new(SECCOUNT).write(count);
        Port::<u8>::new(LBA0).write((lba        & 0xFF) as u8);
        Port::<u8>::new(LBA1).write(((lba >> 8) & 0xFF) as u8);
        Port::<u8>::new(LBA2).write(((lba >>16) & 0xFF) as u8);
    }
}

// ── Public API ─────────────────────────────────────────────────────────────────

/// Read `count` sectors (512 B each) starting at LBA `lba` into `buf`.
/// `buf` must be at least `count as usize * 512` bytes.
/// Returns `false` if no drive is present or a hardware error occurs.
pub fn read_sectors(lba: u32, count: u8, buf: &mut [u8]) -> bool {
    if count == 0 || buf.len() < count as usize * 512 { return false; }
    if !wait_not_busy() { return false; }
    setup_lba28(lba, count);
    unsafe { Port::<u8>::new(CMD).write(CMD_READ); }
    let mut dp: Port<u16> = Port::new(DATA);
    for i in 0..count as usize {
        if !wait_not_busy() { return false; }
        if !wait_drq()      { return false; }
        let base = i * 512;
        for w in 0..256usize {
            let word = unsafe { dp.read() };
            buf[base + w * 2]     = (word & 0xFF) as u8;
            buf[base + w * 2 + 1] = (word >>   8) as u8;
        }
    }
    true
}

/// Write `count` sectors (512 B each) starting at LBA `lba` from `buf`.
/// `buf` must be at least `count as usize * 512` bytes.
/// Returns `false` if no drive is present or a hardware error occurs.
pub fn write_sectors(lba: u32, count: u8, buf: &[u8]) -> bool {
    if count == 0 || buf.len() < count as usize * 512 { return false; }
    if !wait_not_busy() { return false; }
    setup_lba28(lba, count);
    unsafe { Port::<u8>::new(CMD).write(CMD_WRITE); }
    let mut dp: Port<u16> = Port::new(DATA);
    for i in 0..count as usize {
        if !wait_not_busy() { return false; }
        if !wait_drq()      { return false; }
        let base = i * 512;
        for w in 0..256usize {
            let lo = buf[base + w * 2]     as u16;
            let hi = buf[base + w * 2 + 1] as u16;
            unsafe { dp.write(lo | (hi << 8)); }
        }
    }
    // Flush write-cache so data reaches media
    if wait_not_busy() {
        unsafe { Port::<u8>::new(CMD).write(CMD_FLUSH); }
        wait_not_busy();
    }
    true
}
