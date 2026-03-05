//! Persistent VFS storage on the boot disk via ATA PIO.
//!
//! The OS data area lives at LBA 8192 (4 MB into the disk), safely past the
//! 2.5 MB kernel+bootloader region.  The disk images are truncated to 10 MB
//! at build time so this area always exists.
//!
//! ── On-disk layout ──────────────────────────────────────────────────────────
//!
//!  LBA 8192   — header sector (512 bytes)
//!    [0..4]    magic       = b"NDOS"
//!    [4..8]    payload_len : u32 LE   (bytes used in LBA 8193+)
//!
//!  LBA 8193+  — payload (up to 4095 sectors = ~2 MB)
//!
//! ── Payload record format ───────────────────────────────────────────────────
//!
//!  u8   type   : 0 = end-of-data  1 = directory  2 = file
//!  u8   plen   : path length in bytes (≤ 255)
//!  …    path   : absolute path without null terminator (e.g. "/docs/notes.txt")
//!  if type == 2:
//!    u16  dlen   : content length LE (≤ 65535)
//!    …    data   : raw file bytes

use crate::{ata, vfs};
use alloc::{string::String, vec, vec::Vec};

const DATA_LBA:    u32   = 8192;
const MAX_SECTORS: usize = 4095;   // header is sector 0, payload uses 1..4095
const MAX_PAYLOAD: usize = MAX_SECTORS * 512;

// ── Public API ─────────────────────────────────────────────────────────────────

/// Serialise `fs` to the data area.  Silently returns if no ATA drive is
/// present, or if the serialised payload exceeds ~2 MB.
pub fn save(fs: &vfs::FileSystem) {
    let payload = build_payload(fs);
    if payload.len() > MAX_PAYLOAD { return; }

    let psecs      = (payload.len() + 511) / 512;
    let total_secs = 1 + psecs;
    let mut buf    = vec![0u8; total_secs * 512];

    // Header
    buf[0..4].copy_from_slice(b"NDOS");
    buf[4..8].copy_from_slice(&(payload.len() as u32).to_le_bytes());

    // Payload
    buf[512..512 + payload.len()].copy_from_slice(&payload);

    write_chunks(DATA_LBA, &buf);
}

/// Deserialise the VFS from the data area.
/// Returns `None` if no ATA drive, bad magic, or no data written yet.
pub fn load() -> Option<vfs::FileSystem> {
    let mut hdr = [0u8; 512];
    if !ata::read_sectors(DATA_LBA, 1, &mut hdr) { return None; }
    if &hdr[0..4] != b"NDOS"                     { return None; }

    let plen = u32::from_le_bytes([hdr[4], hdr[5], hdr[6], hdr[7]]) as usize;
    if plen == 0 || plen > MAX_PAYLOAD { return None; }

    let psecs  = (plen + 511) / 512;
    let mut pb = vec![0u8; psecs * 512];
    if !read_chunks(DATA_LBA + 1, &mut pb) { return None; }

    Some(deserialize(&pb[..plen]))
}

// ── Serialise ─────────────────────────────────────────────────────────────────

fn build_payload(fs: &vfs::FileSystem) -> Vec<u8> {
    let mut out = Vec::new();
    for (path, is_dir, data) in fs.walk_all() {
        let pb   = path.as_bytes();
        let plen = pb.len().min(255) as u8;
        if is_dir {
            out.push(1u8);
            out.push(plen);
            out.extend_from_slice(&pb[..plen as usize]);
        } else {
            out.push(2u8);
            out.push(plen);
            out.extend_from_slice(&pb[..plen as usize]);
            let dlen = data.len().min(65535) as u16;
            out.extend_from_slice(&dlen.to_le_bytes());
            out.extend_from_slice(&data[..dlen as usize]);
        }
    }
    out.push(0); // end-of-data marker
    out
}

// ── Deserialise ───────────────────────────────────────────────────────────────

fn deserialize(data: &[u8]) -> vfs::FileSystem {
    let mut fs  = vfs::FileSystem::new();
    let mut pos = 0usize;

    while pos < data.len() {
        let rec_type = data[pos]; pos += 1;
        if rec_type == 0 { break; }
        if pos >= data.len() { break; }

        let plen = data[pos] as usize; pos += 1;
        if pos + plen > data.len() { break; }
        let path = match core::str::from_utf8(&data[pos..pos + plen]) {
            Ok(s) => s,
            Err(_) => { pos += plen; continue; }
        };
        pos += plen;

        match rec_type {
            1 => { let _ = fs.mkdir(path); }
            2 => {
                if pos + 2 > data.len() { break; }
                let dlen = u16::from_le_bytes([data[pos], data[pos + 1]]) as usize;
                pos += 2;
                if pos + dlen > data.len() { break; }
                let content = core::str::from_utf8(&data[pos..pos + dlen]).unwrap_or("");
                pos += dlen;
                ensure_parents(&mut fs, path);
                let _ = fs.touch(path);
                let _ = fs.write_file(path, content);
            }
            _ => break,
        }
    }
    fs
}

/// Create all ancestor directories of `file_path` (everything but the last
/// path component) so that a subsequent `touch` will succeed.
fn ensure_parents(fs: &mut vfs::FileSystem, file_path: &str) {
    let parts: Vec<&str> = file_path.split('/').filter(|s| !s.is_empty()).collect();
    if parts.len() <= 1 { return; }
    let mut cur = String::new();
    for part in &parts[..parts.len() - 1] {
        cur.push('/');
        cur.push_str(part);
        if !fs.exists(&cur) { let _ = fs.mkdir(&cur); }
    }
}

// ── I/O helpers (max 255 sectors per ATA command) ────────────────────────────

fn write_chunks(start_lba: u32, buf: &[u8]) {
    let total  = buf.len() / 512;
    let mut lba    = start_lba;
    let mut offset = 0usize;
    while offset < buf.len() {
        let remaining = (total - offset / 512).min(255);
        if remaining == 0 { break; }
        let count = remaining as u8;
        ata::write_sectors(lba, count, &buf[offset..offset + count as usize * 512]);
        lba    += count as u32;
        offset += count as usize * 512;
    }
}

fn read_chunks(start_lba: u32, buf: &mut [u8]) -> bool {
    let total  = buf.len() / 512;
    let mut lba    = start_lba;
    let mut offset = 0usize;
    while offset < buf.len() {
        let remaining = (total - offset / 512).min(255);
        if remaining == 0 { break; }
        let count = remaining as u8;
        if !ata::read_sectors(lba, count, &mut buf[offset..offset + count as usize * 512]) {
            return false;
        }
        lba    += count as u32;
        offset += count as usize * 512;
    }
    true
}
