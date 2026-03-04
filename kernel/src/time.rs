//! Minimal time support — reads RTC, applies tz offset, formats strings.

use x86_64::instructions::port::Port;

fn cmos_read(reg: u8) -> u8 {
    unsafe {
        let mut addr: Port<u8> = Port::new(0x70);
        let mut data: Port<u8> = Port::new(0x71);
        addr.write(reg);
        data.read()
    }
}

fn bcd_to_bin(v: u8) -> u8 { (v & 0x0F) + ((v >> 4) * 10) }

pub struct RtcTime { pub h: u8, pub m: u8, pub s: u8, pub day: u8, pub mon: u8, pub yr: u16 }

pub fn read_rtc() -> RtcTime {
    // Wait for update-in-progress flag to clear
    while cmos_read(0x0A) & 0x80 != 0 {}
    let s   = bcd_to_bin(cmos_read(0x00));
    let m   = bcd_to_bin(cmos_read(0x02));
    let h   = bcd_to_bin(cmos_read(0x04));
    let day = bcd_to_bin(cmos_read(0x07));
    let mon = bcd_to_bin(cmos_read(0x08));
    let yr  = bcd_to_bin(cmos_read(0x09)) as u16 + 2000;
    RtcTime { h, m, s, day, mon, yr }
}

static mut TZ_OFFSET: i8 = 0;

pub fn set_timezone(offset: i8) {
    unsafe { TZ_OFFSET = offset; }
}

pub fn get_timezone() -> i8 { unsafe { TZ_OFFSET } }

/// Returns (gmt_string, local_string)  e.g. ("14:32:05 UTC", "16:32:05 UTC+2")
pub fn formatted_times() -> ([u8; 32], [u8; 32]) {
    let t = read_rtc();
    let tz = unsafe { TZ_OFFSET };

    let mut gmt_h = t.h as i16 % 24;
    if gmt_h < 0 { gmt_h += 24; }

    let mut local_h = (t.h as i16 + tz as i16) % 24;
    if local_h < 0 { local_h += 24; }

    let mut gmt = [0u8; 32];
    let mut loc = [0u8; 32];

    write_time(&mut gmt, gmt_h as u8, t.m, t.s, 0);
    write_time(&mut loc, local_h as u8, t.m, t.s, tz);
    (gmt, loc)
}

fn write_time(buf: &mut [u8; 32], h: u8, m: u8, s: u8, tz: i8) {
    // HH:MM:SS UTC[+/-N]
    buf[0] = b'0' + h / 10;
    buf[1] = b'0' + h % 10;
    buf[2] = b':';
    buf[3] = b'0' + m / 10;
    buf[4] = b'0' + m % 10;
    buf[5] = b':';
    buf[6] = b'0' + s / 10;
    buf[7] = b'0' + s % 10;
    buf[8] = b' ';
    buf[9] = b'U'; buf[10] = b'T'; buf[11] = b'C';
    if tz == 0 {
        buf[12] = 0;
    } else if tz > 0 {
        buf[12] = b'+';
        buf[13] = b'0' + tz as u8 / 10;
        buf[14] = b'0' + tz as u8 % 10;
        buf[15] = 0;
    } else {
        let abs = (-tz) as u8;
        buf[12] = b'-';
        buf[13] = b'0' + abs / 10;
        buf[14] = b'0' + abs % 10;
        buf[15] = 0;
    }
}

pub fn time_str(buf: &[u8; 32]) -> &str {
    let len = buf.iter().position(|&b| b == 0).unwrap_or(32);
    core::str::from_utf8(&buf[..len]).unwrap_or("??:??:??")
}
