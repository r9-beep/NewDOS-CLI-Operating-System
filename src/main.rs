#![no_std]
#![no_main]
#![feature(abi_x86_interrupt)]
#![feature(alloc_error_handler)]

use bootloader::{entry_point, BootInfo};
use core::fmt::Write;
use core::panic::PanicInfo;
use linked_list_allocator::LockedHeap;

mod vga {
    use core::fmt;
    use lazy_static::lazy_static;
    use spin::Mutex;

    #[repr(transparent)]
    struct Volatile<T> {
        value: T,
    }

    impl<T: Copy> Volatile<T> {
        fn read(&self) -> T {
            unsafe { core::ptr::read_volatile(&self.value) }
        }

        fn write(&mut self, value: T) {
            unsafe { core::ptr::write_volatile(&mut self.value, value) }
        }
    }

    #[allow(dead_code)]
    #[derive(Clone, Copy)]
    #[repr(u8)]
    pub enum Color {
        Black = 0,
        Blue = 1,
        Green = 2,
        Cyan = 3,
        Red = 4,
        Magenta = 5,
        Brown = 6,
        LightGray = 7,
        DarkGray = 8,
        LightBlue = 9,
        LightGreen = 10,
        LightCyan = 11,
        LightRed = 12,
        Pink = 13,
        Yellow = 14,
        White = 15,
    }

    #[derive(Clone, Copy)]
    #[repr(transparent)]
    struct ColorCode(u8);

    impl ColorCode {
        fn new(foreground: Color, background: Color) -> Self {
            Self((background as u8) << 4 | (foreground as u8))
        }
    }

    #[derive(Clone, Copy)]
    #[repr(C)]
    struct ScreenChar {
        ascii_character: u8,
        color_code: ColorCode,
    }

    const BUFFER_HEIGHT: usize = 25;
    const BUFFER_WIDTH: usize = 80;

    #[repr(transparent)]
    struct Buffer {
        chars: [[Volatile<ScreenChar>; BUFFER_WIDTH]; BUFFER_HEIGHT],
    }

    pub struct Writer {
        column_position: usize,
        color_code: ColorCode,
        buffer: &'static mut Buffer,
    }

    impl Writer {
        pub fn new() -> Self {
            Self {
                column_position: 0,
                color_code: ColorCode::new(Color::LightGreen, Color::Black),
                buffer: unsafe { &mut *(0xb8000 as *mut Buffer) },
            }
        }

        pub fn set_color(&mut self, foreground: Color, background: Color) {
            self.color_code = ColorCode::new(foreground, background);
        }

        pub fn write_byte(&mut self, byte: u8) {
            match byte {
                b'\n' => self.new_line(),
                byte => {
                    if self.column_position >= BUFFER_WIDTH {
                        self.new_line();
                    }

                    let row = BUFFER_HEIGHT - 1;
                    let col = self.column_position;

                    self.buffer.chars[row][col].write(ScreenChar {
                        ascii_character: byte,
                        color_code: self.color_code,
                    });
                    self.column_position += 1;
                }
            }
        }

        fn new_line(&mut self) {
            for row in 1..BUFFER_HEIGHT {
                for col in 0..BUFFER_WIDTH {
                    let character = self.buffer.chars[row][col].read();
                    self.buffer.chars[row - 1][col].write(character);
                }
            }
            self.clear_row(BUFFER_HEIGHT - 1);
            self.column_position = 0;
        }

        fn clear_row(&mut self, row: usize) {
            let blank = ScreenChar {
                ascii_character: b' ',
                color_code: self.color_code,
            };
            for col in 0..BUFFER_WIDTH {
                self.buffer.chars[row][col].write(blank);
            }
        }

        pub fn write_at(&mut self, row: usize, col: usize, message: &str) {
            let mut col = col;
            for byte in message.bytes() {
                if col >= BUFFER_WIDTH || row >= BUFFER_HEIGHT {
                    break;
                }
                self.buffer.chars[row][col].write(ScreenChar {
                    ascii_character: byte,
                    color_code: self.color_code,
                });
                col += 1;
            }
        }

        pub fn write_byte_at(&mut self, row: usize, col: usize, byte: u8) {
            if row >= BUFFER_HEIGHT || col >= BUFFER_WIDTH {
                return;
            }
            self.buffer.chars[row][col].write(ScreenChar {
                ascii_character: byte,
                color_code: self.color_code,
            });
        }
    }

    impl fmt::Write for Writer {
        fn write_str(&mut self, s: &str) -> fmt::Result {
            for byte in s.bytes() {
                match byte {
                    0x20..=0x7e | b'\n' => self.write_byte(byte),
                    _ => self.write_byte(0xfe),
                }
            }
            Ok(())
        }
    }

    lazy_static! {
        pub static ref WRITER: Mutex<Writer> = Mutex::new(Writer::new());
    }

    pub fn write_status(message: &str) {
        let mut writer = WRITER.lock();
        writer.set_color(Color::LightCyan, Color::Black);
        writer.write_at(0, 0, "                                                                                ");
        writer.write_at(0, 0, message);
        writer.set_color(Color::LightGreen, Color::Black);
    }

    pub fn write_text(message: &str) {
        let mut writer = WRITER.lock();
        let _ = fmt::Write::write_str(&mut *writer, message);
    }

    pub fn write_char(ch: char) {
        let mut writer = WRITER.lock();
        let _ = fmt::Write::write_char(&mut *writer, ch);
    }

    pub fn clear_screen() {
        let mut writer = WRITER.lock();
        writer.set_color(Color::LightGreen, Color::Black);
        for _ in 0..BUFFER_HEIGHT {
            writer.new_line();
        }
    }

    pub fn draw_box(
        row: usize,
        col: usize,
        width: usize,
        height: usize,
        foreground: Color,
        background: Color,
        title: Option<&str>,
    ) {
        if width < 2 || height < 2 {
            return;
        }
        let mut writer = WRITER.lock();
        writer.set_color(foreground, background);

        writer.write_byte_at(row, col, b'+');
        writer.write_byte_at(row, col + width - 1, b'+');
        writer.write_byte_at(row + height - 1, col, b'+');
        writer.write_byte_at(row + height - 1, col + width - 1, b'+');

        for x in (col + 1)..(col + width - 1) {
            writer.write_byte_at(row, x, b'-');
            writer.write_byte_at(row + height - 1, x, b'-');
        }

        for y in (row + 1)..(row + height - 1) {
            writer.write_byte_at(y, col, b'|');
            writer.write_byte_at(y, col + width - 1, b'|');
        }

        if let Some(title) = title {
            writer.write_at(row, col + 2, title);
        }

        writer.set_color(Color::LightGreen, Color::Black);
    }

    pub fn fill_region(
        row: usize,
        col: usize,
        width: usize,
        height: usize,
        foreground: Color,
        background: Color,
        fill: u8,
    ) {
        let mut writer = WRITER.lock();
        writer.set_color(foreground, background);
        for y in row..(row + height) {
            for x in col..(col + width) {
                writer.write_byte_at(y, x, fill);
            }
        }
        writer.set_color(Color::LightGreen, Color::Black);
    }
}

mod gfx {
    pub fn unsupported_message() {
        crate::vga::write_status("Framebuffer unavailable on bootloader v0.9");
    }
}

mod gdt {
    use lazy_static::lazy_static;
    use x86_64::instructions::segmentation::{Segment, CS};
    use x86_64::instructions::tables::load_tss;
    use x86_64::structures::gdt::{Descriptor, GlobalDescriptorTable, SegmentSelector};
    use x86_64::structures::tss::TaskStateSegment;
    use x86_64::VirtAddr;

    const STACK_SIZE: usize = 4096 * 5;

    #[repr(align(16))]
    struct Stack([u8; STACK_SIZE]);

    static mut KERNEL_STACK: Stack = Stack([0; STACK_SIZE]);

    pub struct Selectors {
        pub kernel_code: SegmentSelector,
        pub kernel_data: SegmentSelector,
        pub user_code: SegmentSelector,
        pub user_data: SegmentSelector,
        pub tss: SegmentSelector,
    }

    lazy_static! {
        static ref TSS: TaskStateSegment = {
            let mut tss = TaskStateSegment::new();
            let stack_start = VirtAddr::from_ptr(unsafe { &KERNEL_STACK });
            let stack_end = stack_start + STACK_SIZE;
            tss.privilege_stack_table[0] = stack_end;
            tss
        };
        static ref GDT: (GlobalDescriptorTable, Selectors) = {
            let mut gdt = GlobalDescriptorTable::new();
            let kernel_code = gdt.add_entry(Descriptor::kernel_code_segment());
            let kernel_data = gdt.add_entry(Descriptor::kernel_data_segment());
            let user_data = gdt.add_entry(Descriptor::user_data_segment());
            let user_code = gdt.add_entry(Descriptor::user_code_segment());
            let tss = gdt.add_entry(Descriptor::tss_segment(&TSS));
            (
                gdt,
                Selectors {
                    kernel_code,
                    kernel_data,
                    user_code,
                    user_data,
                    tss,
                },
            )
        };
    }

    pub fn init() {
        GDT.0.load();
        unsafe {
            CS::set_reg(GDT.1.kernel_code);
            x86_64::instructions::segmentation::SS::set_reg(GDT.1.kernel_data);
            x86_64::instructions::segmentation::DS::set_reg(GDT.1.kernel_data);
            x86_64::instructions::segmentation::ES::set_reg(GDT.1.kernel_data);
            load_tss(GDT.1.tss);
        }
    }
}

mod pci {
    use heapless::Vec;
    use x86_64::instructions::port::Port;

    #[derive(Clone, Copy)]
    pub struct PciDevice {
        pub bus: u8,
        pub device: u8,
        pub function: u8,
        pub class_code: u8,
        pub subclass: u8,
    }

    pub fn scan_ahci_controllers() -> Vec<PciDevice, 32> {
        let mut devices: Vec<PciDevice, 32> = Vec::new();
        for bus in 0..=255u8 {
            for device in 0..32u8 {
                for function in 0..8u8 {
                    let vendor = read_config(bus, device, function, 0x00) as u16;
                    if vendor == 0xFFFF {
                        continue;
                    }
                    let class = (read_config(bus, device, function, 0x08) >> 24) as u8;
                    let subclass = (read_config(bus, device, function, 0x08) >> 16) as u8;
                    if class == 0x01 && subclass == 0x06 {
                        let _ = devices.push(PciDevice {
                            bus,
                            device,
                            function,
                            class_code: class,
                            subclass,
                        });
                    }
                }
            }
        }
        devices
    }

    fn read_config(bus: u8, device: u8, function: u8, offset: u8) -> u32 {
        let address = (1u32 << 31)
            | ((bus as u32) << 16)
            | ((device as u32) << 11)
            | ((function as u32) << 8)
            | (offset as u32 & 0xFC);
        unsafe {
            let mut address_port = Port::new(0xCF8);
            let mut data_port = Port::new(0xCFC);
            address_port.write(address);
            data_port.read()
        }
    }
}

mod storage {
    #[repr(C, packed)]
    pub struct GptHeader {
        pub signature: [u8; 8],
        pub revision: u32,
        pub header_size: u32,
        pub header_crc32: u32,
        pub _reserved: u32,
        pub current_lba: u64,
        pub backup_lba: u64,
        pub first_usable_lba: u64,
        pub last_usable_lba: u64,
        pub disk_guid: [u8; 16],
        pub partition_entries_lba: u64,
        pub num_partition_entries: u32,
        pub size_of_partition_entry: u32,
        pub partition_entries_crc32: u32,
    }

    #[repr(C, packed)]
    pub struct GptPartitionEntry {
        pub partition_type_guid: [u8; 16],
        pub unique_partition_guid: [u8; 16],
        pub first_lba: u64,
        pub last_lba: u64,
        pub attributes: u64,
        pub name_utf16: [u16; 36],
    }

    pub fn parse_gpt_header(sector: &[u8]) -> Option<GptHeader> {
        if sector.len() < core::mem::size_of::<GptHeader>() {
            return None;
        }
        let mut header = GptHeader {
            signature: [0; 8],
            revision: 0,
            header_size: 0,
            header_crc32: 0,
            _reserved: 0,
            current_lba: 0,
            backup_lba: 0,
            first_usable_lba: 0,
            last_usable_lba: 0,
            disk_guid: [0; 16],
            partition_entries_lba: 0,
            num_partition_entries: 0,
            size_of_partition_entry: 0,
            partition_entries_crc32: 0,
        };
        unsafe {
            core::ptr::copy_nonoverlapping(
                sector.as_ptr(),
                &mut header as *mut GptHeader as *mut u8,
                core::mem::size_of::<GptHeader>(),
            );
        }
        if &header.signature != b"EFI PART" {
            return None;
        }
        Some(header)
    }

    pub fn parse_gpt_entries(buffer: &[u8], entry_size: usize) -> Option<&[GptPartitionEntry]> {
        if entry_size < core::mem::size_of::<GptPartitionEntry>() {
            return None;
        }
        let entries_len = buffer.len() / entry_size;
        let ptr = buffer.as_ptr() as *const GptPartitionEntry;
        unsafe { Some(core::slice::from_raw_parts(ptr, entries_len)) }
    }

    #[repr(C, packed)]
    pub struct ExfatBootSector {
        pub jump_boot: [u8; 3],
        pub fs_name: [u8; 8],
        pub _must_be_zero: [u8; 53],
        pub partition_offset: u64,
        pub volume_length: u64,
        pub fat_offset: u32,
        pub fat_length: u32,
        pub cluster_heap_offset: u32,
        pub cluster_count: u32,
        pub root_dir_cluster: u32,
        pub volume_serial: u32,
        pub fs_revision: u16,
        pub volume_flags: u16,
        pub bytes_per_sector_shift: u8,
        pub sectors_per_cluster_shift: u8,
        pub number_of_fats: u8,
        pub drive_select: u8,
        pub percent_in_use: u8,
        pub _reserved: [u8; 7],
        pub boot_code: [u8; 390],
        pub boot_signature: [u8; 2],
    }

    pub fn parse_exfat_boot_sector(sector: &[u8]) -> Option<ExfatBootSector> {
        if sector.len() < core::mem::size_of::<ExfatBootSector>() {
            return None;
        }
        let mut bpb = ExfatBootSector {
            jump_boot: [0; 3],
            fs_name: [0; 8],
            _must_be_zero: [0; 53],
            partition_offset: 0,
            volume_length: 0,
            fat_offset: 0,
            fat_length: 0,
            cluster_heap_offset: 0,
            cluster_count: 0,
            root_dir_cluster: 0,
            volume_serial: 0,
            fs_revision: 0,
            volume_flags: 0,
            bytes_per_sector_shift: 0,
            sectors_per_cluster_shift: 0,
            number_of_fats: 0,
            drive_select: 0,
            percent_in_use: 0,
            _reserved: [0; 7],
            boot_code: [0; 390],
            boot_signature: [0; 2],
        };
        unsafe {
            core::ptr::copy_nonoverlapping(
                sector.as_ptr(),
                &mut bpb as *mut ExfatBootSector as *mut u8,
                core::mem::size_of::<ExfatBootSector>(),
            );
        }
        if &bpb.fs_name != b"EXFAT   " {
            return None;
        }
        Some(bpb)
    }

    pub struct StorageReport {
        pub ahci_controllers: usize,
    }

    pub fn detect_storage() -> StorageReport {
        let controllers = crate::pci::scan_ahci_controllers();
        StorageReport {
            ahci_controllers: controllers.len(),
        }
    }

    const SECTOR_SIZE: usize = 512;

    pub fn demo_gpt_parse() -> bool {
        let mut sector = [0u8; SECTOR_SIZE];
        sector[0..8].copy_from_slice(b"EFI PART");
        parse_gpt_header(&sector).is_some()
    }

    pub fn demo_exfat_parse() -> bool {
        let mut sector = [0u8; SECTOR_SIZE];
        sector[3..11].copy_from_slice(b"EXFAT   ");
        parse_exfat_boot_sector(&sector).is_some()
    }
}

mod time {
    use core::sync::atomic::{AtomicI8, Ordering};

    use heapless::String;
    use x86_64::instructions::port::Port;

    static TZ_OFFSET: AtomicI8 = AtomicI8::new(0);

    pub fn set_timezone(offset: &str) -> Result<(), ()> {
        let offset = offset.trim();
        let (sign, digits) = if let Some(rest) = offset.strip_prefix('+') {
            (1i8, rest)
        } else if let Some(rest) = offset.strip_prefix('-') {
            (-1i8, rest)
        } else {
            (1i8, offset)
        };
        let hours: i8 = digits.parse().map_err(|_| ())?;
        if hours.abs() > 12 {
            return Err(());
        }
        TZ_OFFSET.store(hours * sign, Ordering::SeqCst);
        Ok(())
    }

    pub fn timezone_offset() -> String<8> {
        let offset = TZ_OFFSET.load(Ordering::SeqCst);
        let mut out = String::<8>::new();
        if offset >= 0 {
            let _ = out.push('+');
        } else {
            let _ = out.push('-');
        }
        let value = offset.unsigned_abs();
        let _ = push_two(&mut out, value);
        out
    }

    pub fn formatted_times() -> (String<32>, String<32>) {
        let (mut hour, minute, second) = read_rtc();
        let gmt = format_time("GMT", hour, minute, second);

        let offset = TZ_OFFSET.load(Ordering::SeqCst) as i16;
        hour = ((hour as i16 + offset + 24) % 24) as u8;
        let local_label = timezone_offset();
        let local = format_time(local_label.as_str(), hour, minute, second);
        (gmt, local)
    }

    fn format_time(label: &str, hour: u8, minute: u8, second: u8) -> String<32> {
        let mut out = String::<32>::new();
        let _ = out.push_str(label);
        let _ = out.push_str(" ");
        let _ = push_two(&mut out, hour);
        let _ = out.push(':');
        let _ = push_two(&mut out, minute);
        let _ = out.push(':');
        let _ = push_two(&mut out, second);
        out
    }

    fn push_two<const N: usize>(out: &mut String<N>, value: u8) -> Result<(), ()> {
        let tens = value / 10;
        let ones = value % 10;
        out.push((b'0' + tens) as char).map_err(|_| ())?;
        out.push((b'0' + ones) as char).map_err(|_| ())?;
        Ok(())
    }

    fn read_rtc() -> (u8, u8, u8) {
        while update_in_progress() {}
        let second = read_cmos(0x00);
        let minute = read_cmos(0x02);
        let hour = read_cmos(0x04);
        let status_b = read_cmos(0x0B);
        let is_bcd = (status_b & 0x04) == 0;
        let hour = if is_bcd { bcd_to_bin(hour) } else { hour };
        let minute = if is_bcd { bcd_to_bin(minute) } else { minute };
        let second = if is_bcd { bcd_to_bin(second) } else { second };
        (hour, minute, second)
    }

    fn update_in_progress() -> bool {
        read_cmos(0x0A) & 0x80 != 0
    }

    fn read_cmos(register: u8) -> u8 {
        unsafe {
            let mut port = Port::new(0x70);
            let mut data = Port::new(0x71);
            port.write(register);
            data.read()
        }
    }

    fn bcd_to_bin(value: u8) -> u8 {
        (value & 0x0F) + ((value >> 4) * 10)
    }
}

mod memory {
    use bootloader::bootinfo::{MemoryMap, MemoryRegionType};
    use core::fmt::Write;
    use heapless::String;
    use spin::Mutex;

    const FRAME_SIZE: u64 = 4096;
    const MAX_FRAMES: usize = 1024 * 1024;
    const BITMAP_BYTES: usize = MAX_FRAMES / 8;
    const HEAP_SIZE: usize = 1024 * 1024;

    pub struct FrameAllocator {
        bitmap: [u8; BITMAP_BYTES],
        total_frames: usize,
        free_frames: usize,
    }

    impl FrameAllocator {
        const fn new() -> Self {
            Self {
                bitmap: [0xFF; BITMAP_BYTES],
                total_frames: 0,
                free_frames: 0,
            }
        }

        fn mark_range_free(&mut self, start_frame: usize, frames: usize) {
            for frame in start_frame..start_frame.saturating_add(frames) {
                if frame >= MAX_FRAMES {
                    break;
                }
                let byte = frame / 8;
                let bit = frame % 8;
                if self.bitmap[byte] & (1 << bit) != 0 {
                    self.bitmap[byte] &= !(1 << bit);
                    self.free_frames += 1;
                }
            }
        }

        fn mark_range_used(&mut self, start_frame: usize, frames: usize) {
            for frame in start_frame..start_frame.saturating_add(frames) {
                if frame >= MAX_FRAMES {
                    break;
                }
                let byte = frame / 8;
                let bit = frame % 8;
                if self.bitmap[byte] & (1 << bit) == 0 {
                    self.bitmap[byte] |= 1 << bit;
                    self.free_frames = self.free_frames.saturating_sub(1);
                }
            }
        }

        pub fn alloc_frame(&mut self) -> Option<u64> {
            for (byte_index, byte) in self.bitmap.iter_mut().enumerate() {
                if *byte != 0xFF {
                    for bit in 0..8 {
                        if *byte & (1 << bit) == 0 {
                            *byte |= 1 << bit;
                            self.free_frames = self.free_frames.saturating_sub(1);
                            let frame_index = byte_index * 8 + bit;
                            let addr = frame_index as u64 * FRAME_SIZE;
                            return Some(addr);
                        }
                    }
                }
            }
            None
        }

        pub fn free_frame(&mut self, addr: u64) {
            let frame_index = (addr / FRAME_SIZE) as usize;
            if frame_index >= MAX_FRAMES {
                return;
            }
            let byte = frame_index / 8;
            let bit = frame_index % 8;
            if self.bitmap[byte] & (1 << bit) != 0 {
                self.bitmap[byte] &= !(1 << bit);
                self.free_frames += 1;
            }
        }

        pub fn stats(&self) -> (usize, usize) {
            (self.total_frames, self.free_frames)
        }
    }

    static FRAME_ALLOCATOR: Mutex<FrameAllocator> = Mutex::new(FrameAllocator::new());
    static mut HEAP_SPACE: [u8; HEAP_SIZE] = [0; HEAP_SIZE];
    static mut MEMORY_MAP: Option<&'static MemoryMap> = None;

    pub fn init_frame_allocator(map: &'static MemoryMap) {
        unsafe {
            MEMORY_MAP = Some(map);
        }
        let mut allocator = FRAME_ALLOCATOR.lock();
        allocator.total_frames = 0;
        allocator.free_frames = 0;
        allocator.bitmap = [0xFF; BITMAP_BYTES];

        for region in map.iter() {
            let start = region.range.start_addr();
            let end = region.range.end_addr();
            let start_frame = (start / FRAME_SIZE) as usize;
            let end_frame = ((end + FRAME_SIZE - 1) / FRAME_SIZE) as usize;
            let frames = end_frame.saturating_sub(start_frame);
            allocator.total_frames = allocator.total_frames.saturating_add(frames);

            if region.region_type == MemoryRegionType::Usable {
                allocator.mark_range_free(start_frame, frames);
            } else {
                allocator.mark_range_used(start_frame, frames);
            }
        }
    }

    pub fn init_heap() {
        unsafe {
            super::ALLOCATOR
                .lock()
                .init(HEAP_SPACE.as_mut_ptr(), HEAP_SPACE.len());
        }
    }

    pub fn summarize(map: &MemoryMap) -> String<64> {
        let mut total_regions = 0usize;
        let mut usable_regions = 0usize;
        let mut usable_bytes: u64 = 0;

        for region in map.iter() {
            total_regions += 1;
            if region.region_type == MemoryRegionType::Usable {
                usable_regions += 1;
                usable_bytes += region.range.end_addr() - region.range.start_addr();
            }
        }

        let mut summary: String<64> = String::new();
        let _ = write!(
            summary,
            "Mem: regions={} usable={} usable_mb={}",
            total_regions,
            usable_regions,
            usable_bytes / (1024 * 1024)
        );
        summary
    }

    pub fn allocator_summary() -> String<64> {
        let (total, free) = FRAME_ALLOCATOR.lock().stats();
        let mut summary: String<64> = String::new();
        let _ = write!(
            summary,
            "PMM: total={} free={} heap_kb={}",
            total,
            free,
            HEAP_SIZE / 1024
        );
        summary
    }

    pub fn allocate_demo_page() -> Option<u64> {
        FRAME_ALLOCATOR.lock().alloc_frame()
    }

    pub fn map() -> &'static MemoryMap {
        unsafe { MEMORY_MAP.expect("memory map not initialized") }
    }
}

mod vfs {
    use heapless::String;
    use heapless::Vec;

    const MAX_ENTRIES: usize = 32;
    const MAX_NAME: usize = 24;
    const MAX_DATA: usize = 256;

    #[derive(Clone)]
    pub enum EntryKind {
        File,
        Dir,
    }

    #[derive(Clone)]
    pub struct Entry {
        pub name: String<MAX_NAME>,
        pub kind: EntryKind,
        pub data: String<MAX_DATA>,
    }

    pub struct FileSystem {
        entries: Vec<Entry, MAX_ENTRIES>,
    }

    impl FileSystem {
        pub fn new() -> Self {
            Self {
                entries: Vec::new(),
            }
        }

        pub fn list(&self) -> Vec<Entry, MAX_ENTRIES> {
            self.entries.clone()
        }

        pub fn mkdir(&mut self, name: &str) -> bool {
            if self.find(name).is_some() || self.entries.is_full() {
                return false;
            }
            let mut entry = Entry {
                name: String::new(),
                kind: EntryKind::Dir,
                data: String::new(),
            };
            let _ = entry.name.push_str(name);
            self.entries.push(entry).is_ok()
        }

        pub fn touch(&mut self, name: &str) -> bool {
            if self.find(name).is_some() || self.entries.is_full() {
                return false;
            }
            let mut entry = Entry {
                name: String::new(),
                kind: EntryKind::File,
                data: String::new(),
            };
            let _ = entry.name.push_str(name);
            self.entries.push(entry).is_ok()
        }

        pub fn write(&mut self, name: &str, content: &str) -> bool {
            let Some(entry) = self.find_mut(name) else {
                return false;
            };
            if matches!(entry.kind, EntryKind::Dir) {
                return false;
            }
            entry.data.clear();
            let _ = entry.data.push_str(content);
            true
        }

        pub fn read(&self, name: &str) -> Option<String<MAX_DATA>> {
            let entry = self.find(name)?;
            if matches!(entry.kind, EntryKind::Dir) {
                return None;
            }
            Some(entry.data.clone())
        }

        pub fn delete(&mut self, name: &str) -> bool {
            if let Some(index) = self.entries.iter().position(|e| e.name.as_str() == name) {
                self.entries.remove(index);
                return true;
            }
            false
        }

        fn find(&self, name: &str) -> Option<&Entry> {
            self.entries.iter().find(|e| e.name.as_str() == name)
        }

        fn find_mut(&mut self, name: &str) -> Option<&mut Entry> {
            self.entries.iter_mut().find(|e| e.name.as_str() == name)
        }
    }
}

mod shell {
    use heapless::String;
    use heapless::Vec;

    use crate::memory;
    use crate::storage;
    use crate::tui;
    use crate::vga;
    use crate::vfs::{EntryKind, FileSystem};

    const MAX_INPUT: usize = 128;

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Role {
        User,
        Admin,
    }

    pub fn init() {
        init_fs();
        write_line("Type 'pierre help' for commands.");
        prompt();
    }

    pub fn handle_input(key: crate::keyboard::KeyInput) {
        if let Some(mode) = current_mode() {
            if mode == InputMode::Tui {
                match tui::handle_input(key) {
                    tui::TuiAction::Exit => {
                        set_mode(InputMode::Shell);
                        write_line("");
                        prompt();
                    }
                    tui::TuiAction::Handled => {}
                }
                return;
            }
            if mode == InputMode::Editor {
                handle_editor_input(key);
                return;
            }
        }
        let state = unsafe { STATE.get_or_insert_with(State::new) };

        match key {
            crate::keyboard::KeyInput::Char('\n') => {
                write_line("");
                process_command(state);
                prompt();
            }
            crate::keyboard::KeyInput::Char('\x08') => {
                if state.buffer.pop().is_some() {
                    vga::write_char('\u{8}');
                    vga::write_char(' ');
                    vga::write_char('\u{8}');
                }
            }
            crate::keyboard::KeyInput::Char(ch) => {
                if state.buffer.len() < MAX_INPUT {
                    let _ = state.buffer.push(ch);
                    vga::write_char(ch);
                }
            }
            _ => {}
        }
    }

    struct State {
        buffer: String<MAX_INPUT>,
    }

    static mut STATE: Option<State> = None;
    static mut MODE: InputMode = InputMode::Shell;

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum InputMode {
        Shell,
        Tui,
        Editor,
    }

    impl State {
        fn new() -> Self {
            Self {
                buffer: String::new(),
            }
        }
    }

    fn process_command(state: &mut State) {
        let input = state.buffer.clone();
        state.buffer.clear();
        handle_line(input.trim());
    }

    pub fn handle_line(line: &str) {
        if line.is_empty() {
            return;
        }

        let mut parts = line.split_whitespace();
        let Some(role_token) = parts.next() else {
            return;
        };

        let role = match role_token {
            "pierre" => Role::User,
            "suppiere" => Role::Admin,
            _ => {
                write_line("Prefix with 'pierre' or 'suppiere'.");
                return;
            }
        };

        let Some(command) = parts.next() else {
            write_line("No command provided.");
            return;
        };

        match command {
            "help" => cmd_help(),
            "dir" => cmd_dir(),
            "mkdir" => cmd_mkdir(parts.next()),
            "touch" => cmd_touch(parts.next()),
            "write" => cmd_write(parts.next(), parts.next()),
            "cat" => cmd_cat(parts.next()),
            "del" => cmd_del(parts.next()),
            "cls" => cmd_cls(),
            "mem" => cmd_mem(),
            "pmm" => cmd_pmm(),
            "gfx" => cmd_gfx(role),
            "whoami" => cmd_whoami(role),
            "version" => cmd_version(),
            "storage" => cmd_storage(),
            "gpt" => cmd_gpt(),
            "exfat" => cmd_exfat(),
            "banner" => cmd_banner(),
            "tui" => cmd_tui(),
            "edit" => cmd_edit(parts.next()),
            "time" => cmd_time(),
            "tz" => cmd_tz(parts.next()),
            _ => write_line("Unknown command. Try: pierre help"),
        }
    }

    fn cmd_help() {
        write_line("Commands:");
        write_line("  pierre help      - show help");
        write_line("  pierre dir       - list files/directories");
        write_line("  pierre mkdir     - create directory");
        write_line("  pierre touch     - create empty file");
        write_line("  pierre write     - write file contents");
        write_line("  pierre cat       - show file contents");
        write_line("  pierre del       - delete file/directory");
        write_line("  pierre cls       - clear screen");
        write_line("  pierre mem       - memory map summary");
        write_line("  pierre pmm       - PMM/heap summary");
        write_line("  pierre whoami    - show current role");
        write_line("  pierre version   - show kernel version");
        write_line("  pierre storage   - storage detection status");
        write_line("  pierre gpt       - test GPT parser");
        write_line("  pierre exfat     - test exFAT parser");
        write_line("  pierre banner    - show banner");
        write_line("  pierre tui       - show text UI layout");
        write_line("  pierre edit      - open text editor");
        write_line("  pierre time      - show GMT/local time");
        write_line("  pierre tz        - set timezone offset");
        write_line("  suppiere gfx     - redraw framebuffer demo");
    }

    fn cmd_dir() {
        let fs = unsafe { FS.as_ref().expect("fs not initialized") };
        let entries = fs.list();
        if entries.is_empty() {
            write_line("(empty)");
            return;
        }
        for entry in entries.iter() {
            match entry.kind {
                EntryKind::Dir => {
                    let mut line = String::<MAX_INPUT>::new();
                    let _ = line.push_str("<DIR> ");
                    let _ = line.push_str(entry.name.as_str());
                    write_line(line.as_str());
                }
                EntryKind::File => {
                    let mut line = String::<MAX_INPUT>::new();
                    let _ = line.push_str("      ");
                    let _ = line.push_str(entry.name.as_str());
                    write_line(line.as_str());
                }
            }
        }
    }

    fn cmd_mkdir(name: Option<&str>) {
        let Some(name) = name else {
            write_line("mkdir requires a name.");
            return;
        };
        let fs = unsafe { FS.as_mut().expect("fs not initialized") };
        if fs.mkdir(name) {
            write_line("Directory created.");
        } else {
            write_line("Unable to create directory.");
        }
    }

    fn cmd_touch(name: Option<&str>) {
        let Some(name) = name else {
            write_line("touch requires a name.");
            return;
        };
        let fs = unsafe { FS.as_mut().expect("fs not initialized") };
        if fs.touch(name) {
            write_line("File created.");
        } else {
            write_line("Unable to create file.");
        }
    }

    fn cmd_write(name: Option<&str>, content: Option<&str>) {
        let Some(name) = name else {
            write_line("write requires a filename.");
            return;
        };
        let Some(content) = content else {
            write_line("write requires content.");
            return;
        };
        let fs = unsafe { FS.as_mut().expect("fs not initialized") };
        if fs.write(name, content) {
            write_line("File updated.");
        } else {
            write_line("Unable to write file.");
        }
    }

    fn cmd_cat(name: Option<&str>) {
        let Some(name) = name else {
            write_line("cat requires a filename.");
            return;
        };
        let fs = unsafe { FS.as_ref().expect("fs not initialized") };
        if let Some(data) = fs.read(name) {
            write_line(data.as_str());
        } else {
            write_line("Unable to read file.");
        }
    }

    fn cmd_del(name: Option<&str>) {
        let Some(name) = name else {
            write_line("del requires a name.");
            return;
        };
        let fs = unsafe { FS.as_mut().expect("fs not initialized") };
        if fs.delete(name) {
            write_line("Deleted.");
        } else {
            write_line("Unable to delete.");
        }
    }

    fn cmd_cls() {
        vga::clear_screen();
    }

    fn cmd_mem() {
        let summary = memory::summarize(memory::map());
        write_line(summary.as_str());
    }

    fn cmd_pmm() {
        let summary = memory::allocator_summary();
        write_line(summary.as_str());
    }

    fn cmd_gfx(role: Role) {
        if role != Role::Admin {
            write_line("gfx requires suppiere (admin).");
            return;
        }
        crate::gfx::unsupported_message();
        write_line("Framebuffer support requires newer bootloader.");
    }

    fn cmd_whoami(role: Role) {
        match role {
            Role::User => write_line("Role: pierre (user)"),
            Role::Admin => write_line("Role: suppiere (admin)"),
        }
    }

    fn cmd_version() {
        write_line("NewDOS Base Kernel v0.1");
    }

    fn cmd_storage() {
        let report = storage::detect_storage();
        if report.ahci_controllers > 0 {
            write_line("Storage: AHCI controller detected.");
        } else {
            write_line("Storage: no AHCI controller detected.");
        }
    }

    fn cmd_gpt() {
        if storage::demo_gpt_parse() {
            write_line("GPT parser: OK");
        } else {
            write_line("GPT parser: failed");
        }
    }

    fn cmd_exfat() {
        if storage::demo_exfat_parse() {
            write_line("exFAT parser: OK");
        } else {
            write_line("exFAT parser: failed");
        }
    }

    fn cmd_banner() {
        crate::print_banner();
    }

    fn cmd_tui() {
        tui::draw();
        set_mode(InputMode::Tui);
        write_line("Text UI active. Use W/S + Enter, Q to exit.");
    }

    fn cmd_edit(name: Option<&str>) {
        let Some(name) = name else {
            write_line("edit requires a filename.");
            return;
        };
        editor_open(name);
    }

    fn cmd_time() {
        let (gmt, local) = crate::time::formatted_times();
        write_line(gmt.as_str());
        write_line(local.as_str());
    }

    fn cmd_tz(offset: Option<&str>) {
        let Some(offset) = offset else {
            write_line("tz requires an offset like +2 or -5.");
            return;
        };
        match crate::time::set_timezone(offset) {
            Ok(_) => write_line("Timezone updated."),
            Err(_) => write_line("Invalid timezone offset."),
        }
    }

    fn prompt() {
        vga::write_text("[NewDOS]> ");
    }

    fn write_line(message: &str) {
        if tui::is_console_mode() {
            tui::console_write_line(message);
        } else {
            vga::write_text(message);
            vga::write_text("\n");
        }
    }

    static mut FS: Option<FileSystem> = None;

    fn init_fs() {
        unsafe {
            if FS.is_none() {
                FS = Some(FileSystem::new());
            }
        }
    }

    fn set_mode(mode: InputMode) {
        unsafe {
            MODE = mode;
        }
    }

    fn current_mode() -> Option<InputMode> {
        unsafe { Some(MODE) }
    }

    pub fn list_entries() -> Vec<crate::vfs::Entry, 32> {
        let fs = unsafe { FS.as_ref().expect("fs not initialized") };
        fs.list()
    }

    struct EditorState {
        filename: String<32>,
        buffer: Vec<char, 1024>,
    }

    static mut EDITOR: Option<EditorState> = None;

    fn editor_open(name: &str) {
        let mut state = EditorState {
            filename: String::new(),
            buffer: Vec::new(),
        };
        let _ = state.filename.push_str(name);
        if let Some(existing) = unsafe { FS.as_ref() }.and_then(|fs| fs.read(name)) {
            for ch in existing.chars() {
                let _ = state.buffer.push(ch);
            }
        }
        unsafe {
            EDITOR = Some(state);
        }
        set_mode(InputMode::Editor);
        vga::clear_screen();
        vga::write_text("NewDOS Editor (F9=Save, F10=Exit)\n");
        render_editor();
    }

    fn handle_editor_input(key: crate::keyboard::KeyInput) {
        match key {
            crate::keyboard::KeyInput::F9 => {
                editor_save();
            }
            crate::keyboard::KeyInput::F10 => {
                editor_exit();
            }
            crate::keyboard::KeyInput::Char('\x08') => {
                if let Some(state) = unsafe { EDITOR.as_mut() } {
                    let _ = state.buffer.pop();
                    render_editor();
                }
            }
            crate::keyboard::KeyInput::Char(ch) => {
                if let Some(state) = unsafe { EDITOR.as_mut() } {
                    if state.buffer.push(ch).is_ok() {
                        render_editor();
                    }
                }
            }
            _ => {}
        }
    }

    fn render_editor() {
        if let Some(state) = unsafe { EDITOR.as_ref() } {
            vga::clear_screen();
            vga::write_text("NewDOS Editor (F9=Save, F10=Exit)\n");
            for ch in state.buffer.iter() {
                vga::write_char(*ch);
            }
        }
    }

    fn editor_save() {
        let (filename, contents) = if let Some(state) = unsafe { EDITOR.as_ref() } {
            let mut contents = String::<256>::new();
            for ch in state.buffer.iter() {
                let _ = contents.push(*ch);
            }
            (state.filename.clone(), contents)
        } else {
            return;
        };
        let fs = unsafe { FS.as_mut().expect("fs not initialized") };
        if !fs.write(filename.as_str(), contents.as_str()) {
            let _ = fs.touch(filename.as_str());
            if fs.write(filename.as_str(), contents.as_str()) {
                vga::write_status("Editor: saved");
            } else {
                vga::write_status("Editor: save failed");
            }
        } else {
            vga::write_status("Editor: saved");
        }
    }

    fn editor_exit() {
        unsafe {
            EDITOR = None;
        }
        set_mode(InputMode::Shell);
        vga::clear_screen();
        print_banner();
        prompt();
    }
}

mod tui {
    use heapless::String;

    use crate::vga::{self, Color};
    use crate::vfs::EntryKind;

    const APPS: [&str; 3] = ["Console", "Settings", "About"];
    const APP_ROW_START: usize = 4;
    const APP_COL_START: usize = 5;
    static mut SELECTED: usize = 0;
    static mut MODE: TuiMode = TuiMode::Launcher;

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum TuiMode {
        Launcher,
        Console,
    }

    pub enum TuiAction {
        Handled,
        Exit,
    }

    pub fn draw() {
        vga::clear_screen();
        vga::fill_region(0, 0, 80, 1, Color::Black, Color::LightGray, b' ');
        vga::fill_region(1, 0, 80, 24, Color::LightGreen, Color::Black, b' ');

        vga::draw_box(2, 2, 38, 10, Color::LightCyan, Color::Black, Some(" System "));
        vga::draw_box(2, 41, 37, 10, Color::LightCyan, Color::Black, Some(" Storage "));
        vga::draw_box(13, 2, 76, 10, Color::LightCyan, Color::Black, Some(" Console "));

        vga::write_status("NewDOS Text UI");
        draw_app_list();
        draw_cursor(0);
        draw_storage_list();
        draw_console_help();
        vga::write_text("\n");
    }

    pub fn handle_input(key: crate::keyboard::KeyInput) -> TuiAction {
        match key {
            crate::keyboard::KeyInput::Char('q') | crate::keyboard::KeyInput::Char('Q') => {
                vga::write_status("NewDOS Text UI exited");
                return TuiAction::Exit;
            }
            crate::keyboard::KeyInput::F1 => switch_mode(TuiMode::Console),
            crate::keyboard::KeyInput::F2 => switch_mode(TuiMode::Launcher),
            crate::keyboard::KeyInput::Char('w') | crate::keyboard::KeyInput::Char('W') => {
                if mode() == TuiMode::Launcher {
                    move_selection(-1);
                }
            }
            crate::keyboard::KeyInput::Char('s') | crate::keyboard::KeyInput::Char('S') => {
                if mode() == TuiMode::Launcher {
                    move_selection(1);
                }
            }
            crate::keyboard::KeyInput::Char('\n') => {
                if mode() == TuiMode::Launcher {
                    launch_selected();
                } else {
                    console_submit();
                }
            }
            crate::keyboard::KeyInput::F9 | crate::keyboard::KeyInput::F10 => {}
            crate::keyboard::KeyInput::Char('\x08') => {
                if mode() == TuiMode::Console {
                    console_backspace();
                }
            }
            crate::keyboard::KeyInput::Char(ch) => {
                if mode() == TuiMode::Console {
                    console_write_char(ch);
                }
            }
            crate::keyboard::KeyInput::Unknown => {}
        }
        TuiAction::Handled
    }

    fn draw_app_list() {
        for (index, app) in APPS.iter().enumerate() {
            let row = APP_ROW_START + index;
            let mut label = heapless::String::<32>::new();
            let _ = label.push_str("   ");
            let _ = label.push_str(app);
            let mut writer = vga::WRITER.lock();
            writer.write_at(row, APP_COL_START, label.as_str());
        }
    }

    fn draw_cursor(index: usize) {
        let mut writer = vga::WRITER.lock();
        let row = APP_ROW_START + index;
        writer.write_at(row, APP_COL_START, "> ");
    }

    fn clear_cursor(index: usize) {
        let mut writer = vga::WRITER.lock();
        let row = APP_ROW_START + index;
        writer.write_at(row, APP_COL_START, "  ");
    }

    fn move_selection(delta: isize) {
        unsafe {
            let current = SELECTED;
            clear_cursor(current);
            let max = APPS.len() as isize - 1;
            let mut next = current as isize + delta;
            if next < 0 {
                next = 0;
            } else if next > max {
                next = max;
            }
            SELECTED = next as usize;
            draw_cursor(SELECTED);
        }
    }

    fn launch_selected() {
        let selected = unsafe { SELECTED };
        if let Some(name) = APPS.get(selected) {
            let mut status = heapless::String::<64>::new();
            let _ = status.push_str("Launching ");
            let _ = status.push_str(name);
            let _ = status.push_str(" (placeholder)");
            vga::write_status(status.as_str());
            if *name == "Console" {
                switch_mode(TuiMode::Console);
            }
            if *name == "Settings" {
                draw_settings();
            }
        }
    }

    fn draw_storage_list() {
        let entries = crate::shell::list_entries();
        let mut row = 4usize;
        for entry in entries.iter().take(5) {
            let mut line = String::<32>::new();
            match entry.kind {
                EntryKind::Dir => {
                    let _ = line.push_str("<DIR> ");
                }
                EntryKind::File => {
                    let _ = line.push_str("      ");
                }
            }
            let _ = line.push_str(entry.name.as_str());
            let mut writer = vga::WRITER.lock();
            writer.write_at(row, 44, line.as_str());
            row += 1;
        }
    }

    fn draw_console_help() {
        let mut writer = vga::WRITER.lock();
        writer.write_at(14, 4, "F1 Console  F2 Apps  Q Exit");
    }

    fn draw_settings() {
        let mut writer = vga::WRITER.lock();
        writer.write_at(3, 44, "Timezone:");
        let offset = crate::time::timezone_offset();
        let mut line = String::<16>::new();
        let _ = line.push_str(offset.as_str());
        writer.write_at(4, 44, line.as_str());
    }

    fn switch_mode(new_mode: TuiMode) {
        unsafe {
            MODE = new_mode;
        }
        match new_mode {
            TuiMode::Console => vga::write_status("TUI console mode"),
            TuiMode::Launcher => vga::write_status("TUI launcher mode"),
        }
    }

    fn mode() -> TuiMode {
        unsafe { MODE }
    }

    pub fn is_console_mode() -> bool {
        mode() == TuiMode::Console
    }

    static mut CONSOLE_LINE: Option<String<64>> = None;

    pub fn console_write_line(message: &str) {
        let mut writer = vga::WRITER.lock();
        writer.write_at(16, 4, "                                                        ");
        writer.write_at(16, 4, message);
    }

    fn console_write_char(ch: char) {
        unsafe {
            if CONSOLE_LINE.is_none() {
                CONSOLE_LINE = Some(String::new());
            }
            if let Some(line) = CONSOLE_LINE.as_mut() {
                if line.push(ch).is_ok() {
                    console_write_line(line.as_str());
                }
            }
        }
    }

    fn console_backspace() {
        unsafe {
            if let Some(line) = CONSOLE_LINE.as_mut() {
                let _ = line.pop();
                console_write_line(line.as_str());
            }
        }
    }

    fn console_submit() {
        let line = unsafe {
            if let Some(line) = CONSOLE_LINE.take() {
                line
            } else {
                String::new()
            }
        };
        if !line.is_empty() {
            crate::shell::handle_line(line.as_str());
        }
        console_write_line("");
    }
}

mod interrupts {
    use lazy_static::lazy_static;
    use pic8259::ChainedPics;
    use spin::Mutex;
    use x86_64::structures::idt::{InterruptDescriptorTable, InterruptStackFrame};
    use x86_64::PrivilegeLevel;

    use crate::keyboard;

    pub const PIC_1_OFFSET: u8 = 32;
    pub const PIC_2_OFFSET: u8 = PIC_1_OFFSET + 8;

    pub static PICS: Mutex<ChainedPics> = Mutex::new(unsafe {
        ChainedPics::new(PIC_1_OFFSET, PIC_2_OFFSET)
    });

    #[derive(Clone, Copy)]
    #[repr(u8)]
    pub enum InterruptIndex {
        Timer = PIC_1_OFFSET,
        Keyboard,
        Mouse = PIC_2_OFFSET + 4,
    }

    impl InterruptIndex {
        fn as_u8(self) -> u8 {
            self as u8
        }

        fn as_usize(self) -> usize {
            usize::from(self.as_u8())
        }
    }

    lazy_static! {
        static ref IDT: InterruptDescriptorTable = {
            let mut idt = InterruptDescriptorTable::new();
            idt[InterruptIndex::Timer.as_usize()].set_handler_fn(timer_interrupt_handler);
            idt[InterruptIndex::Keyboard.as_usize()].set_handler_fn(keyboard_interrupt_handler);
            idt[InterruptIndex::Mouse.as_usize()].set_handler_fn(mouse_interrupt_handler);
            idt[0x80]
                .set_handler_fn(syscall_interrupt_handler)
                .set_privilege_level(PrivilegeLevel::Ring3);
            idt
        };
    }

    pub fn init_idt() {
        IDT.load();
    }

    extern "x86-interrupt" fn timer_interrupt_handler(_stack_frame: InterruptStackFrame) {
        unsafe {
            PICS.lock()
                .notify_end_of_interrupt(InterruptIndex::Timer.as_u8());
        }
    }

    extern "x86-interrupt" fn keyboard_interrupt_handler(_stack_frame: InterruptStackFrame) {
        keyboard::handle_keyboard_interrupt();
        unsafe {
            PICS.lock()
                .notify_end_of_interrupt(InterruptIndex::Keyboard.as_u8());
        }
    }

    extern "x86-interrupt" fn mouse_interrupt_handler(_stack_frame: InterruptStackFrame) {
        crate::mouse::handle_mouse_interrupt();
        unsafe {
            PICS.lock()
                .notify_end_of_interrupt(InterruptIndex::Mouse.as_u8());
        }
    }

    extern "x86-interrupt" fn syscall_interrupt_handler(_stack_frame: InterruptStackFrame) {
        crate::vga::write_status("Syscall: not implemented");
    }
}

mod keyboard {
    use core::fmt::Write;
    use core::sync::atomic::{AtomicBool, Ordering};

    use heapless::String;
    use lazy_static::lazy_static;
    use pc_keyboard::{layouts, HandleControl, Keyboard, ScancodeSet1};
    use spin::Mutex;
    use x86_64::instructions::port::Port;

    use crate::shell;
    use crate::vga;

    lazy_static! {
        static ref KEYBOARD: Mutex<Keyboard<layouts::Us104Key, ScancodeSet1>> =
            Mutex::new(Keyboard::new(ScancodeSet1::new(), layouts::Us104Key, HandleControl::Ignore));
    }

    static HAS_SCANCODE: AtomicBool = AtomicBool::new(false);
    static mut SCANCODE: u8 = 0;

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum KeyInput {
        Char(char),
        F1,
        F2,
        F9,
        F10,
        Unknown,
    }

    pub fn handle_keyboard_interrupt() {
        let mut port = Port::new(0x60);
        let scancode: u8 = unsafe { port.read() };
        unsafe {
            SCANCODE = scancode;
        }
        HAS_SCANCODE.store(true, Ordering::SeqCst);
    }

    pub fn poll() -> Option<KeyInput> {
        if !HAS_SCANCODE.swap(false, Ordering::SeqCst) {
            return None;
        }

        let scancode = unsafe { SCANCODE };
        let mut keyboard = KEYBOARD.lock();
        if let Ok(Some(key_event)) = keyboard.add_byte(scancode) {
            if let Some(key) = keyboard.process_keyevent(key_event) {
                return Some(match key {
                    pc_keyboard::DecodedKey::Unicode(ch) => KeyInput::Char(ch),
                    pc_keyboard::DecodedKey::RawKey(raw) => match raw {
                        pc_keyboard::KeyCode::F1 => KeyInput::F1,
                        pc_keyboard::KeyCode::F2 => KeyInput::F2,
                        pc_keyboard::KeyCode::F9 => KeyInput::F9,
                        pc_keyboard::KeyCode::F10 => KeyInput::F10,
                        _ => KeyInput::Unknown,
                    },
                });
            }
        }
        None
    }

    pub fn display_key(ch: char) {
        let mut status: String<32> = String::new();
        match ch {
            '\n' => {
                let _ = write!(status, "Key: ENTER");
            }
            '\x08' => {
                let _ = write!(status, "Key: BACKSPACE");
            }
            ch => {
                let _ = write!(status, "Key: {}", ch);
            }
        }
        vga::write_status(status.as_str());
    }

    pub fn handle_shell_input(key: KeyInput) {
        shell::handle_input(key);
    }
}

mod mouse {
    use core::fmt::Write;
    use core::sync::atomic::{AtomicU8, Ordering};

    use heapless::String;
    use x86_64::instructions::port::Port;

    use crate::vga;

    static PACKET_INDEX: AtomicU8 = AtomicU8::new(0);
    static mut PACKET: [u8; 3] = [0; 3];
    static mut X: i16 = 40;
    static mut Y: i16 = 12;

    pub fn init() {
        unsafe {
            let mut command_port = Port::new(0x64);
            let mut data_port = Port::new(0x60);

            command_port.write(0xA8u8); // enable aux device
            command_port.write(0x20u8); // get controller command byte
            let mut status: u8 = data_port.read();
            status |= 0x02; // enable IRQ12
            command_port.write(0x60u8);
            data_port.write(status);

            send_mouse_command(0xF6); // defaults
            send_mouse_command(0xF4); // enable streaming
        }
    }

    pub fn handle_mouse_interrupt() {
        let mut data_port = Port::new(0x60);
        let byte: u8 = unsafe { data_port.read() };

        let index = PACKET_INDEX.load(Ordering::SeqCst);
        unsafe {
            PACKET[index as usize] = byte;
        }
        let next = (index + 1) % 3;
        PACKET_INDEX.store(next, Ordering::SeqCst);

        if next == 0 {
            update_position();
        }
    }

    fn update_position() {
        let packet = unsafe { PACKET };
        let x_move = packet[1] as i8 as i16;
        let y_move = packet[2] as i8 as i16;

        unsafe {
            X = (X + x_move).clamp(0, 79);
            Y = (Y - y_move).clamp(1, 24);
            let left = packet[0] & 0x1 != 0;
            let right = packet[0] & 0x2 != 0;
            let middle = packet[0] & 0x4 != 0;
            let mut status: String<64> = String::new();
            let _ = write!(
                status,
                "Mouse: X={:02} Y={:02} L={} M={} R={}",
                X,
                Y,
                left as u8,
                middle as u8,
                right as u8
            );
            vga::write_status(status.as_str());
        }
    }

    fn send_mouse_command(command: u8) {
        unsafe {
            let mut command_port = Port::new(0x64);
            let mut data_port = Port::new(0x60);
            command_port.write(0xD4u8);
            data_port.write(command);
            let _ack: u8 = data_port.read();
        }
    }
}

fn print_banner() {
    let mut writer = vga::WRITER.lock();
    writer.set_color(vga::Color::Yellow, vga::Color::Black);
    let _ = writeln!(
        writer,
        "==============================\n=        NEWDOS VGA          =\n=   MINIMAL KERNEL BASE      =\n=============================="
    );
    writer.set_color(vga::Color::LightGreen, vga::Color::Black);
}

use crate::interrupts::PICS;

entry_point!(kernel_main);

fn kernel_main(boot_info: &'static BootInfo) -> ! {
    gdt::init();
    interrupts::init_idt();
    unsafe { PICS.lock().initialize() };
    x86_64::instructions::interrupts::enable();

    print_banner();
    mouse::init();

    memory::init_frame_allocator(&boot_info.memory_map);
    memory::init_heap();

    let mem_summary = memory::summarize(&boot_info.memory_map);
    vga::write_status(mem_summary.as_str());
    let pmm_summary = memory::allocator_summary();
    vga::write_status(pmm_summary.as_str());

    let _demo_frame = memory::allocate_demo_page();

    let storage_report = storage::detect_storage();
    if storage_report.ahci_controllers > 0 {
        vga::write_status("Storage: AHCI controller detected");
    } else {
        vga::write_status("Storage: no AHCI controller detected");
    }

    shell::init();

    loop {
    if let Some(key) = keyboard::poll() {
        keyboard::handle_shell_input(key);
    }
        x86_64::instructions::hlt();
    }
}

#[global_allocator]
static ALLOCATOR: LockedHeap = LockedHeap::empty();

#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    let mut writer = vga::WRITER.lock();
    let _ = writeln!(writer, "\nKERNEL PANIC: {}", info);
    loop {
        x86_64::instructions::hlt();
    }
}

#[alloc_error_handler]
fn alloc_error(layout: core::alloc::Layout) -> ! {
    let mut writer = vga::WRITER.lock();
    let _ = writeln!(writer, "\nALLOC ERROR: {:?}", layout);
    loop {
        x86_64::instructions::hlt();
    }
}
