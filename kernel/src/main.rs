#![no_std]
#![no_main]
#![feature(abi_x86_interrupt)]
#![feature(alloc_error_handler)]

extern crate alloc;

use bootloader_api::{entry_point, BootInfo, BootloaderConfig};
use bootloader_api::config::Mapping;
use x86_64::VirtAddr;

pub mod allocator;
pub mod cli;
pub mod framebuffer;
pub mod gdt;
pub mod gui;
pub mod interrupts;
pub mod keyboard;
pub mod memory;
pub mod mouse;
pub mod serial;
pub mod shell;
pub mod storage;
pub mod time;
pub mod vfs;
pub mod vga;

// ── bootloader configuration ──────────────────────────────────────────────────

static BOOTLOADER_CONFIG: BootloaderConfig = {
    let mut cfg = BootloaderConfig::new_default();
    cfg.mappings.physical_memory = Some(Mapping::Dynamic);
    cfg
};

entry_point!(kernel_main, config = &BOOTLOADER_CONFIG);

// ── boot mode ─────────────────────────────────────────────────────────────────

enum BootMode { Gui, Cli }

fn await_boot_mode() -> BootMode {
    loop {
        x86_64::instructions::hlt();
        if let Some(c) = keyboard::pop_char() {
            match c {
                '1' | '\r' | '\n' => return BootMode::Gui,
                '2'               => return BootMode::Cli,
                _                 => {}
            }
        }
    }
}

// ── kernel entry point ────────────────────────────────────────────────────────

fn kernel_main(boot_info: &'static mut BootInfo) -> ! {
    serial_println!("NewDOS kernel starting (bootloader 0.11)...");

    gdt::init();
    serial_println!("[OK] GDT");

    interrupts::init();
    serial_println!("[OK] Interrupts");

    let phys_offset = VirtAddr::new(
        boot_info.physical_memory_offset.into_option().unwrap_or(0),
    );
    let mut mapper      = unsafe { memory::init(phys_offset) };
    let mut frame_alloc = unsafe { memory::BootInfoFrameAllocator::init(&boot_info.memory_regions) };
    allocator::init_heap(&mut mapper, &mut frame_alloc).expect("heap init failed");
    serial_println!("[OK] Heap");

    let (sw, sh) = if let Some(fb) = boot_info.framebuffer.as_mut() {
        let info = fb.info();
        let (w, h) = (info.width, info.height);
        serial_println!("[OK] Framebuffer {}x{}  format={:?}", w, h, info.pixel_format);
        let fb_static: &'static mut _ = unsafe { &mut *(fb as *mut _) };
        framebuffer::init(fb_static);
        (w, h)
    } else {
        serial_println!("[WARN] No framebuffer");
        (1024, 768)
    };

    keyboard::init();
    serial_println!("[OK] Keyboard (PS/2 port 1 enabled)");

    mouse::init();
    serial_println!("[OK] Mouse (PS/2 port 2 enabled)");

    // ── Boot menu ─────────────────────────────────────────────────────────────
    framebuffer::draw_boot_menu();
    serial_println!("[OK] Boot menu — waiting for selection");

    match await_boot_mode() {
        BootMode::Gui => {
            serial_println!("[OK] GUI mode selected");
            mouse::set_bounds(sw, sh);
            let mut gui = gui::Gui::new(sw, sh);
            loop {
                gui.tick();
                let updated = mouse::STATE.lock().updated;
                if !updated { x86_64::instructions::hlt(); }
            }
        }
        BootMode::Cli => {
            serial_println!("[OK] CLI mode selected");
            let mut cli = cli::CliMode::new();
            loop {
                cli.tick();
                x86_64::instructions::hlt();
            }
        }
    }
}

// ── panic handler ─────────────────────────────────────────────────────────────

#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    serial_println!("KERNEL PANIC: {}", info);
    {
        let mut guard = framebuffer::FB_WRITER.lock();
        if let Some(w) = guard.as_mut() {
            w.set_colors(framebuffer::Rgb::RED, framebuffer::Rgb::BLACK);
            use core::fmt::Write;
            let _ = write!(w, "\n\n  PANIC: {}\n", info);
        }
    }
    loop { x86_64::instructions::hlt(); }
}

#[alloc_error_handler]
fn oom(layout: core::alloc::Layout) -> ! {
    panic!("out of memory: {:?}", layout);
}
