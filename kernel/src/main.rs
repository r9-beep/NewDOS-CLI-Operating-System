#![no_std]
#![no_main]
#![feature(abi_x86_interrupt)]
#![feature(alloc_error_handler)]

extern crate alloc;

use bootloader_api::{entry_point, BootInfo, BootloaderConfig};
use bootloader_api::config::Mapping;
use x86_64::VirtAddr;

pub mod allocator;
pub mod framebuffer;
pub mod gdt;
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

// ── kernel entry point ────────────────────────────────────────────────────────

fn kernel_main(boot_info: &'static mut BootInfo) -> ! {
    // Serial first so we have debug output from the start
    serial_println!("NewDOS kernel starting (bootloader 0.11)...");

    // GDT / TSS
    gdt::init();
    serial_println!("[OK] GDT");

    // IDT + PIC
    interrupts::init();
    serial_println!("[OK] Interrupts");

    // Physical memory manager + kernel heap
    let phys_offset = VirtAddr::new(
        boot_info.physical_memory_offset.into_option().unwrap_or(0),
    );
    let mut mapper = unsafe { memory::init(phys_offset) };
    let mut frame_alloc = unsafe {
        memory::BootInfoFrameAllocator::init(&boot_info.memory_regions)
    };
    allocator::init_heap(&mut mapper, &mut frame_alloc)
        .expect("heap init failed");
    serial_println!("[OK] Heap");

    // Framebuffer / VESA / UEFI GOP graphics
    if let Some(fb) = boot_info.framebuffer.as_mut() {
        serial_println!(
            "[OK] Framebuffer {}x{}  format={:?}",
            fb.info().width,
            fb.info().height,
            fb.info().pixel_format,
        );
        // Safety: we take a &'static mut by extending the lifetime of the
        // bootloader-provided framebuffer, which lives for the whole boot.
        let fb_static: &'static mut _ = unsafe {
            &mut *(fb as *mut _)
        };
        framebuffer::init(fb_static);
        framebuffer::draw_splash();
        serial_println!("[OK] Splash screen drawn");
    } else {
        serial_println!("[WARN] No framebuffer; using VGA text mode");
    }

    // PS/2 mouse
    mouse::init();
    serial_println!("[OK] Mouse");

    // Shell
    let mut shell = shell::ShellState::new();
    shell.init();
    serial_println!("[OK] Shell ready");

    // ── main loop ─────────────────────────────────────────────────────────────
    loop {
        shell.tick();
        x86_64::instructions::hlt();
    }
}

// ── panic handler ─────────────────────────────────────────────────────────────

#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    serial_println!("KERNEL PANIC: {}", info);
    // Also show on framebuffer if available
    {
        let mut guard = framebuffer::FB_WRITER.lock();
        if let Some(w) = guard.as_mut() {
            w.set_colors(framebuffer::Rgb::RED, framebuffer::Rgb::BLACK);
            use core::fmt::Write;
            let _ = write!(w, "\n\n  KERNEL PANIC: {}\n", info);
        }
    }
    loop {
        x86_64::instructions::hlt();
    }
}

// ── OOM handler ───────────────────────────────────────────────────────────────

#[alloc_error_handler]
fn oom(layout: core::alloc::Layout) -> ! {
    panic!("out of memory: {:?}", layout);
}
