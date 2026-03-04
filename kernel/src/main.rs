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

// ── kernel entry point ────────────────────────────────────────────────────────

fn kernel_main(boot_info: &'static mut BootInfo) -> ! {
    serial_println!("NewDOS kernel starting (bootloader 0.11 / GUI)...");

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

    // Framebuffer — required for the GUI
    let (sw, sh) = if let Some(fb) = boot_info.framebuffer.as_mut() {
        let info = fb.info();
        let (w, h) = (info.width, info.height);
        serial_println!("[OK] Framebuffer {}x{}  format={:?}", w, h, info.pixel_format);
        let fb_static: &'static mut _ = unsafe { &mut *(fb as *mut _) };
        framebuffer::init(fb_static);
        (w, h)
    } else {
        serial_println!("[WARN] No framebuffer — VGA text fallback");
        (1024, 768) // assume default; GUI won't render without FB
    };

    mouse::init();
    serial_println!("[OK] Mouse");

    // ── Start GUI ─────────────────────────────────────────────────────────────
    serial_println!("[OK] Starting GUI ({}x{})", sw, sh);
    let mut gui = gui::Gui::new(sw, sh);

    // Busy loop — redraws GUI every iteration for smooth cursor movement
    loop {
        gui.tick();
        // Small yield via hlt only if no mouse activity to reduce CPU heat
        let updated = mouse::STATE.lock().updated;
        if !updated {
            x86_64::instructions::hlt();
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
