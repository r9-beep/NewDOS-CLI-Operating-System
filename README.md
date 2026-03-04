# NewDOS Base Kernel

This project builds a **bootable x86_64 kernel** using **bootloader 0.11** (BIOS + UEFI).
It provides VESA VBE / UEFI GOP pixel-level framebuffer graphics, VGA text fallback,
PS/2 keyboard and mouse, a minimal CLI, and an in-memory filesystem.

## Features

- **VESA VBE / UEFI GOP framebuffer** — pixel drawing, shapes, circles, 8×8 font rendering.
- **Digital video out** — framebuffer pixels are output to HDMI / DisplayPort / VGA through
  the GPU; BIOS systems use VESA VBE mode-set, UEFI systems use GOP (set by bootloader 0.11).
- VGA text-mode fallback when no framebuffer is available.
- Keyboard and mouse input via PS/2 (IRQ1 / IRQ12).
- Bootloader 0.11 memory map (physical memory manager + 1 MiB kernel heap).
- GDT / TSS with kernel (ring 0) and user (ring 3) segments.
- GPT + exFAT struct helpers (no block device driver yet).
- PCI scan for AHCI controllers.
- Simple CLI with `pierre` (user) and `suppiere` (admin) command prefixes.
- In-memory VFS (mkdir / touch / write / cat / del).
- Simple text editor (`pierre edit <file>`) — F9 save, F10 exit.
- RTC time + timezone commands.
- Serial debug output on COM1.

## Graphics / Video output

The bootloader 0.11 automatically negotiates the highest available resolution using
**VESA VBE** (BIOS boot) or **UEFI GOP** (UEFI boot), then passes a linear framebuffer
pointer to the kernel in `BootInfo::framebuffer`.  The kernel writes RGB pixels directly
into this buffer — the GPU outputs the result on all connected displays (HDMI, DisplayPort,
DVI, VGA) without any additional driver code.

`suppiere gfx` redraws the colour palette / gradient / circle demo at any time.

## CLI usage

```
pierre help
pierre ls / pierre dir
pierre mkdir docs
pierre touch notes.txt
pierre write notes.txt hello world
pierre cat notes.txt
pierre del notes.txt
pierre mem
pierre storage
pierre gpt
pierre exfat
pierre edit notes.txt     (F9=save  F10=exit)
pierre time
pierre tz +2
pierre cd /
pierre user alice
pierre device NewDOS-Laptop
pierre version
pierre banner
suppiere gfx              (redraw framebuffer demo)
suppiere restart
```

## Build

Requirements:

```bash
rustup toolchain install nightly
rustup component add rust-src llvm-tools-preview --toolchain nightly
```

### One-step build (Makefile)

```bash
make          # compiles kernel + creates images/NewDOS-bios.img and images/NewDOS-uefi.img
make run-bios # launch in QEMU (BIOS, VESA VBE framebuffer)
make run-uefi # launch in QEMU (UEFI GOP, needs OVMF firmware)
```

### Manual build

```bash
# 1. Compile the kernel (bare-metal, no_std)
cargo +nightly build --package kernel --release \
    -Z build-std=core,compiler_builtins,alloc \
    -Z build-std-features=compiler-builtins-mem \
    -Z json-target-spec \
    --target kernel/x86_64-newdos.json

# 2. Create BIOS + UEFI disk images
cargo +nightly run --package disk-image-builder -- \
    target/x86_64-newdos/release/kernel images
```

### Run in QEMU

```bash
# BIOS (VESA VBE graphics)
qemu-system-x86_64 -drive format=raw,file=images/NewDOS-bios.img -m 256M -serial stdio

# UEFI (GOP graphics, higher resolution)
qemu-system-x86_64 \
    -bios /usr/share/OVMF/OVMF_CODE.fd \
    -drive format=raw,file=images/NewDOS-uefi.img \
    -m 256M -serial stdio
```

## Project structure

```
Cargo.toml                  workspace root
Makefile
kernel/                     OS kernel crate (no_std, bootloader_api 0.11)
  Cargo.toml
  x86_64-newdos.json        custom LLVM target spec
  src/
    main.rs                 entry point, boot sequence
    framebuffer.rs          VESA/GOP pixel graphics + 8×8 font
    gdt.rs                  GDT / TSS (ring 0 + ring 3)
    interrupts.rs           IDT, PIC, exception handlers
    keyboard.rs             PS/2 keyboard (pc-keyboard crate)
    mouse.rs                PS/2 mouse
    memory.rs               page-table mapper + frame allocator
    allocator.rs            kernel heap (linked_list_allocator)
    serial.rs               COM1 serial debug output
    vga.rs                  VGA text-mode fallback
    shell.rs                CLI shell
    vfs.rs                  in-memory virtual filesystem
    storage.rs              PCI/AHCI scan + GPT/exFAT structs
    time.rs                 RTC read + timezone formatting
disk-image-builder/         host tool: wraps kernel ELF → bootable .img
  Cargo.toml
  src/main.rs
images/                     build output: NewDOS-bios.img, NewDOS-uefi.img
```

> **Note:** There is no disk driver or disk-backed filesystem yet — VFS data lives in RAM only.
> Ring 3 user-mode processes are not implemented yet.
