# NewDOS Base Kernel

This project builds a **bootable x86_64 base kernel** used as the foundation for the POPCoRN
and NKS kernels. It provides VGA text output plus basic PS/2 keyboard and mouse interrupt
handling, with a minimal CLI and an in-memory filesystem (no disk-backed FS yet).

## Features

- VGA text-mode output (banner + status line).
- Basic framebuffer graphics demo when a bootloader-provided framebuffer is available.
- Keyboard and mouse input via PS/2 scancodes.
- Bootloader-provided memory map summary at startup.
- Basic physical memory manager (PMM) and kernel heap initialization.
- Ring 0/3 groundwork (GDT/TSS setup with kernel + user segments).
- GPT + exFAT parsing structs/helpers (no block device driver yet).
- PCI scanning for AHCI controllers (SSD detection groundwork).
- Syscall interrupt gate (0x80) stub for future ring 3 calls.
- Simple CLI with `pierre` (user) and `suppiere` (admin) command prefixes.
- In-memory filesystem (mkdir/touch/write/cat/del) for CLI demos.
- GPT/exFAT parser demo commands (`pierre gpt`, `pierre exfat`).
- Text-based UI layout demo (`pierre tui`) for VGA-only setups.

> **Note:** There is no disk driver or disk-backed filesystem, so this is a minimal kernel foundation.
> External video output (HDMI/DisplayPort) depends on firmware-provided framebuffers (UEFI GOP).
> There is no native DisplayPort protocol/driver in this kernel yet.

## Privilege rings (ring 0 / ring 3)

The kernel runs in **ring 0**. User-mode (ring 3) processes are **not implemented yet**. A future
step would be to add page tables, user stacks, and a syscall/interrupt interface to transition
between ring 3 and ring 0.

## CLI usage

Commands use a role prefix:

- `pierre` = normal user
- `suppiere` = admin

Example commands:

```
pierre help
pierre dir
pierre mkdir docs
pierre touch notes.txt
pierre write notes.txt hello
pierre cat notes.txt
pierre del notes.txt
pierre mem
pierre gpt
pierre exfat
pierre tui
suppiere gfx
```

The text UI uses **W/S** to move, **Enter** to open, and **Q** to exit.

## Build and run in QEMU (BIOS)

You need the Rust nightly toolchain and the `bootimage` tool:

```bash
rustup default nightly
rustup target add x86_64-unknown-none
cargo install bootimage
```

Then build and run:

```bash
cargo bootimage
qemu-system-x86_64 -drive format=raw,file=target/x86_64-newdos/debug/bootimage-NewDOS-CLI-Operating-System.bin
```

> **Note:** `bootloader` v0.9 does **not** support the `uefi` feature. This repo is BIOS-only
> unless you upgrade the bootloader dependency.
