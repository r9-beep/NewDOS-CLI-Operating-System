//! Creates BIOS and UEFI bootable disk images from the compiled kernel binary.
//!
//! Usage:
//!   disk-image-builder <path-to-kernel-elf> <out-dir>
//!
//! Produces:
//!   <out-dir>/NewDOS-bios.img   — BIOS bootable raw disk image (VESA VBE)
//!   <out-dir>/NewDOS-uefi.img   — UEFI bootable disk image (UEFI GOP)

use bootloader::DiskImageBuilder;
use std::{path::PathBuf, env};

fn main() {
    let mut args = env::args().skip(1);
    let kernel_path = PathBuf::from(
        args.next().expect("Usage: disk-image-builder <kernel-elf> <out-dir>"),
    );
    let out_dir = PathBuf::from(
        args.next().unwrap_or_else(|| String::from(".")),
    );

    println!("Building disk images from: {}", kernel_path.display());

    let mut builder = DiskImageBuilder::new(kernel_path);

    // BIOS image — bootloader uses VESA VBE for graphics (digital video out via GPU)
    let bios_path = out_dir.join("NewDOS-bios.img");
    builder.create_bios_image(&bios_path)
        .expect("failed to create BIOS image");
    println!("BIOS image: {}", bios_path.display());

    // UEFI image — bootloader uses UEFI GOP (better resolution, HDMI/DP/VGA)
    let uefi_path = out_dir.join("NewDOS-uefi.img");
    builder.create_uefi_image(&uefi_path)
        .expect("failed to create UEFI image");
    println!("UEFI image: {}", uefi_path.display());

    println!("");
    println!("Run in QEMU (BIOS):");
    println!("  qemu-system-x86_64 -drive format=raw,file={}", bios_path.display());
    println!("");
    println!("Run in QEMU (UEFI, needs OVMF):");
    println!("  qemu-system-x86_64 -bios /usr/share/OVMF/OVMF_CODE.fd \\");
    println!("    -drive format=raw,file={}", uefi_path.display());
}
