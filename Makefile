KERNEL_ELF := target/x86_64-newdos/release/kernel
OUT_DIR     := images

.PHONY: all kernel image bios uefi iso run-bios run-uefi clean

all: kernel image

kernel:
	cargo +nightly build --package kernel --release \
	    -Z build-std=core,compiler_builtins,alloc \
	    -Z build-std-features=compiler-builtins-mem \
	    -Z json-target-spec \
	    --target kernel/x86_64-newdos.json

image: $(KERNEL_ELF)
	mkdir -p $(OUT_DIR)
	cargo +nightly run --package disk-image-builder -- $(KERNEL_ELF) $(OUT_DIR)
	# Extend to 10 MB so the data area at LBA 8192 (4 MB) always exists
	truncate -s 10M $(OUT_DIR)/NewDOS-bios.img
	truncate -s 10M $(OUT_DIR)/NewDOS-uefi.img

bios: kernel image
	@echo "BIOS image: $(OUT_DIR)/NewDOS-bios.img"

uefi: kernel image
	@echo "UEFI image: $(OUT_DIR)/NewDOS-uefi.img"

# Build a UEFI-bootable ISO (El Torito EFI, platform 0xEF).
# The EFI System Partition lives at LBA 34, length 4096 sectors in the UEFI disk image.
# Flash the .img files to USB for BIOS boot; burn/use the ISO for UEFI CD/DVD boot.
iso: image
	mkdir -p /tmp/newdos-iso
	dd if=$(OUT_DIR)/NewDOS-uefi.img bs=512 skip=34 count=4096 \
	    of=/tmp/newdos-iso/efi.img 2>/dev/null
	xorriso -as mkisofs \
	    -o $(OUT_DIR)/NewDOS.iso \
	    -V "NEWDOS" \
	    --efi-boot efi.img \
	    -efi-boot-part --efi-boot-image \
	    /tmp/newdos-iso/ 2>/dev/null
	rm -rf /tmp/newdos-iso
	@echo "ISO:       $(OUT_DIR)/NewDOS.iso   (UEFI boot from CD/DVD)"
	@echo "Flash USB: dd if=$(OUT_DIR)/NewDOS-bios.img of=/dev/sdX bs=4M"

run-bios: bios
	qemu-system-x86_64 \
	    -drive format=raw,file=$(OUT_DIR)/NewDOS-bios.img \
	    -m 256M \
	    -serial stdio \
	    -no-reboot

run-uefi: uefi
	qemu-system-x86_64 \
	    -bios /usr/share/OVMF/OVMF_CODE.fd \
	    -drive format=raw,file=$(OUT_DIR)/NewDOS-uefi.img \
	    -m 256M \
	    -serial stdio \
	    -no-reboot

clean:
	cargo clean
	rm -rf $(OUT_DIR)
