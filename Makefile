KERNEL_ELF := target/x86_64-newdos/release/kernel
OUT_DIR     := images

.PHONY: all kernel image bios uefi run-bios run-uefi clean

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

bios: kernel image
	@echo "BIOS image: $(OUT_DIR)/NewDOS-bios.img"

uefi: kernel image
	@echo "UEFI image: $(OUT_DIR)/NewDOS-uefi.img"

run-bios: bios
	qemu-system-x86_64 \
	    -drive format=raw,file=$(OUT_DIR)/NewDOS-bios.img \
	    -m 256M \
	    -serial stdio \
	    -device ps2-mouse \
	    -no-reboot

run-uefi: uefi
	qemu-system-x86_64 \
	    -bios /usr/share/OVMF/OVMF_CODE.fd \
	    -drive format=raw,file=$(OUT_DIR)/NewDOS-uefi.img \
	    -m 256M \
	    -serial stdio \
	    -device ps2-mouse \
	    -no-reboot

clean:
	cargo clean
	rm -rf $(OUT_DIR)
