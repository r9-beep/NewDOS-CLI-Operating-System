//! PCI scan for AHCI / SATA controllers and basic GPT/exFAT struct stubs.

use x86_64::instructions::port::Port;

fn pci_read(bus: u8, dev: u8, func: u8, offset: u8) -> u32 {
    let addr: u32 = 0x8000_0000
        | ((bus  as u32) << 16)
        | ((dev  as u32) << 11)
        | ((func as u32) << 8)
        | ((offset & 0xFC) as u32);
    unsafe {
        let mut cfg_addr: Port<u32> = Port::new(0xCF8);
        let mut cfg_data: Port<u32> = Port::new(0xCFC);
        cfg_addr.write(addr);
        cfg_data.read()
    }
}

pub fn detect_ahci() -> bool {
    for bus in 0u8..=255 {
        for dev in 0u8..32 {
            let id = pci_read(bus, dev, 0, 0);
            if id == 0xFFFF_FFFF { continue; }
            let class = pci_read(bus, dev, 0, 0x08);
            let class_code = (class >> 24) & 0xFF;
            let sub_class  = (class >> 16) & 0xFF;
            // 0x01 = Mass storage, 0x06 = SATA (AHCI)
            if class_code == 0x01 && sub_class == 0x06 {
                return true;
            }
        }
    }
    false
}

#[repr(C, packed)]
pub struct GptHeader {
    pub signature:        [u8; 8],
    pub revision:         u32,
    pub header_size:      u32,
    pub header_crc32:     u32,
    pub reserved:         u32,
    pub my_lba:           u64,
    pub alternate_lba:    u64,
    pub first_usable_lba: u64,
    pub last_usable_lba:  u64,
    pub disk_guid:        [u8; 16],
    pub part_entry_lba:   u64,
    pub num_part_entries: u32,
    pub part_entry_size:  u32,
    pub part_crc32:       u32,
}

impl GptHeader {
    pub fn is_valid(&self) -> bool {
        &self.signature == b"EFI PART"
    }
}

#[repr(C, packed)]
pub struct ExfatBootSector {
    pub jump_boot:        [u8; 3],
    pub oem_name:         [u8; 8],
    pub _reserved:        [u8; 53],
    pub partition_offset: u64,
    pub volume_length:    u64,
    pub fat_offset:       u32,
    pub fat_length:       u32,
    pub cluster_heap_off: u32,
    pub cluster_count:    u32,
    pub first_cluster_of_root: u32,
    pub volume_serial:    u32,
    pub fs_revision:      u16,
    pub volume_flags:     u16,
    pub bytes_per_sector: u8,
    pub sectors_per_cluster: u8,
    pub num_fats:         u8,
    pub drive_select:     u8,
    pub percent_in_use:   u8,
    pub _reserved2:       [u8; 7],
}
