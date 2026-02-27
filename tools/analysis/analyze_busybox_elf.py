#!/usr/bin/env python3
"""Analyze busybox-static ELF structure for relocation support."""
import struct

with open('binaries/busybox-static', 'rb') as f:
    data = f.read()

# ELF Header
magic = data[:4]
print(f"Magic: {magic}")
ei_class = data[4]  # 1=32bit, 2=64bit
print(f"Class: {ei_class} ({'64-bit' if ei_class == 2 else '32-bit'})")

# Parse 64-bit ELF header
e_type = struct.unpack('<H', data[16:18])[0]
type_names = {0: 'NONE', 1: 'REL', 2: 'EXEC', 3: 'DYN', 4: 'CORE'}
print(f"Type: {e_type} ({type_names.get(e_type, 'UNKNOWN')})")

e_entry = struct.unpack('<Q', data[24:32])[0]
print(f"Entry: 0x{e_entry:X}")

e_phoff = struct.unpack('<Q', data[32:40])[0]
print(f"Program headers offset: 0x{e_phoff:X}")

e_phentsize = struct.unpack('<H', data[54:56])[0]
print(f"Program header entry size: {e_phentsize}")

e_phnum = struct.unpack('<H', data[56:58])[0]
print(f"Number of program headers: {e_phnum}")

# Parse program headers
print("\n=== Program Headers ===")
PT_TYPES = {
    0: 'NULL', 1: 'LOAD', 2: 'DYNAMIC', 3: 'INTERP', 4: 'NOTE',
    5: 'SHLIB', 6: 'PHDR', 7: 'TLS',
    0x6474e550: 'GNU_EH_FRAME', 0x6474e551: 'GNU_STACK',
    0x6474e552: 'GNU_RELRO', 0x6474e553: 'GNU_PROPERTY'
}

dynamic_offset = None
dynamic_size = None
relro_addr = None
relro_size = None

for i in range(e_phnum):
    offset = e_phoff + i * e_phentsize
    p_type = struct.unpack('<I', data[offset:offset+4])[0]
    p_flags = struct.unpack('<I', data[offset+4:offset+8])[0]
    p_offset = struct.unpack('<Q', data[offset+8:offset+16])[0]
    p_vaddr = struct.unpack('<Q', data[offset+16:offset+24])[0]
    p_paddr = struct.unpack('<Q', data[offset+24:offset+32])[0]
    p_filesz = struct.unpack('<Q', data[offset+32:offset+40])[0]
    p_memsz = struct.unpack('<Q', data[offset+40:offset+48])[0]
    p_align = struct.unpack('<Q', data[offset+48:offset+56])[0]

    type_name = PT_TYPES.get(p_type, f'0x{p_type:X}')
    flags_str = ''.join([
        'R' if p_flags & 4 else '-',
        'W' if p_flags & 2 else '-',
        'X' if p_flags & 1 else '-'
    ])
    print(f"  [{i}] {type_name:15} {flags_str}  vaddr=0x{p_vaddr:08X}  filesz=0x{p_filesz:X}  memsz=0x{p_memsz:X}")

    if p_type == 2:  # PT_DYNAMIC
        dynamic_offset = p_offset
        dynamic_size = p_filesz
        dynamic_vaddr = p_vaddr
        print(f"      -> DYNAMIC at offset 0x{p_offset:X}, size {p_filesz}")

    if p_type == 0x6474e552:  # PT_GNU_RELRO
        relro_addr = p_vaddr
        relro_size = p_memsz

# Parse DYNAMIC section
if dynamic_offset:
    print("\n=== Dynamic Section ===")
    DT_TAGS = {
        0: 'NULL', 1: 'NEEDED', 2: 'PLTRELSZ', 3: 'PLTGOT', 4: 'HASH',
        5: 'STRTAB', 6: 'SYMTAB', 7: 'RELA', 8: 'RELASZ', 9: 'RELAENT',
        10: 'STRSZ', 11: 'SYMENT', 12: 'INIT', 13: 'FINI', 14: 'SONAME',
        15: 'RPATH', 16: 'SYMBOLIC', 17: 'REL', 18: 'RELSZ', 19: 'RELENT',
        20: 'PLTREL', 21: 'DEBUG', 22: 'TEXTREL', 23: 'JMPREL', 24: 'BIND_NOW',
        25: 'INIT_ARRAY', 26: 'FINI_ARRAY', 27: 'INIT_ARRAYSZ', 28: 'FINI_ARRAYSZ',
        29: 'RUNPATH', 30: 'FLAGS', 32: 'PREINIT_ARRAY', 33: 'PREINIT_ARRAYSZ',
        0x6ffffef5: 'GNU_HASH', 0x6ffffff0: 'VERSYM', 0x6ffffffa: 'RELCOUNT',
        0x6ffffffb: 'FLAGS_1', 0x6ffffffe: 'VERNEED', 0x6fffffff: 'VERNEEDNUM',
        0x6ffffff9: 'RELACOUNT'
    }

    rela_offset = None
    rela_size = None
    relcount = None

    pos = dynamic_offset
    while pos < dynamic_offset + dynamic_size:
        d_tag = struct.unpack('<Q', data[pos:pos+8])[0]
        d_val = struct.unpack('<Q', data[pos+8:pos+16])[0]
        tag_name = DT_TAGS.get(d_tag, f'0x{d_tag:X}')

        if d_tag == 0:  # DT_NULL
            break

        # Print important ones
        if d_tag in [7, 8, 9, 23, 25, 27, 0x6ffffff9, 0x6ffffffa]:
            print(f"  {tag_name:15} = 0x{d_val:X}")

        if d_tag == 7:  # DT_RELA
            rela_offset = d_val
        if d_tag == 8:  # DT_RELASZ
            rela_size = d_val
        if d_tag == 0x6ffffff9:  # DT_RELACOUNT
            relcount = d_val

        pos += 16

    if rela_offset and rela_size:
        print(f"\n=== Relocations (RELA) ===")
        print(f"Offset: 0x{rela_offset:X}, Size: {rela_size}, Count: {relcount or 'unknown'}")

        # Parse first few relocations
        # For static-PIE, most relocations are R_AARCH64_RELATIVE (type 1027)
        R_AARCH64_TYPES = {
            0: 'NONE', 257: 'ABS64', 258: 'ABS32',
            1026: 'JUMP_SLOT', 1027: 'RELATIVE', 1025: 'GLOB_DAT'
        }

        rela_count = rela_size // 24  # Each RELA entry is 24 bytes
        print(f"Total RELA entries: {rela_count}")

        # Show first 10
        for i in range(min(10, rela_count)):
            entry_offset = rela_offset + i * 24
            r_offset = struct.unpack('<Q', data[entry_offset:entry_offset+8])[0]
            r_info = struct.unpack('<Q', data[entry_offset+8:entry_offset+16])[0]
            r_addend = struct.unpack('<q', data[entry_offset+16:entry_offset+24])[0]  # signed

            r_type = r_info & 0xFFFFFFFF
            r_sym = r_info >> 32
            type_name = R_AARCH64_TYPES.get(r_type, f'type_{r_type}')

            print(f"  [{i:4}] offset=0x{r_offset:08X}  type={type_name:12}  addend=0x{r_addend:08X}")

        # Count types
        type_counts = {}
        for i in range(rela_count):
            entry_offset = rela_offset + i * 24
            r_info = struct.unpack('<Q', data[entry_offset+8:entry_offset+16])[0]
            r_type = r_info & 0xFFFFFFFF
            type_counts[r_type] = type_counts.get(r_type, 0) + 1

        print("\nRelocation type counts:")
        for r_type, count in sorted(type_counts.items()):
            type_name = R_AARCH64_TYPES.get(r_type, f'type_{r_type}')
            print(f"  {type_name}: {count}")
