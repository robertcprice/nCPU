#!/usr/bin/env python3
"""Verify ELF loading - compare file bytes vs memory."""
import struct

# Read the binary file
with open('binaries/busybox-static', 'rb') as f:
    data = f.read()

print("=== ELF Header ===")
e_entry = struct.unpack('<Q', data[24:32])[0]
e_phoff = struct.unpack('<Q', data[32:40])[0]
e_phentsize = struct.unpack('<H', data[54:56])[0]
e_phnum = struct.unpack('<H', data[56:58])[0]

print(f"Entry: 0x{e_entry:X}")
print(f"Program headers: {e_phnum} at offset 0x{e_phoff:X}")

# Find DYNAMIC segment
print("\n=== Program Headers ===")
for i in range(e_phnum):
    offset = e_phoff + i * e_phentsize
    p_type = struct.unpack('<I', data[offset:offset+4])[0]
    p_offset = struct.unpack('<Q', data[offset+8:offset+16])[0]
    p_vaddr = struct.unpack('<Q', data[offset+16:offset+24])[0]
    p_filesz = struct.unpack('<Q', data[offset+32:offset+40])[0]
    
    type_names = {1: 'LOAD', 2: 'DYNAMIC', 6: 'PHDR', 0x6474e551: 'GNU_STACK', 0x6474e552: 'GNU_RELRO'}
    type_name = type_names.get(p_type, f'0x{p_type:X}')
    
    if p_type in (1, 2):  # LOAD or DYNAMIC
        print(f"  [{i}] {type_name:12} file_off=0x{p_offset:X} vaddr=0x{p_vaddr:X} filesz=0x{p_filesz:X}")
        
        if p_type == 2:  # DYNAMIC
            print(f"\n=== DYNAMIC segment content (from file at offset 0x{p_offset:X}) ===")
            DT_TAGS = {
                0: 'NULL', 1: 'NEEDED', 7: 'RELA', 8: 'RELASZ', 9: 'RELAENT',
                25: 'INIT_ARRAY', 27: 'INIT_ARRAYSZ', 0x6ffffff9: 'RELACOUNT'
            }
            for j in range(20):
                dyn_off = p_offset + j * 16
                if dyn_off + 16 > len(data):
                    break
                d_tag = struct.unpack('<Q', data[dyn_off:dyn_off+8])[0]
                d_val = struct.unpack('<Q', data[dyn_off+8:dyn_off+16])[0]
                tag_name = DT_TAGS.get(d_tag, f'0x{d_tag:X}')
                print(f"    DT[{j:2}]: tag={tag_name:15} val=0x{d_val:X}")
                
                # Show raw bytes for first few
                if j < 3:
                    raw = data[dyn_off:dyn_off+16]
                    print(f"            raw: {raw.hex()}")
                
                if d_tag == 0:
                    break

# Check what SHOULD be at vaddr 0x11D130 
print("\n=== Checking what should be at vaddr 0x11D130 ===")
# Find which segment contains 0x11D130
for i in range(e_phnum):
    offset = e_phoff + i * e_phentsize
    p_type = struct.unpack('<I', data[offset:offset+4])[0]
    p_offset = struct.unpack('<Q', data[offset+8:offset+16])[0]
    p_vaddr = struct.unpack('<Q', data[offset+16:offset+24])[0]
    p_filesz = struct.unpack('<Q', data[offset+32:offset+40])[0]
    p_memsz = struct.unpack('<Q', data[offset+40:offset+48])[0]
    
    if p_type == 1:  # LOAD
        if p_vaddr <= 0x11D130 < p_vaddr + p_memsz:
            file_off = p_offset + (0x11D130 - p_vaddr)
            print(f"  vaddr 0x11D130 is in segment at vaddr 0x{p_vaddr:X}")
            print(f"  File offset would be: 0x{file_off:X}")
            print(f"  Bytes at that offset: {data[file_off:file_off+32].hex()}")
            
            # Decode as DYNAMIC entries
            print("\n  Decoded as DYNAMIC:")
            for j in range(4):
                dyn_off = file_off + j * 16
                d_tag = struct.unpack('<Q', data[dyn_off:dyn_off+8])[0]
                d_val = struct.unpack('<Q', data[dyn_off+8:dyn_off+16])[0]
                tag_name = DT_TAGS.get(d_tag, f'0x{d_tag:X}')
                print(f"    DT[{j}]: tag={tag_name:15} val=0x{d_val:X}")
