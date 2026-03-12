"""Tests for shared ELF symbol parsing used by the GPU debugging tools."""

import importlib.util
import struct
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ELF_LOADER_PATH = PROJECT_ROOT / "ncpu/os/gpu/elf_loader.py"
_SPEC = importlib.util.spec_from_file_location("elf_loader_local", ELF_LOADER_PATH)
_ELF_LOADER = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_ELF_LOADER)

format_symbolized_address = _ELF_LOADER.format_symbolized_address
lookup_function_symbol = _ELF_LOADER.lookup_function_symbol
parse_elf_function_symbols = _ELF_LOADER.parse_elf_function_symbols


def _write_test_elf(path: Path) -> None:
    """Write a tiny ELF64 file with a strtab and symtab."""
    strtab = b"\x00foo\x00bar\x00"
    symtab_offset = 0xC0
    strtab_offset = 0x80
    shoff = 0x100
    shentsize = 64
    shnum = 3

    data = bytearray(0x100 + shentsize * shnum)
    data[:4] = b"\x7fELF"
    data[4] = 2  # ELF64
    data[5] = 1  # little-endian
    struct.pack_into("<Q", data, 40, shoff)
    struct.pack_into("<H", data, 58, shentsize)
    struct.pack_into("<H", data, 60, shnum)

    data[strtab_offset:strtab_offset + len(strtab)] = strtab

    # Null symbol.
    sym0 = symtab_offset
    struct.pack_into("<IBBHQQ", data, sym0, 0, 0, 0, 0, 0, 0)
    # foo @ 0x1000 size 0x40
    sym1 = symtab_offset + 24
    struct.pack_into("<IBBHQQ", data, sym1, 1, 0x12, 0, 1, 0x1000, 0x40)
    # bar @ 0x1040 size 0 (common for minimal symbol tables)
    sym2 = symtab_offset + 48
    struct.pack_into("<IBBHQQ", data, sym2, 5, 0x12, 0, 1, 0x1040, 0)

    # Section 1: strtab
    sh1 = shoff + shentsize
    struct.pack_into("<I", data, sh1 + 4, 3)  # SHT_STRTAB
    struct.pack_into("<Q", data, sh1 + 24, strtab_offset)
    struct.pack_into("<Q", data, sh1 + 32, len(strtab))

    # Section 2: symtab
    sh2 = shoff + shentsize * 2
    struct.pack_into("<I", data, sh2 + 4, 2)  # SHT_SYMTAB
    struct.pack_into("<Q", data, sh2 + 24, symtab_offset)
    struct.pack_into("<Q", data, sh2 + 32, 72)  # 3 entries
    struct.pack_into("<I", data, sh2 + 40, 1)   # link -> strtab
    struct.pack_into("<Q", data, sh2 + 56, 24)  # Elf64_Sym size

    path.write_bytes(data)


def test_parse_elf_function_symbols_extracts_function_entries(tmp_path):
    elf_path = tmp_path / "symbols.elf"
    _write_test_elf(elf_path)

    symbols = parse_elf_function_symbols(elf_path)

    assert symbols == {
        0x1000: ("foo", 0x40),
        0x1040: ("bar", 0),
    }


def test_lookup_function_symbol_returns_name_and_offset():
    symbols = {
        0x1000: ("foo", 0x40),
        0x1040: ("bar", 0),
    }

    assert lookup_function_symbol(0x1018, symbols) == ("foo", 0x18)
    assert lookup_function_symbol(0x1048, symbols) == ("bar", 0x8)
    assert lookup_function_symbol(0x0FF0, symbols) is None


def test_format_symbolized_address_falls_back_to_raw_pc():
    symbols = {0x1000: ("foo", 0x40)}

    assert format_symbolized_address(0x1000, symbols) == "foo (0x00001000)"
    assert format_symbolized_address(0x1010, symbols) == "foo+0x10 (0x00001010)"
    assert format_symbolized_address(0x2000, symbols) == "0x00002000"
