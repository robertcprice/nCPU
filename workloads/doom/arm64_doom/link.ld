/* Linker script for Neural CPU DOOM */

ENTRY(_start)

MEMORY
{
    CODE (rx)  : ORIGIN = 0x10000, LENGTH = 64K
    DATA (rw)  : ORIGIN = 0x80000, LENGTH = 64K
}

SECTIONS
{
    .text : {
        *(.text*)
        *(.rodata*)
    } > CODE

    .data : {
        *(.data*)
    } > DATA

    .bss : {
        *(.bss*)
    } > DATA
}
