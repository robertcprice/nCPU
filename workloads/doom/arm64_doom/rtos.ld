/* Neural RTOS Linker Script */

ENTRY(_start)

MEMORY
{
    CODE (rx)  : ORIGIN = 0x10000, LENGTH = 64K
    DATA (rw)  : ORIGIN = 0x20000, LENGTH = 64K
    STACK (rw) : ORIGIN = 0x70000, LENGTH = 64K
}

SECTIONS
{
    .text : {
        *(.text._start)
        *(.text*)
    } > CODE

    .rodata : {
        *(.rodata*)
    } > CODE

    .data : {
        *(.data*)
    } > DATA

    .bss : {
        *(.bss*)
        *(COMMON)
    } > DATA

    /* Stack pointer at top of stack region */
    _stack_top = ORIGIN(STACK) + LENGTH(STACK);
}
