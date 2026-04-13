# Network Socket Support

## Overview

The nCPU GPU execution environment now supports standard Linux TCP socket operations, enabling real network communication from musl-compiled programs running on the GPU.

## Implemented Syscalls

| Syscall | Number | Description |
|---------|--------|-------------|
| socket | 198 | Create a new socket (AF_INET + SOCK_STREAM) |
| bind | 200 | Bind socket to address/port |
| listen | 201/202 | Listen for connections |
| connect | 201 | Connect to remote address |
| accept | 203 | Accept incoming connection |

## Usage

Compile C programs with musl libc for aarch64:
```bash
aarch64-linux-musl-gcc -static program.c -o program
```

Run on GPU:
```python
from ncpu.os.gpu.rust_backend import run_elf
result = run_elf('/path/to/program')
```

## Example

```c
#include <stdio.h>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>

int main() {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) return 1;

    struct sockaddr_in addr = {0};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(8080);
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");

    bind(sock, (struct sockaddr *)&addr, sizeof(addr));
    listen(sock, 1);
    // Accept connections...
    close(sock);
    return 0;
}
```

## Notes

- Only IPv4 TCP (AF_INET + SOCK_STREAM) is supported
- Sockets bridge to the host's network stack
- Read/write on sockets uses the VFS layer
- UDP support can be added similarly
