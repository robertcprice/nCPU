/*
 * httpd.c -- Minimal HTTP/1.0 server for freestanding ARM64 Metal GPU.
 *
 * Serves files from the GPU filesystem over emulated TCP sockets.
 * One connection at a time, close-after-response (HTTP/1.0 semantics).
 *
 * Compile: aarch64-elf-gcc -nostdlib -ffreestanding -static -O2
 *          -march=armv8-a -mgeneral-regs-only -T demos/arm64.ld
 *          -I demos -e _start demos/arm64_start.S demos/net/httpd.c
 *          -o /tmp/httpd.elf
 */

#ifdef __CCGPU__
#include "arm64_selfhost.h"
#else
#include "arm64_libc.h"
#endif

/* ======================================================================== */
/* CONFIGURATION                                                            */
/* ======================================================================== */

#define HTTP_PORT      8080
#define LISTEN_BACKLOG 4
#define REQ_BUF_SIZE   2048
#define FILE_BUF_SIZE  4096
#define HDR_BUF_SIZE   512
#define PATH_MAX_LEN   256

/* AF_INET = 2, SOCK_STREAM = 1 */
#define AF_INET      2
#define SOCK_STREAM  1

/* ======================================================================== */
/* CONTENT-TYPE DETECTION                                                   */
/* ======================================================================== */

/*
 * Determine MIME type from file extension.
 * Checks the last '.' in the path and matches known extensions.
 * Falls back to text/plain for unknown or missing extensions.
 */
static const char *detect_content_type(const char *path) {
    const char *dot = NULL;
    const char *p = path;

    /* Find the last dot in the path */
    while (*p) {
        if (*p == '.') dot = p;
        p++;
    }

    if (!dot) return "text/plain";

    if (strcmp(dot, ".html") == 0 || strcmp(dot, ".htm") == 0)
        return "text/html";
    if (strcmp(dot, ".txt") == 0)
        return "text/plain";
    if (strcmp(dot, ".json") == 0)
        return "application/json";
    if (strcmp(dot, ".css") == 0)
        return "text/css";
    if (strcmp(dot, ".js") == 0)
        return "application/javascript";
    if (strcmp(dot, ".xml") == 0)
        return "application/xml";
    if (strcmp(dot, ".csv") == 0)
        return "text/csv";
    if (strcmp(dot, ".svg") == 0)
        return "image/svg+xml";
    if (strcmp(dot, ".ico") == 0)
        return "image/x-icon";
    if (strcmp(dot, ".png") == 0)
        return "image/png";
    if (strcmp(dot, ".jpg") == 0 || strcmp(dot, ".jpeg") == 0)
        return "image/jpeg";
    if (strcmp(dot, ".gif") == 0)
        return "image/gif";

    return "text/plain";
}

/* ======================================================================== */
/* HTTP REQUEST PARSING                                                     */
/* ======================================================================== */

/*
 * Extract the request path from an HTTP request line.
 * Expects "GET /path HTTP/1.x\r\n..." format.
 * Writes the path into `out_path` (null-terminated).
 * Returns 0 on success, -1 if the request is malformed or not a GET.
 */
static int parse_request_path(const char *req, char *out_path, int max_len) {
    /* Must start with "GET " */
    if (strncmp(req, "GET ", 4) != 0)
        return -1;

    const char *start = req + 4;

    /* Path must begin with '/' */
    if (*start != '/')
        return -1;

    /* Find the end of the path (space before HTTP/1.x) */
    const char *end = start;
    while (*end && *end != ' ' && *end != '\r' && *end != '\n')
        end++;

    int len = (int)(end - start);
    if (len <= 0 || len >= max_len)
        return -1;

    memcpy(out_path, start, len);
    out_path[len] = '\0';

    return 0;
}

/* ======================================================================== */
/* HTTP RESPONSE HELPERS                                                    */
/* ======================================================================== */

/*
 * Send a complete HTTP/1.0 response with headers and body.
 * Builds the header in a stack buffer, then sends header + body.
 */
static void send_response(int client_fd, int status_code,
                          const char *status_text, const char *content_type,
                          const char *body, int body_len) {
    char hdr[HDR_BUF_SIZE];
    int hdr_len;

    hdr_len = snprintf(hdr, sizeof(hdr),
        "HTTP/1.0 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %d\r\n"
        "Connection: close\r\n"
        "\r\n",
        status_code, status_text, content_type, body_len);

    send(client_fd, hdr, hdr_len);

    if (body && body_len > 0)
        send(client_fd, body, body_len);
}

/*
 * Send a fixed 404 Not Found response.
 */
static void send_404(int client_fd, const char *path) {
    const char *body = "<!DOCTYPE html>\n"
        "<html><head><title>404 Not Found</title></head>\n"
        "<body><h1>404 Not Found</h1>\n"
        "<p>The requested resource was not found on this server.</p>\n"
        "</body></html>\n";
    int body_len = strlen(body);

    send_response(client_fd, 404, "Not Found", "text/html", body, body_len);
    printf("GET %s -> 404\n", path);
}

/*
 * Send a fixed 400 Bad Request response.
 */
static void send_400(int client_fd) {
    const char *body = "Bad Request";
    send_response(client_fd, 400, "Bad Request", "text/plain",
                  body, strlen(body));
    printf("(malformed request) -> 400\n");
}

/*
 * Send a fixed 500 Internal Server Error response.
 */
static void send_500(int client_fd, const char *path) {
    const char *body = "Internal Server Error";
    send_response(client_fd, 500, "Internal Server Error", "text/plain",
                  body, strlen(body));
    printf("GET %s -> 500\n", path);
}

/* ======================================================================== */
/* FILE SERVING                                                             */
/* ======================================================================== */

/*
 * Read a file from the GPU filesystem and send it as an HTTP response.
 *
 * Strategy: read the entire file into a malloc'd buffer (GPU filesystem
 * files are small), then send as a single response. This avoids needing
 * chunked transfer encoding.
 *
 * Returns 0 on success, -1 on failure.
 */
static int serve_file(int client_fd, const char *path) {
    int fd = open(path, O_RDONLY);
    if (fd < 0)
        return -1;

    /*
     * Determine file size by reading into a dynamically grown buffer.
     * The GPU filesystem does not support lseek-to-end, so we read
     * in chunks and accumulate.
     */
    int capacity = FILE_BUF_SIZE;
    int total = 0;
    char *buf = (char *)malloc(capacity);
    if (!buf) {
        close(fd);
        return -1;
    }

    ssize_t n;
    while ((n = read(fd, buf + total, capacity - total)) > 0) {
        total += (int)n;
        if (total >= capacity) {
            /* Grow the buffer */
            int new_cap = capacity * 2;
            char *new_buf = (char *)realloc(buf, new_cap);
            if (!new_buf) {
                free(buf);
                close(fd);
                return -1;
            }
            buf = new_buf;
            capacity = new_cap;
        }
    }
    close(fd);

    /* Detect content type and send */
    const char *ctype = detect_content_type(path);
    send_response(client_fd, 200, "OK", ctype, buf, total);

    printf("GET %s -> 200 (%d bytes)\n", path, total);

    free(buf);
    return 0;
}

/* ======================================================================== */
/* REQUEST HANDLER                                                          */
/* ======================================================================== */

/*
 * Handle a single HTTP/1.0 request on an accepted connection.
 *
 * 1. Receive the request into a buffer.
 * 2. Parse the request method and path.
 * 3. Map "/" to "/index.html".
 * 4. Attempt to serve the file from the GPU filesystem.
 * 5. Send 404 if the file does not exist.
 * 6. Close the connection (HTTP/1.0 — no keep-alive).
 */
static void handle_connection(int client_fd) {
    char req_buf[REQ_BUF_SIZE];
    char path[PATH_MAX_LEN];
    char resolved[PATH_MAX_LEN];

    /* Receive the request */
    ssize_t n = recv(client_fd, req_buf, sizeof(req_buf) - 1);
    if (n <= 0) {
        close(client_fd);
        return;
    }
    req_buf[n] = '\0';

    /* Parse the GET path */
    if (parse_request_path(req_buf, path, sizeof(path)) != 0) {
        send_400(client_fd);
        close(client_fd);
        return;
    }

    /* Map "/" to "/index.html" */
    if (strcmp(path, "/") == 0) {
        strcpy(resolved, "/index.html");
    } else {
        strcpy(resolved, path);
    }

    /* Serve the file or send 404 */
    if (serve_file(client_fd, resolved) != 0) {
        send_404(client_fd, resolved);
    }

    close(client_fd);
}

/* ======================================================================== */
/* SERVER SETUP                                                             */
/* ======================================================================== */

/*
 * Create a TCP listener socket bound to 0.0.0.0:port.
 * Returns the listening socket fd, or -1 on error.
 */
static int create_listener(int port) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        printf("httpd: socket() failed\n");
        return -1;
    }

    if (bind(fd, 0, port) < 0) {
        printf("httpd: bind() failed on port %d\n", port);
        close(fd);
        return -1;
    }

    if (listen(fd, LISTEN_BACKLOG) < 0) {
        printf("httpd: listen() failed\n");
        close(fd);
        return -1;
    }

    return fd;
}

/* ======================================================================== */
/* MAIN                                                                     */
/* ======================================================================== */

int main(void) {
    printf("========================================\n");
    printf("  GPU HTTP Server (ARM64 Metal)\n");
    printf("========================================\n");
    printf("Listening on port %d\n", HTTP_PORT);
    printf("Document root: /\n");
    printf("Default file:  /index.html\n");
    printf("----------------------------------------\n");

    int listen_fd = create_listener(HTTP_PORT);
    if (listen_fd < 0) {
        printf("httpd: failed to start server\n");
        return 1;
    }

    printf("httpd: ready, waiting for connections...\n\n");

    /* Accept loop — one connection at a time (HTTP/1.0) */
    while (1) {
        int client_fd = accept(listen_fd);
        if (client_fd < 0) {
            printf("httpd: accept() failed, retrying...\n");
            continue;
        }

        handle_connection(client_fd);
    }

    /* Unreachable — server runs until killed */
    close(listen_fd);
    return 0;
}
