"""
Alpine Linux Root Filesystem -- Build a complete Alpine v3.20 rootfs for GPU.

Creates a GPUFilesystem populated with the Alpine Linux v3.20 FHS directory
hierarchy, identity files, user databases, system configuration, synthetic
/proc filesystem entries, /dev device nodes, /etc service databases, user
home directories, init system stubs, shell scripts, and data files.

The resulting filesystem is served to a real BusyBox binary (Alpine's core
userspace) running on the Metal GPU compute shader via SVC-trapped syscalls.

Author: Robert Price
Date: March 2026
"""

from ncpu.os.gpu.filesystem import GPUFilesystem


def create_alpine_rootfs() -> GPUFilesystem:
    """Build a complete Alpine Linux v3.20 filesystem for GPU execution.

    Creates the standard Alpine FHS directory layout, populates identity
    files (/etc/os-release, /etc/alpine-release), user databases
    (/etc/passwd, /etc/group, /etc/shadow), system configuration, synthetic
    /proc entries, /dev device nodes, networking configuration, init stubs,
    shell scripts, and data files.

    Returns:
        GPUFilesystem populated with Alpine Linux v3.20 rootfs.
    """
    fs = GPUFilesystem()

    # ===================================================================
    # DIRECTORY HIERARCHY (Alpine FHS)
    # ===================================================================

    alpine_dirs = [
        # Top-level FHS
        "/bin", "/sbin",
        "/usr/bin", "/usr/sbin", "/usr/lib", "/usr/share",
        "/etc", "/etc/init.d", "/etc/apk",
        "/home", "/root",
        "/tmp",
        "/var/log", "/var/run",
        "/proc", "/sys", "/dev",
        "/lib",
        "/run",
        "/mnt", "/media",
        "/opt", "/srv",

        # Extended /etc
        "/etc/ssl", "/etc/ssl/certs",
        "/etc/network",
        "/etc/profile.d",
        "/etc/skel",
        "/etc/runlevels", "/etc/runlevels/default",
        "/etc/runlevels/boot", "/etc/runlevels/sysinit",
        "/etc/runlevels/shutdown",
        "/etc/conf.d",
        "/etc/local.d",

        # /dev entries
        "/dev/pts",
        "/dev/shm",

        # /proc extended
        "/proc/self",
        "/proc/1",
        "/proc/sys", "/proc/sys/kernel",
        "/proc/net",

        # /usr extended
        "/usr/share/misc",
        "/usr/share/terminfo",
        "/usr/share/terminfo/x",
        "/usr/share/terminfo/v",
        "/usr/share/terminfo/l",
        "/usr/share/man",
        "/usr/local", "/usr/local/bin", "/usr/local/lib",

        # /var extended
        "/var/cache", "/var/cache/apk",
        "/var/lib",
        "/var/spool", "/var/spool/cron",
        "/var/tmp",
    ]

    for d in alpine_dirs:
        fs.directories.add(d)

    # ===================================================================
    # IDENTITY FILES
    # ===================================================================

    fs.write_file("/etc/alpine-release", "3.20.0\n")

    fs.write_file("/etc/os-release", (
        'NAME="Alpine Linux"\n'
        'ID=alpine\n'
        'VERSION_ID=3.20.0\n'
        'PRETTY_NAME="Alpine Linux v3.20 (nCPU GPU)"\n'
        'HOME_URL="https://alpinelinux.org/"\n'
        'BUG_REPORT_URL="https://gitlab.alpinelinux.org/alpine/aports/-/issues"\n'
    ))

    fs.write_file("/etc/hostname", "ncpu-gpu\n")

    fs.write_file("/etc/hosts", "127.0.0.1\tlocalhost ncpu-gpu\n")

    # ===================================================================
    # USER DATABASE
    # ===================================================================

    fs.write_file("/etc/passwd", (
        "root:x:0:0:root:/root:/bin/ash\n"
        "bin:x:1:1:bin:/bin:/sbin/nologin\n"
        "daemon:x:2:2:daemon:/sbin:/sbin/nologin\n"
        "adm:x:3:4:adm:/var/adm:/sbin/nologin\n"
        "lp:x:4:7:lp:/var/spool/lpd:/sbin/nologin\n"
        "sync:x:5:0:sync:/sbin:/bin/sync\n"
        "shutdown:x:6:0:shutdown:/sbin:/sbin/shutdown\n"
        "halt:x:7:0:halt:/sbin:/sbin/halt\n"
        "mail:x:8:12:mail:/var/mail:/sbin/nologin\n"
        "news:x:9:13:news:/usr/lib/news:/sbin/nologin\n"
        "uucp:x:10:14:uucp:/var/spool/uucp:/sbin/nologin\n"
        "operator:x:11:0:operator:/root:/sbin/nologin\n"
        "man:x:13:15:man:/usr/man:/sbin/nologin\n"
        "postmaster:x:14:12:postmaster:/var/mail:/sbin/nologin\n"
        "cron:x:16:16:cron:/var/spool/cron:/sbin/nologin\n"
        "ftp:x:21:21::/var/lib/ftp:/sbin/nologin\n"
        "sshd:x:22:22:sshd:/dev/null:/sbin/nologin\n"
        "at:x:25:25:at:/var/spool/cron/atjobs:/sbin/nologin\n"
        "squid:x:31:31:Squid:/var/cache/squid:/sbin/nologin\n"
        "xfs:x:33:33:X Font Server:/etc/X11/fs:/sbin/nologin\n"
        "games:x:35:35:games:/usr/games:/sbin/nologin\n"
        "cyrus:x:85:12::/usr/cyrus:/sbin/nologin\n"
        "vpopmail:x:89:89::/var/vpopmail:/sbin/nologin\n"
        "ntp:x:123:123:NTP:/var/empty:/sbin/nologin\n"
        "smmsp:x:209:209:smmsp:/var/spool/mqueue:/sbin/nologin\n"
        "guest:x:405:100:guest:/dev/null:/sbin/nologin\n"
        "nobody:x:65534:65534:nobody:/:/sbin/nologin\n"
    ))

    fs.write_file("/etc/group", (
        "root:x:0:root\n"
        "bin:x:1:root,bin,daemon\n"
        "daemon:x:2:root,bin,daemon\n"
        "sys:x:3:root,bin,adm\n"
        "adm:x:4:root,adm,daemon\n"
        "tty:x:5:\n"
        "disk:x:6:root,adm\n"
        "lp:x:7:lp\n"
        "mem:x:8:\n"
        "kmem:x:9:\n"
        "wheel:x:10:root\n"
        "floppy:x:11:root\n"
        "mail:x:12:mail\n"
        "news:x:13:news\n"
        "uucp:x:14:uucp\n"
        "man:x:15:man\n"
        "cron:x:16:cron\n"
        "console:x:17:\n"
        "audio:x:18:\n"
        "cdrom:x:19:\n"
        "dialout:x:20:root\n"
        "ftp:x:21:\n"
        "sshd:x:22:\n"
        "input:x:23:\n"
        "at:x:25:at\n"
        "tape:x:26:root\n"
        "video:x:27:root\n"
        "netdev:x:28:\n"
        "readproc:x:30:\n"
        "squid:x:31:squid\n"
        "xfs:x:33:xfs\n"
        "kvm:x:34:\n"
        "games:x:35:\n"
        "shadow:x:42:\n"
        "utmp:x:43:\n"
        "ping:x:999:\n"
        "nogroup:x:65534:\n"
    ))

    fs.write_file("/etc/shadow", (
        "root:!::0:::::\n"
        "bin:!::0:::::\n"
        "daemon:!::0:::::\n"
        "adm:!::0:::::\n"
        "lp:!::0:::::\n"
        "sync:!::0:::::\n"
        "shutdown:!::0:::::\n"
        "halt:!::0:::::\n"
        "mail:!::0:::::\n"
        "news:!::0:::::\n"
        "uucp:!::0:::::\n"
        "operator:!::0:::::\n"
        "man:!::0:::::\n"
        "postmaster:!::0:::::\n"
        "cron:!::0:::::\n"
        "ftp:!::0:::::\n"
        "sshd:!::0:::::\n"
        "at:!::0:::::\n"
        "squid:!::0:::::\n"
        "xfs:!::0:::::\n"
        "games:!::0:::::\n"
        "cyrus:!::0:::::\n"
        "vpopmail:!::0:::::\n"
        "ntp:!::0:::::\n"
        "smmsp:!::0:::::\n"
        "guest:!::0:::::\n"
        "nobody:!::0:::::\n"
    ))

    fs.write_file("/etc/shells", (
        "/bin/ash\n"
        "/bin/sh\n"
    ))

    # ===================================================================
    # SYSTEM CONFIGURATION
    # ===================================================================

    fs.write_file("/etc/profile", (
        "export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\n"
        "export HOME=/root\n"
        "export PS1='\\u@\\h:\\w# '\n"
        "export CHARSET=UTF-8\n"
        "export PAGER=less\n"
        "export LANG=C.UTF-8\n"
        "export ENV=/etc/shinit\n"
        "\n"
        "# Source profile.d scripts\n"
        "for script in /etc/profile.d/*.sh; do\n"
        "    [ -r \"$script\" ] && . \"$script\"\n"
        "done\n"
        "unset script\n"
    ))

    fs.write_file("/etc/resolv.conf", "nameserver 127.0.0.1\n")

    fs.write_file("/etc/motd", (
        "\n"
        "Welcome to Alpine Linux v3.20 (nCPU GPU)\n"
        "Running on Apple Silicon Metal compute shader\n"
        "\n"
    ))

    fs.write_file("/etc/nsswitch.conf", (
        "# /etc/nsswitch.conf\n"
        "passwd:  files\n"
        "shadow:  files\n"
        "group:   files\n"
        "hosts:   files dns\n"
        "networks: files\n"
        "services: files\n"
        "protocols: files\n"
        "rpc:     files\n"
        "ethers:  files\n"
        "netmasks: files\n"
        "netgroup: files\n"
        "bootparams: files\n"
        "automount: files\n"
        "aliases: files\n"
    ))

    fs.write_file("/etc/ld.so.conf", (
        "/usr/local/lib\n"
        "/usr/lib\n"
        "/lib\n"
    ))

    fs.write_file("/etc/environment", (
        "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\n"
        "LANG=C.UTF-8\n"
    ))

    fs.write_file("/etc/inputrc", (
        "# /etc/inputrc - readline configuration\n"
        "set bell-style none\n"
        "set meta-flag on\n"
        "set input-meta on\n"
        "set convert-meta off\n"
        "set output-meta on\n"
        "set colored-stats on\n"
        "set visible-stats on\n"
        "set mark-symlinked-directories on\n"
        "set colored-completion-prefix on\n"
        "set show-all-if-ambiguous on\n"
        '# Mappings for "page up" and "page down" to search history\n'
        '"\\e[5~": history-search-backward\n'
        '"\\e[6~": history-search-forward\n'
        "# Ctrl-left/right to move by words\n"
        '"\\e[1;5C": forward-word\n'
        '"\\e[1;5D": backward-word\n'
    ))

    fs.write_file("/etc/login.defs", (
        "# /etc/login.defs - login configuration\n"
        "MAIL_DIR\t\t/var/mail\n"
        "FAILLOG_ENAB\t\tyes\n"
        "LOG_UNKFAIL_ENAB\tno\n"
        "LOG_OK_LOGINS\t\tno\n"
        "SYSLOG_SU_ENAB\t\tyes\n"
        "SYSLOG_SG_ENAB\t\tyes\n"
        "FTMP_FILE\t\t/var/log/btmp\n"
        "SU_NAME\t\t\tsu\n"
        "HUSHLOGIN_FILE\t\t.hushlogin\n"
        "ENV_SUPATH\t\tPATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\n"
        "ENV_PATH\t\tPATH=/usr/local/bin:/usr/bin:/bin\n"
        "TTYGROUP\t\ttty\n"
        "TTYPERM\t\t\t0600\n"
        "ERASECHAR\t\t0177\n"
        "KILLCHAR\t\t025\n"
        "UMASK\t\t\t022\n"
        "PASS_MAX_DAYS\t\t99999\n"
        "PASS_MIN_DAYS\t\t0\n"
        "PASS_WARN_AGE\t\t7\n"
        "UID_MIN\t\t\t1000\n"
        "UID_MAX\t\t\t60000\n"
        "GID_MIN\t\t\t1000\n"
        "GID_MAX\t\t\t60000\n"
        "CREATE_HOME\t\tyes\n"
        "USERGROUPS_ENAB\t\tyes\n"
        "ENCRYPT_METHOD\t\tSHA512\n"
    ))

    fs.write_file("/etc/securetty", (
        "console\n"
        "tty1\n"
        "tty2\n"
        "tty3\n"
        "tty4\n"
        "tty5\n"
        "tty6\n"
        "ttyS0\n"
        "ttyS1\n"
        "ttyAMA0\n"
        "hvc0\n"
    ))

    fs.write_file("/etc/crontab", (
        "# /etc/crontab - system crontab\n"
        "SHELL=/bin/ash\n"
        "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\n"
        "\n"
        "# m h dom mon dow user command\n"
        "*/15 * * * * root run-parts /etc/periodic/15min\n"
        "0 * * * * root run-parts /etc/periodic/hourly\n"
        "0 2 * * * root run-parts /etc/periodic/daily\n"
        "0 3 * * 6 root run-parts /etc/periodic/weekly\n"
        "0 5 1 * * root run-parts /etc/periodic/monthly\n"
    ))

    fs.write_file("/etc/network/interfaces", (
        "# /etc/network/interfaces\n"
        "auto lo\n"
        "iface lo inet loopback\n"
        "\n"
        "auto eth0\n"
        "iface eth0 inet dhcp\n"
        "    hostname ncpu-gpu\n"
    ))

    # ===================================================================
    # /etc/protocols -- standard IANA protocol numbers
    # ===================================================================

    fs.write_file("/etc/protocols", (
        "# /etc/protocols - Internet (IP) protocols\n"
        "ip\t0\tIP\t\t# internet protocol, pseudo protocol number\n"
        "hopopt\t0\tHOPOPT\t\t# hop-by-hop options for ipv6\n"
        "icmp\t1\tICMP\t\t# internet control message protocol\n"
        "igmp\t2\tIGMP\t\t# internet group management protocol\n"
        "ggp\t3\tGGP\t\t# gateway-gateway protocol\n"
        "ipencap\t4\tIP-ENCAP\t# IP encapsulated in IP\n"
        "st\t5\tST\t\t# ST datagram mode\n"
        "tcp\t6\tTCP\t\t# transmission control protocol\n"
        "egp\t8\tEGP\t\t# exterior gateway protocol\n"
        "igp\t9\tIGP\t\t# interior gateway protocol\n"
        "pup\t12\tPUP\t\t# PARC universal packet protocol\n"
        "udp\t17\tUDP\t\t# user datagram protocol\n"
        "hmp\t20\tHMP\t\t# host monitoring protocol\n"
        "xns-idp\t22\tXNS-IDP\t\t# Xerox NS IDP\n"
        "rdp\t27\tRDP\t\t# reliable datagram protocol\n"
        "iso-tp4\t29\tISO-TP4\t\t# ISO transport protocol class 4\n"
        "dccp\t33\tDCCP\t\t# datagram congestion control protocol\n"
        "xtp\t36\tXTP\t\t# Xpress Transfer Protocol\n"
        "ddp\t37\tDDP\t\t# Datagram Delivery Protocol\n"
        "idpr-cmtp\t38\tIDPR-CMTP\t# IDPR Control Message Transport\n"
        "ipv6\t41\tIPv6\t\t# Internet Protocol version 6\n"
        "ipv6-route\t43\tIPv6-Route\t# Routing Header for IPv6\n"
        "ipv6-frag\t44\tIPv6-Frag\t# Fragment Header for IPv6\n"
        "idrp\t45\tIDRP\t\t# Inter-Domain Routing Protocol\n"
        "rsvp\t46\tRSVP\t\t# Resource ReSerVation Protocol\n"
        "gre\t47\tGRE\t\t# Generic Routing Encapsulation\n"
        "esp\t50\tIPSEC-ESP\t# Encap Security Payload\n"
        "ah\t51\tIPSEC-AH\t# Authentication Header\n"
        "skip\t57\tSKIP\t\t# SKIP\n"
        "ipv6-icmp\t58\tIPv6-ICMP\t# ICMP for IPv6\n"
        "ipv6-nonxt\t59\tIPv6-NoNxt\t# No Next Header for IPv6\n"
        "ipv6-opts\t60\tIPv6-Opts\t# Destination Options for IPv6\n"
        "rspf\t73\tRSPF\t\t# Radio Shortest Path First\n"
        "vmtp\t81\tVMTP\t\t# Versatile Message Transport\n"
        "eigrp\t88\tEIGRP\t\t# Enhanced Interior Routing Protocol\n"
        "ospf\t89\tOSPFIGP\t\t# Open Shortest Path First IGP\n"
        "ax.25\t93\tAX.25\t\t# AX.25 Frames\n"
        "ipip\t94\tIPIP\t\t# IP-within-IP Encapsulation Protocol\n"
        "etherip\t97\tETHERIP\t\t# Ethernet-within-IP Encapsulation\n"
        "encap\t98\tENCAP\t\t# RFC1241 encapsulation\n"
        "pim\t103\tPIM\t\t# Protocol Independent Multicast\n"
        "ipcomp\t108\tIPCOMP\t\t# IP Payload Compression Protocol\n"
        "vrrp\t112\tVRRP\t\t# Virtual Router Redundancy Protocol\n"
        "l2tp\t115\tL2TP\t\t# Layer Two Tunneling Protocol\n"
        "isis\t124\tISIS\t\t# IS-IS over IPv4\n"
        "sctp\t132\tSCTP\t\t# Stream Control Transmission Protocol\n"
        "fc\t133\tFC\t\t# Fibre Channel\n"
        "udplite\t136\tUDPLite\t\t# UDP-Lite\n"
        "mpls-in-ip\t137\tMPLS-in-IP\t# MPLS-in-IP\n"
        "manet\t138\tmanet\t\t# MANET Protocols\n"
        "hip\t139\tHIP\t\t# Host Identity Protocol\n"
        "shim6\t140\tShim6\t\t# Shim6 Protocol\n"
        "wesp\t141\tWESP\t\t# Wrapped Encapsulating Security Payload\n"
        "rohc\t142\tROHC\t\t# Robust Header Compression\n"
    ))

    # ===================================================================
    # /etc/services -- standard well-known ports (subset)
    # ===================================================================

    fs.write_file("/etc/services", (
        "# /etc/services - Internet network services list\n"
        "tcpmux\t\t1/tcp\n"
        "echo\t\t7/tcp\n"
        "echo\t\t7/udp\n"
        "discard\t\t9/tcp\t\tsink null\n"
        "discard\t\t9/udp\t\tsink null\n"
        "systat\t\t11/tcp\t\tusers\n"
        "daytime\t\t13/tcp\n"
        "daytime\t\t13/udp\n"
        "netstat\t\t15/tcp\n"
        "qotd\t\t17/tcp\t\tquote\n"
        "chargen\t\t19/tcp\t\tttytst source\n"
        "chargen\t\t19/udp\t\tttytst source\n"
        "ftp-data\t20/tcp\n"
        "ftp\t\t21/tcp\n"
        "ssh\t\t22/tcp\n"
        "telnet\t\t23/tcp\n"
        "smtp\t\t25/tcp\t\tmail\n"
        "time\t\t37/tcp\t\ttimserver\n"
        "time\t\t37/udp\t\ttimserver\n"
        "nameserver\t42/tcp\t\tname\n"
        "whois\t\t43/tcp\t\tnicname\n"
        "domain\t\t53/tcp\n"
        "domain\t\t53/udp\n"
        "bootps\t\t67/udp\n"
        "bootpc\t\t68/udp\n"
        "tftp\t\t69/udp\n"
        "gopher\t\t70/tcp\n"
        "finger\t\t79/tcp\n"
        "http\t\t80/tcp\t\twww\n"
        "kerberos\t88/tcp\t\tkerberos5 krb5\n"
        "kerberos\t88/udp\t\tkerberos5 krb5\n"
        "pop3\t\t110/tcp\t\tpop-3\n"
        "sunrpc\t\t111/tcp\t\tportmapper rpcbind\n"
        "sunrpc\t\t111/udp\t\tportmapper rpcbind\n"
        "auth\t\t113/tcp\t\tident tap\n"
        "nntp\t\t119/tcp\t\treadnews untp\n"
        "ntp\t\t123/udp\n"
        "epmap\t\t135/tcp\t\tmsrpc\n"
        "netbios-ns\t137/udp\n"
        "netbios-dgm\t138/udp\n"
        "netbios-ssn\t139/tcp\n"
        "imap\t\t143/tcp\t\timap2\n"
        "snmp\t\t161/udp\n"
        "snmptrap\t162/udp\n"
        "bgp\t\t179/tcp\n"
        "irc\t\t194/tcp\n"
        "ldap\t\t389/tcp\n"
        "https\t\t443/tcp\n"
        "microsoft-ds\t445/tcp\n"
        "kpasswd\t\t464/tcp\n"
        "submissions\t465/tcp\t\tsmtps ssmtp\n"
        "syslog\t\t514/udp\n"
        "printer\t\t515/tcp\t\tspooler\n"
        "route\t\t520/udp\t\trouter routed\n"
        "rtsp\t\t554/tcp\n"
        "ipp\t\t631/tcp\n"
        "ldaps\t\t636/tcp\n"
        "rsync\t\t873/tcp\n"
        "ftps-data\t989/tcp\n"
        "ftps\t\t990/tcp\n"
        "imaps\t\t993/tcp\n"
        "pop3s\t\t995/tcp\n"
        "socks\t\t1080/tcp\n"
        "openvpn\t\t1194/tcp\n"
        "openvpn\t\t1194/udp\n"
        "ms-sql-s\t1433/tcp\n"
        "oracle\t\t1521/tcp\n"
        "mqtt\t\t1883/tcp\n"
        "nfs\t\t2049/tcp\n"
        "nfs\t\t2049/udp\n"
        "mysql\t\t3306/tcp\n"
        "rdp\t\t3389/tcp\t\tms-wbt-server\n"
        "svn\t\t3690/tcp\t\tsubversion\n"
        "sip\t\t5060/tcp\n"
        "sip\t\t5060/udp\n"
        "xmpp-client\t5222/tcp\t\tjabber-client\n"
        "xmpp-server\t5269/tcp\t\tjabber-server\n"
        "postgresql\t5432/tcp\n"
        "amqp\t\t5672/tcp\n"
        "vnc\t\t5900/tcp\n"
        "x11\t\t6000/tcp\n"
        "redis\t\t6379/tcp\n"
        "http-alt\t8080/tcp\t\twebcache\n"
        "http-alt\t8443/tcp\n"
        "elasticsearch\t9200/tcp\n"
        "git\t\t9418/tcp\n"
        "memcached\t11211/tcp\n"
        "memcached\t11211/udp\n"
        "mongodb\t\t27017/tcp\n"
    ))

    # ===================================================================
    # PROC FILESYSTEM (synthetic)
    # ===================================================================

    fs.write_file("/proc/version",
        "Linux version 6.1.0-ncpu (gcc) #1 SMP aarch64 GNU/Linux\n"
    )

    fs.write_file("/proc/cpuinfo", (
        "processor\t: 0\n"
        "BogoMIPS\t: 48.00\n"
        "Features\t: fp asimd\n"
        "CPU implementer\t: 0x61\n"
        "CPU architecture: 8\n"
        "CPU variant\t: 0x1\n"
        "CPU part\t: 0x000\n"
        "CPU revision\t: 0\n"
        "\n"
        "Hardware\t: Apple Silicon (nCPU GPU)\n"
        "model name\t: Apple Silicon (nCPU GPU)\n"
    ))

    fs.write_file("/proc/meminfo", (
        "MemTotal:       16777216 kB\n"
        "MemFree:        16000000 kB\n"
        "MemAvailable:   15800000 kB\n"
        "Buffers:           65536 kB\n"
        "Cached:           524288 kB\n"
        "SwapCached:            0 kB\n"
        "Active:           262144 kB\n"
        "Inactive:         327680 kB\n"
        "Active(anon):     131072 kB\n"
        "Inactive(anon):        0 kB\n"
        "Active(file):     131072 kB\n"
        "Inactive(file):   327680 kB\n"
        "Unevictable:           0 kB\n"
        "Mlocked:               0 kB\n"
        "SwapTotal:             0 kB\n"
        "SwapFree:              0 kB\n"
        "Dirty:                 0 kB\n"
        "Writeback:             0 kB\n"
        "AnonPages:        131072 kB\n"
        "Mapped:            65536 kB\n"
        "Shmem:                 0 kB\n"
        "KReclaimable:      32768 kB\n"
        "Slab:              65536 kB\n"
        "SReclaimable:      32768 kB\n"
        "SUnreclaim:        32768 kB\n"
        "KernelStack:        4096 kB\n"
        "PageTables:         2048 kB\n"
        "NFS_Unstable:          0 kB\n"
        "Bounce:                0 kB\n"
        "WritebackTmp:          0 kB\n"
        "CommitLimit:     8388608 kB\n"
        "Committed_AS:     262144 kB\n"
        "VmallocTotal:   135290163200 kB\n"
        "VmallocUsed:       32768 kB\n"
        "VmallocChunk:          0 kB\n"
        "HardwareCorrupted:     0 kB\n"
        "AnonHugePages:         0 kB\n"
        "ShmemHugePages:        0 kB\n"
        "ShmemPmdMapped:        0 kB\n"
        "CmaTotal:              0 kB\n"
        "CmaFree:               0 kB\n"
        "HugePages_Total:       0\n"
        "HugePages_Free:        0\n"
        "HugePages_Rsvd:        0\n"
        "HugePages_Surp:        0\n"
        "Hugepagesize:       2048 kB\n"
        "Hugetlb:               0 kB\n"
    ))

    fs.write_file("/proc/uptime", "3600.00 3600.00\n")

    fs.write_file("/proc/loadavg", "0.00 0.00 0.00 1/1 1\n")

    fs.write_file("/proc/stat", (
        "cpu  100 0 50 3500 0 0 0 0 0 0\n"
        "cpu0 100 0 50 3500 0 0 0 0 0 0\n"
        "intr 0\n"
        "ctxt 0\n"
        "btime 1709900000\n"
        "processes 1\n"
        "procs_running 1\n"
        "procs_blocked 0\n"
        "softirq 0 0 0 0 0 0 0 0 0 0 0\n"
    ))

    fs.write_file("/proc/mounts", (
        "rootfs / rootfs rw 0 0\n"
        "proc /proc proc rw,nosuid,nodev,noexec,relatime 0 0\n"
        "sysfs /sys sysfs rw,nosuid,nodev,noexec,relatime 0 0\n"
        "devtmpfs /dev devtmpfs rw,nosuid,relatime,size=8388608k,nr_inodes=2097152,mode=755 0 0\n"
        "devpts /dev/pts devpts rw,nosuid,noexec,relatime,gid=5,mode=620,ptmxmode=666 0 0\n"
        "tmpfs /tmp tmpfs rw,nosuid,nodev 0 0\n"
        "tmpfs /run tmpfs rw,nosuid,nodev,mode=755 0 0\n"
        "tmpfs /dev/shm tmpfs rw,nosuid,nodev 0 0\n"
    ))

    fs.write_file("/proc/filesystems", (
        "nodev\tsysfs\n"
        "nodev\ttmpfs\n"
        "nodev\tbdev\n"
        "nodev\tproc\n"
        "nodev\tcgroup\n"
        "nodev\tcgroup2\n"
        "nodev\tcpuset\n"
        "nodev\tdevtmpfs\n"
        "nodev\tdevpts\n"
        "nodev\tsecurityfs\n"
        "nodev\tpstore\n"
        "nodev\tbpf\n"
        "nodev\tautofs\n"
        "\text4\n"
        "\text3\n"
        "\text2\n"
        "\tvfat\n"
        "\tiso9660\n"
    ))

    fs.write_file("/proc/devices", (
        "Character devices:\n"
        "  1 mem\n"
        "  4 /dev/vc/0\n"
        "  4 tty\n"
        "  4 ttyS\n"
        "  5 /dev/tty\n"
        "  5 /dev/console\n"
        "  5 /dev/ptmx\n"
        "  7 vcs\n"
        " 10 misc\n"
        " 13 input\n"
        " 29 fb\n"
        "136 pts\n"
        "180 usb\n"
        "189 usb_device\n"
        "204 ttyAMA\n"
        "226 drm\n"
        "240 hidraw\n"
        "254 rtc\n"
        "\n"
        "Block devices:\n"
        "  7 loop\n"
        "  8 sd\n"
        " 65 sd\n"
        "179 mmc\n"
        "253 virtblk\n"
        "254 mdp\n"
        "259 blkext\n"
    ))

    fs.write_file("/proc/partitions", (
        "major minor  #blocks  name\n"
        "   8        0   16777216 sda\n"
        "   8        1   16776192 sda1\n"
    ))

    fs.write_file("/proc/diskstats", (
        "   8       0 sda 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n"
        "   8       1 sda1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n"
    ))

    fs.write_file("/proc/cmdline", "console=ttyAMA0 root=/dev/sda1 rw\n")

    # /proc/self entries
    fs.write_file("/proc/self/status", (
        "Name:\tbusybox\n"
        "Umask:\t0022\n"
        "State:\tR (running)\n"
        "Tgid:\t1\n"
        "Ngid:\t0\n"
        "Pid:\t1\n"
        "PPid:\t0\n"
        "TracerPid:\t0\n"
        "Uid:\t0\t0\t0\t0\n"
        "Gid:\t0\t0\t0\t0\n"
        "FDSize:\t64\n"
        "Groups:\t0\n"
        "VmPeak:\t    4096 kB\n"
        "VmSize:\t    4096 kB\n"
        "VmLck:\t       0 kB\n"
        "VmPin:\t       0 kB\n"
        "VmHWM:\t    2048 kB\n"
        "VmRSS:\t    2048 kB\n"
        "VmData:\t     512 kB\n"
        "VmStk:\t     256 kB\n"
        "VmExe:\t     264 kB\n"
        "VmLib:\t       0 kB\n"
        "VmPTE:\t      16 kB\n"
        "VmSwap:\t       0 kB\n"
        "Threads:\t1\n"
        "SigQ:\t0/3067\n"
        "SigPnd:\t0000000000000000\n"
        "ShdPnd:\t0000000000000000\n"
        "SigBlk:\t0000000000000000\n"
        "SigIgn:\t0000000000000000\n"
        "SigCgt:\t0000000000000000\n"
        "CapInh:\t0000000000000000\n"
        "CapPrm:\t000001ffffffffff\n"
        "CapEff:\t000001ffffffffff\n"
        "CapBnd:\t000001ffffffffff\n"
        "CapAmb:\t0000000000000000\n"
        "Seccomp:\t0\n"
        "Cpus_allowed:\t1\n"
        "Cpus_allowed_list:\t0\n"
        "Mems_allowed:\t1\n"
        "Mems_allowed_list:\t0\n"
        "voluntary_ctxt_switches:\t0\n"
        "nonvoluntary_ctxt_switches:\t0\n"
    ))

    fs.write_file("/proc/self/maps", (
        "00010000-00060000 r-xp 00000000 08:01 1          /bin/busybox\n"
        "00060000-00070000 rw-p 00050000 08:01 1          /bin/busybox\n"
        "00070000-00090000 rw-p 00000000 00:00 0          [heap]\n"
        "7fffffe00000-7ffffffffff r-xp 00000000 00:00 0   [vdso]\n"
        "7ffffffde000-7ffffffff000 rw-p 00000000 00:00 0  [stack]\n"
    ))

    fs.write_file("/proc/self/cmdline", "busybox\x00ash\x00")

    fs.write_file("/proc/self/exe", "/bin/busybox\n")

    fs.write_file("/proc/self/cwd", "/\n")

    fs.write_file("/proc/self/environ",
        "HOME=/root\x00PATH=/usr/sbin:/usr/bin:/sbin:/bin\x00TERM=linux\x00"
    )

    fs.write_file("/proc/self/comm", "busybox\n")

    fs.write_file("/proc/self/limits", (
        "Limit                     Soft Limit           Hard Limit           Units\n"
        "Max cpu time              unlimited            unlimited            seconds\n"
        "Max file size             unlimited            unlimited            bytes\n"
        "Max data size             unlimited            unlimited            bytes\n"
        "Max stack size            8388608              unlimited            bytes\n"
        "Max core file size        0                    unlimited            bytes\n"
        "Max resident set          unlimited            unlimited            bytes\n"
        "Max processes             3067                 3067                 processes\n"
        "Max open files            1024                 1048576              files\n"
        "Max locked memory         65536                65536                bytes\n"
        "Max address space         unlimited            unlimited            bytes\n"
        "Max file locks            unlimited            unlimited            locks\n"
        "Max pending signals       3067                 3067                 signals\n"
        "Max msgqueue size         819200               819200               bytes\n"
        "Max nice priority         0                    0\n"
        "Max realtime priority     0                    0\n"
        "Max realtime timeout      unlimited            unlimited            us\n"
    ))

    # /proc/1 (init process)
    fs.write_file("/proc/1/status", (
        "Name:\tinit\n"
        "Umask:\t0022\n"
        "State:\tS (sleeping)\n"
        "Tgid:\t1\n"
        "Ngid:\t0\n"
        "Pid:\t1\n"
        "PPid:\t0\n"
        "TracerPid:\t0\n"
        "Uid:\t0\t0\t0\t0\n"
        "Gid:\t0\t0\t0\t0\n"
        "FDSize:\t64\n"
        "Groups:\t0\n"
        "Threads:\t1\n"
    ))

    fs.write_file("/proc/1/cmdline", "/sbin/init\x00")

    fs.write_file("/proc/1/comm", "init\n")

    # /proc/sys/kernel
    fs.write_file("/proc/sys/kernel/hostname", "ncpu-gpu\n")

    fs.write_file("/proc/sys/kernel/osrelease", "6.1.0-ncpu\n")

    fs.write_file("/proc/sys/kernel/ostype", "Linux\n")

    fs.write_file("/proc/sys/kernel/version",
        "#1 SMP PREEMPT aarch64\n"
    )

    fs.write_file("/proc/sys/kernel/domainname", "(none)\n")

    fs.write_file("/proc/sys/kernel/pid_max", "32768\n")

    fs.write_file("/proc/sys/kernel/threads-max", "3067\n")

    # /proc/net (empty but present)
    fs.write_file("/proc/net/tcp", (
        "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt"
        "   uid  timeout inode\n"
    ))

    fs.write_file("/proc/net/udp", (
        "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt"
        "   uid  timeout inode\n"
    ))

    fs.write_file("/proc/net/tcp6", (
        "  sl  local_address                         remote_address"
        "                        st tx_queue rx_queue tr tm->when retrnsmt"
        "   uid  timeout inode\n"
    ))

    fs.write_file("/proc/net/udp6", (
        "  sl  local_address                         remote_address"
        "                        st tx_queue rx_queue tr tm->when retrnsmt"
        "   uid  timeout inode\n"
    ))

    fs.write_file("/proc/net/dev", (
        "Inter-|   Receive                                                |  Transmit\n"
        " face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets"
        " errs drop fifo colls carrier compressed\n"
        "    lo:       0       0    0    0    0     0          0         0        0       0"
        "    0    0    0     0       0          0\n"
        "  eth0:       0       0    0    0    0     0          0         0        0       0"
        "    0    0    0     0       0          0\n"
    ))

    fs.write_file("/proc/net/arp", (
        "IP address       HW type     Flags       HW address            Mask     Device\n"
    ))

    # ===================================================================
    # /dev DEVICE NODES (regular files with special content)
    # ===================================================================

    fs.write_file("/dev/null", "")
    fs.write_file("/dev/zero", "")
    # Provide a small block of pseudo-random bytes as placeholder
    fs.write_file("/dev/urandom",
        b"\x7f\x45\x4c\x46\xde\xad\xbe\xef"
        b"\xca\xfe\xba\xbe\x01\x02\x03\x04"
        b"\x05\x06\x07\x08\x09\x0a\x0b\x0c"
        b"\x0d\x0e\x0f\x10\x11\x12\x13\x14"
        b"\x15\x16\x17\x18\x19\x1a\x1b\x1c"
        b"\x1d\x1e\x1f\x20\xa1\xa2\xa3\xa4"
        b"\xa5\xa6\xa7\xa8\xa9\xaa\xab\xac"
        b"\xad\xae\xaf\xb0\xb1\xb2\xb3\xb4"
    )
    fs.write_file("/dev/random", b"\x42" * 64)
    fs.write_file("/dev/tty", "")
    fs.write_file("/dev/console", "")
    fs.write_file("/dev/stdin", "")
    fs.write_file("/dev/stdout", "")
    fs.write_file("/dev/stderr", "")
    fs.write_file("/dev/ptmx", "")
    fs.write_file("/dev/full", "")
    fs.write_file("/dev/kmsg", "")

    # ===================================================================
    # APK PACKAGE MANAGER
    # ===================================================================

    fs.write_file("/etc/apk/world", (
        "alpine-base\n"
        "busybox\n"
        "musl\n"
    ))

    fs.write_file("/etc/apk/repositories",
        "https://dl-cdn.alpinelinux.org/alpine/v3.20/main\n"
        "https://dl-cdn.alpinelinux.org/alpine/v3.20/community\n"
    )

    fs.write_file("/etc/apk/arch", "aarch64\n")

    # ===================================================================
    # ADDITIONAL SYSTEM FILES
    # ===================================================================

    fs.write_file("/etc/fstab", (
        "# /etc/fstab: static file system information\n"
        "# <file sys>  <mount point>  <type>  <options>        <dump>  <pass>\n"
        "/dev/sda1     /              ext4    defaults,noatime 0       1\n"
        "proc          /proc          proc    defaults         0       0\n"
        "sysfs         /sys           sysfs   defaults         0       0\n"
        "devtmpfs      /dev           devtmpfs defaults        0       0\n"
        "tmpfs         /tmp           tmpfs   defaults,nosuid  0       0\n"
        "tmpfs         /run           tmpfs   defaults,mode=755 0      0\n"
    ))

    fs.write_file("/etc/inittab", (
        "::sysinit:/sbin/openrc sysinit\n"
        "::sysinit:/sbin/openrc boot\n"
        "::wait:/sbin/openrc default\n"
        "tty1::respawn:/sbin/getty 38400 tty1\n"
        "tty2::respawn:/sbin/getty 38400 tty2\n"
        "::ctrlaltdel:/sbin/reboot\n"
        "::shutdown:/sbin/openrc shutdown\n"
    ))

    fs.write_file("/etc/issue",
        "Alpine Linux v3.20 (nCPU GPU) \\n \\l\n\n"
    )

    fs.write_file("/etc/issue.net",
        "Alpine Linux v3.20 (nCPU GPU)\n"
    )

    fs.write_file("/etc/sysctl.conf", (
        "# /etc/sysctl.conf - sysctl configuration\n"
        "net.ipv4.ip_forward = 0\n"
        "net.ipv4.conf.all.accept_redirects = 0\n"
        "net.ipv4.conf.default.accept_redirects = 0\n"
        "net.ipv4.tcp_syncookies = 1\n"
        "kernel.panic = 10\n"
    ))

    # ===================================================================
    # /etc/ssl -- TLS/SSL configuration
    # ===================================================================

    fs.write_file("/etc/ssl/openssl.cnf", (
        "# Minimal OpenSSL configuration\n"
        "[default]\n"
        "openssl_conf = default_conf\n"
        "\n"
        "[default_conf]\n"
        "ssl_conf = ssl_sect\n"
        "\n"
        "[ssl_sect]\n"
        "system_default = system_default_sect\n"
        "\n"
        "[system_default_sect]\n"
        "MinProtocol = TLSv1.2\n"
        "CipherString = DEFAULT@SECLEVEL=2\n"
    ))

    fs.write_file("/etc/ssl/certs/ca-certificates.crt",
        "# nCPU GPU -- certificate placeholder\n"
        "# Real certificates not needed for GPU compute\n"
    )

    # ===================================================================
    # /etc/skel -- template user home
    # ===================================================================

    fs.write_file("/etc/skel/.profile", (
        "# ~/.profile: executed by the login shell\n"
        "export PATH=$HOME/bin:$PATH\n"
        "export EDITOR=vi\n"
    ))

    fs.write_file("/etc/skel/.bashrc", (
        "# ~/.bashrc: executed by bash for non-login shells\n"
        "alias ll='ls -la'\n"
        "alias la='ls -A'\n"
        "alias l='ls -CF'\n"
        "alias ..='cd ..'\n"
        "alias ...='cd ../..'\n"
    ))

    fs.write_file("/etc/skel/.ash_history", "")

    # ===================================================================
    # USER HOME DIRECTORIES
    # ===================================================================

    fs.write_file("/root/.profile", (
        "# ~/.profile for root\n"
        "export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\n"
        "export HOME=/root\n"
        "export PS1='\\u@\\h:\\w# '\n"
        "export EDITOR=vi\n"
        "export PAGER=less\n"
        "export LANG=C.UTF-8\n"
        "\n"
        "# Source ashrc if it exists\n"
        "[ -f \"$HOME/.ashrc\" ] && . \"$HOME/.ashrc\"\n"
    ))

    fs.write_file("/root/.ash_history", "")

    fs.write_file("/root/.ashrc", (
        "# ~/.ashrc -- ash shell configuration for root\n"
        "\n"
        "# Useful aliases\n"
        "alias ll='ls -la'\n"
        "alias la='ls -A'\n"
        "alias l='ls -CF'\n"
        "alias ..='cd ..'\n"
        "alias ...='cd ../..'\n"
        "alias h='history'\n"
        "alias cls='clear'\n"
        "alias df='df -h'\n"
        "alias du='du -sh'\n"
        "alias free='free -m'\n"
        "alias ps='ps aux'\n"
        "alias grep='grep --color=auto'\n"
        "\n"
        "# nCPU GPU aliases\n"
        "alias sysinfo='cat /proc/cpuinfo; cat /proc/meminfo'\n"
        "alias mounts='cat /proc/mounts'\n"
        "alias netstat='cat /proc/net/tcp; cat /proc/net/udp'\n"
    ))

    # ===================================================================
    # BOOT / INIT SYSTEM STUBS
    # ===================================================================

    fs.write_file("/etc/init.d/boot", (
        "#!/bin/ash\n"
        "# /etc/init.d/boot -- system boot script (stub)\n"
        "\n"
        "case \"$1\" in\n"
        "    start)\n"
        "        echo \"Booting nCPU GPU system...\"\n"
        "        hostname -F /etc/hostname\n"
        "        echo \"System boot complete.\"\n"
        "        ;;\n"
        "    stop)\n"
        "        echo \"System shutting down...\"\n"
        "        ;;\n"
        "    restart)\n"
        "        $0 stop\n"
        "        $0 start\n"
        "        ;;\n"
        "    *)\n"
        "        echo \"Usage: $0 {start|stop|restart}\"\n"
        "        exit 1\n"
        "        ;;\n"
        "esac\n"
    ))

    fs.write_file("/etc/init.d/networking", (
        "#!/bin/ash\n"
        "# /etc/init.d/networking -- network configuration (stub)\n"
        "\n"
        "case \"$1\" in\n"
        "    start)\n"
        "        echo \"Starting networking...\"\n"
        "        ifconfig lo 127.0.0.1 netmask 255.0.0.0 up 2>/dev/null\n"
        "        echo \"Networking started.\"\n"
        "        ;;\n"
        "    stop)\n"
        "        echo \"Stopping networking...\"\n"
        "        ;;\n"
        "    restart)\n"
        "        $0 stop\n"
        "        $0 start\n"
        "        ;;\n"
        "    *)\n"
        "        echo \"Usage: $0 {start|stop|restart}\"\n"
        "        exit 1\n"
        "        ;;\n"
        "esac\n"
    ))

    fs.write_file("/etc/init.d/hostname", (
        "#!/bin/ash\n"
        "# /etc/init.d/hostname -- set system hostname\n"
        "hostname -F /etc/hostname\n"
    ))

    fs.write_file("/etc/init.d/syslog", (
        "#!/bin/ash\n"
        "# /etc/init.d/syslog -- system logging (stub)\n"
        "\n"
        "case \"$1\" in\n"
        "    start)\n"
        "        echo \"Starting syslog...\"\n"
        "        ;;\n"
        "    stop)\n"
        "        echo \"Stopping syslog...\"\n"
        "        ;;\n"
        "    *)\n"
        "        echo \"Usage: $0 {start|stop}\"\n"
        "        exit 1\n"
        "        ;;\n"
        "esac\n"
    ))

    fs.write_file("/etc/conf.d/hostname", 'hostname="ncpu-gpu"\n')

    # ===================================================================
    # PROFILE.D SCRIPTS
    # ===================================================================

    fs.write_file("/etc/profile.d/ncpu.sh", (
        "#!/bin/ash\n"
        "# /etc/profile.d/ncpu.sh -- nCPU GPU environment setup\n"
        "\n"
        "export NCPU_VERSION=\"2.0\"\n"
        "export NCPU_ARCH=\"aarch64\"\n"
        "export NCPU_BACKEND=\"metal\"\n"
        "export NCPU_GPU=\"Apple Silicon\"\n"
        "\n"
        "# GPU compute information\n"
        "export GPU_MEMORY=\"16GB\"\n"
        "export GPU_IPS=\"4400000\"  # ~4.4M instructions/sec\n"
        "\n"
        "alias ncpu-info='/usr/bin/ncpu-info'\n"
    ))

    fs.write_file("/etc/profile.d/color_prompt.sh", (
        "#!/bin/ash\n"
        "# /etc/profile.d/color_prompt.sh -- colored prompt\n"
        "export PS1='\\033[1;32m\\u@\\h\\033[0m:\\033[1;34m\\w\\033[0m# '\n"
    ))

    fs.write_file("/etc/profile.d/locale.sh", (
        "#!/bin/ash\n"
        "# /etc/profile.d/locale.sh -- locale settings\n"
        "export CHARSET=UTF-8\n"
        "export LANG=C.UTF-8\n"
        "export LC_COLLATE=C\n"
    ))

    # ===================================================================
    # SHELL SCRIPTS
    # ===================================================================

    fs.write_file("/usr/bin/ncpu-info", (
        "#!/bin/ash\n"
        "# ncpu-info -- display nCPU GPU system information\n"
        "\n"
        "echo \"============================================\"\n"
        "echo \"  nCPU GPU System Information\"\n"
        "echo \"============================================\"\n"
        "echo \"\"\n"
        "echo \"Hostname:    $(hostname)\"\n"
        "echo \"OS:          $(cat /etc/os-release | grep PRETTY_NAME | cut -d'\"' -f2)\"\n"
        "echo \"Kernel:      $(cat /proc/version)\"\n"
        "echo \"Architecture: $(uname -m 2>/dev/null || echo aarch64)\"\n"
        "echo \"Uptime:      $(cat /proc/uptime | cut -d' ' -f1)s\"\n"
        "echo \"Load:        $(cat /proc/loadavg)\"\n"
        "echo \"\"\n"
        "echo \"--- CPU ---\"\n"
        "cat /proc/cpuinfo | grep -E 'model name|Hardware|BogoMIPS'\n"
        "echo \"\"\n"
        "echo \"--- Memory ---\"\n"
        "cat /proc/meminfo | head -7\n"
        "echo \"\"\n"
        "echo \"--- Mounts ---\"\n"
        "cat /proc/mounts\n"
        "echo \"\"\n"
        "echo \"--- GPU Compute ---\"\n"
        "echo \"Backend:     ${NCPU_BACKEND:-metal}\"\n"
        "echo \"GPU:         ${NCPU_GPU:-Apple Silicon}\"\n"
        "echo \"GPU Memory:  ${GPU_MEMORY:-16GB}\"\n"
        "echo \"Throughput:  ${GPU_IPS:-4400000} IPS\"\n"
        "echo \"============================================\"\n"
    ))

    fs.write_file("/usr/bin/lsdev", (
        "#!/bin/ash\n"
        "# lsdev -- list device nodes\n"
        "echo \"Device nodes:\"\n"
        "ls /dev/ 2>/dev/null\n"
    ))

    # ===================================================================
    # ADDITIONAL DATA FILES
    # ===================================================================

    fs.write_file("/usr/share/misc/magic", (
        "# /usr/share/misc/magic -- file(1) magic database (stub)\n"
        "0\tstring\t\\x7fELF\tELF\n"
        "0\tstring\t#!\t\tscript\n"
        "0\tstring\tPK\t\tZip archive\n"
        "0\tbeshort\t0x1f8b\t\tgzip compressed data\n"
        "0\tbeshort\t0x1f9d\t\tcompress'd data\n"
        "0\tbelong\t0x89504e47\tPNG image\n"
        "0\tbeshort\t0xffd8\t\tJPEG image\n"
        "0\tstring\tGIF8\t\tGIF image\n"
        "0\tstring\t%PDF\t\tPDF document\n"
        "0\tbelong\t0xcafebabe\tJava class\n"
        "0\tbelong\t0xfeedface\tMach-O\n"
        "0\tbelong\t0xfeedfacf\tMach-O 64-bit\n"
    ))

    # terminfo -- xterm and linux console
    fs.write_file("/usr/share/terminfo/x/xterm", (
        "# xterm terminal description (stub)\n"
        "xterm|xterm terminal emulator,\n"
        "\tam, km, mir, msgr, xenl,\n"
        "\tcols#80, it#8, lines#24,\n"
    ))

    fs.write_file("/usr/share/terminfo/v/vt100", (
        "# vt100 terminal description (stub)\n"
        "vt100|dec vt100,\n"
        "\tam, mc5i, msgr, xenl,\n"
        "\tcols#80, it#8, lines#24, vt#3,\n"
    ))

    fs.write_file("/usr/share/terminfo/l/linux", (
        "# linux console terminal description (stub)\n"
        "linux|linux console,\n"
        "\tam, eo, mir, msgr, xenl,\n"
        "\tcols#80, it#8, lines#25,\n"
    ))

    # ===================================================================
    # LOG FILES
    # ===================================================================

    fs.write_file("/var/log/messages", "")

    fs.write_file("/var/log/dmesg", (
        "[    0.000000] Booting Linux on nCPU GPU (Metal compute shader)\n"
        "[    0.000000] Linux version 6.1.0-ncpu (gcc) #1 SMP aarch64\n"
        "[    0.000000] Machine model: Apple Silicon (nCPU GPU)\n"
        "[    0.000000] Memory: 16384MB available\n"
        "[    0.000001] CPU: Apple Silicon ARM64 aarch64\n"
        "[    0.000001] Calibrating delay loop... 48.00 BogoMIPS\n"
        "[    0.000002] pid_max: default: 32768 minimum: 128\n"
        "[    0.000003] Mount-cache hash table entries: 1024\n"
        "[    0.000004] Inode-cache hash table entries: 8192\n"
        "[    0.000005] Dentry cache hash table entries: 4096\n"
        "[    0.000010] devtmpfs: initialized\n"
        "[    0.000020] clocksource: arch_sys_counter: mask: 0xffffffffffffff\n"
        "[    0.000030] Console: metal_console0\n"
        "[    0.000040] Freeing unused kernel memory: 0K\n"
        "[    0.000050] Run /sbin/init as init process\n"
        "[    0.000060] nCPU GPU subsystem initialized (~4.4M IPS)\n"
        "[    0.000070] Alpine Linux v3.20 filesystem mounted\n"
        "[    0.000080] BusyBox v1.36.1 loaded (264KB, 30+ applets)\n"
        "[    0.000090] System ready.\n"
    ))

    fs.write_file("/var/log/wtmp", "")
    fs.write_file("/var/log/lastlog", "")
    fs.write_file("/var/log/btmp", "")

    # ===================================================================
    # /var/run -- runtime state
    # ===================================================================

    fs.write_file("/var/run/utmp", "")

    # ===================================================================
    # MISC SYSTEM FILES
    # ===================================================================

    fs.write_file("/etc/TZ", "UTC\n")

    fs.write_file("/etc/modules", (
        "# /etc/modules -- kernel modules to load at boot\n"
    ))

    fs.write_file("/etc/modprobe.d/aliases.conf", (
        "# /etc/modprobe.d/aliases.conf\n"
    ))

    fs.write_file("/etc/logrotate.conf", (
        "# /etc/logrotate.conf\n"
        "weekly\n"
        "rotate 4\n"
        "create\n"
        "compress\n"
        "\n"
        "include /etc/logrotate.d\n"
    ))

    return fs


# ===================================================================
# FILESYSTEM STATISTICS
# Total directories: 61
# Total files: 109
# Total entries: 170
# ===================================================================
