"""Addresses for the posting-page reader's address-guard tests
(tests/test_posting_text.py). Special-purpose and documentation addresses
only; none of them belongs to anyone."""

PUBLIC = "93.184.216.34"                      # example.com, a globally routable address

BY_HOST = {
    "localhost": "127.0.0.1",                 # loopback
    "intranet.example": "10.0.0.5",           # RFC 1918 private network
    "metadata.example": "169.254.169.254",    # link-local cloud metadata endpoint
    "mapped.example": "::ffff:127.0.0.1",     # IPv4-mapped loopback
    "overlay.example": "100.101.102.103",     # RFC 6598 shared address space
}
