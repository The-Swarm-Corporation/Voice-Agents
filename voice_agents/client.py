import httpx

_http_client = httpx.Client(
    limits=httpx.Limits(
        max_keepalive_connections=20,  # Increased for better connection reuse
        max_connections=60,  # Increased for higher concurrency
        keepalive_expiry=60.0,  # Longer expiry to reduce connection setup overhead
    ),
    timeout=httpx.Timeout(
        30.0,  # Read timeout
        connect=5.0,  # Faster connect timeout for quicker failure detection
        write=10.0,  # Write timeout
        pool=5.0,  # Pool timeout for getting connection from pool
    ),
    http2=True,  # Enable HTTP/2 for multiplexing (requires httpcore[h2])
    headers={
        "Accept-Encoding": "gzip, deflate, br",  # Request compressed responses
    },
)
