"""Read selected public BOP ZIP members over HTTP byte ranges."""

from __future__ import annotations

import io
import urllib.request
import zipfile


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url: str):
        self.url = url
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as response:
            self.size = int(response.headers["Content-Length"])
            self.etag = response.headers.get("x-linked-etag") or response.headers.get("ETag")
        self.pos = 0
        self.requests = 0
        self.transferred = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = (0 if whence == 0 else self.pos if whence == 1 else self.size) + offset
        return self.pos

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = self.size - self.pos
        size = min(size, self.size - self.pos)
        if size <= 0:
            return b""
        start = self.pos
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{start + size - 1}"})
        with urllib.request.urlopen(request, timeout=180) as response:
            content_range = response.headers.get("Content-Range")
            if response.status != 206 or not content_range or not content_range.startswith(f"bytes {start}-"):
                raise IOError(f"HTTP range request was not honored: {response.status} {content_range}")
            data = response.read()
        if len(data) != size:
            raise IOError(f"Short HTTP range read: {len(data)} != {size}")
        self.pos += size
        self.requests += 1
        self.transferred += size
        return data


def open_zip(url: str) -> tuple[zipfile.ZipFile, HTTPRangeFile]:
    source = HTTPRangeFile(url)
    return zipfile.ZipFile(source), source
