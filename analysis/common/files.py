"""File descriptions for local input and output records."""

from pathlib import Path


def file_info(path: str | Path) -> dict[str, str | int]:
    path = Path(path)
    return {"name": path.name, "size_bytes": path.stat().st_size}
