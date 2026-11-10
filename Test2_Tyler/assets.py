"""Explicit download of a checksum-pinned external YOLO model; never overwrites a model."""
import hashlib
import os
from pathlib import Path
import urllib.request
from .adapters.yolo import MODEL_PATH, MODEL_URL, verify_model


def main():
    if MODEL_PATH.exists():
        print("Already present, verified:", verify_model(MODEL_PATH))
        return
    partial = MODEL_PATH.with_suffix(".download")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create preserves any prior partial download for diagnosis.
    with urllib.request.urlopen(MODEL_URL, timeout=30) as response, partial.open("xb") as out:
        while block := response.read(1024 * 1024):
            out.write(block)
        out.flush()
        os.fsync(out.fileno())
    verify_model(partial)
    partial.rename(MODEL_PATH)
    print("Downloaded and SHA256-verified:", MODEL_PATH)


if __name__ == "__main__":
    main()
