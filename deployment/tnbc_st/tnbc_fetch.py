"""Stream Zenodo 14204217 tarballs, md5-check the whole stream, extract only the members needed.

Nothing is written to disk except the selected members (tarfile 'data' filter, so no path traversal)
plus, for large JPEG members that are not kept, a record of their size, md5 and pixel dimensions (to
identify which image frame byArray spot tables refer to). A .done.json per tar records the md5 check.

Usage: .venv/bin/python deployment/tnbc_st/tnbc_fetch.py Clinical.tar misc.tar rawCountsMatrices.tar byArray.tar
"""
from __future__ import annotations

import fnmatch
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
EXTRAS = ROOT / "data" / "TNBC_ST" / "extras"
URL = "https://zenodo.org/records/14204217/files/{}?download=1"
# tar -> member basename patterns to keep (None = keep all)
KEEP = {"Clinical.tar": None, "misc.tar": None, "rawCountsMatrices.tar": None,
        "byArray.tar": ["spot_data*", "allSpots*", "annotBySpot*", "artefacts*", "small.jpg", "*.txt", "*.tsv"]}
# Large images inside byArray: record size/md5/dims only.
PROBE = ["*.jpg", "*.jpeg", "*.png", "*.tif"]


class Hashing(io.RawIOBase):
    def __init__(self, raw):
        self.raw, self.md5, self.bytes = raw, hashlib.md5(), 0

    def readable(self):
        return True

    def readinto(self, buffer):
        data = self.raw.read(len(buffer))
        n = len(data)
        buffer[:n] = data
        self.md5.update(data)
        self.bytes += n
        return n


def jpeg_size(data):
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(io.BytesIO(data)) as image:
        return list(image.size)


def fetch(name):
    record = {f["key"]: f for f in json.loads((EXTRAS / "record.json").read_text())["files"]}[name]
    expected = record["checksum"].split(":")[1]
    target = EXTRAS / name.replace(".tar", "")
    done = target / ".done.json"
    if done.exists() and json.loads(done.read_text()).get("md5_ok"):
        print(f"{name}: already done", flush=True)
        return
    target.mkdir(parents=True, exist_ok=True)
    started = time.time()
    response = urllib.request.urlopen(URL.format(urllib.request.quote(name)), timeout=120)
    stream = Hashing(response)
    buffered = io.BufferedReader(stream, 1 << 22)
    kept, probed, skipped = [], [], 0
    patterns = KEEP[name]
    with tarfile.open(fileobj=buffered, mode="r|") as tar:
        for member in tar:
            base = Path(member.name).name
            if not member.isfile() or base.startswith("._"):
                continue
            if patterns is None or any(fnmatch.fnmatch(base, p) for p in patterns):
                tar.extract(member, target, filter="data")
                kept.append([member.name, member.size])
            elif any(fnmatch.fnmatch(base.lower(), p) for p in PROBE):
                data = tar.extractfile(member).read()
                try:
                    dims = jpeg_size(data)
                except Exception as error:  # noqa: BLE001
                    dims = repr(error)
                probed.append({"name": member.name, "size": member.size, "md5": hashlib.md5(data).hexdigest(), "dims": dims})
                del data
            else:
                skipped += 1
    # Drain anything after the end-of-archive marker so the md5 covers the whole file.
    while buffered.read(1 << 22):
        pass
    md5 = stream.md5.hexdigest()
    result = {"file": name, "bytes": stream.bytes, "expected_bytes": record["size"], "md5": md5, "expected_md5": expected,
              "md5_ok": md5 == expected and stream.bytes == record["size"], "kept": kept, "probed_images": probed,
              "skipped_members": skipped, "seconds": time.time() - started}
    (target / "members.json").write_text(json.dumps(result, indent=1))
    done.write_text(json.dumps({k: v for k, v in result.items() if k not in ("kept", "probed_images")}, indent=1))
    print(f"{name}: md5_ok={result['md5_ok']} kept {len(kept)}, probed {len(probed)}, {result['seconds']:.0f}s", flush=True)
    if not result["md5_ok"]:
        sys.exit(f"{name}: md5 mismatch {md5} != {expected}")


if __name__ == "__main__":
    for tar_name in sys.argv[1:]:
        fetch(tar_name)
