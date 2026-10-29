"""Resumable, per-member extraction of TCGA-ESCA.tar that survives unreadable regions of the drive.

Each SVS is copied by byte offset into a .part file and renamed only when complete. A member
whose data cannot be read after retries is recorded in unreadable_members.json and skipped.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tarfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from esca_common import OUT, RAW, TAR, save_json  # noqa: E402

CHUNK = 16 << 20


def copy_member(handle, member, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    done = partial.stat().st_size if partial.exists() else 0
    with open(partial, "ab") as out:
        while done < member.size:
            for attempt in range(4):
                try:
                    handle.seek(member.offset_data + done)
                    data = handle.read(min(CHUNK, member.size - done))
                    break
                except OSError:
                    time.sleep(2 * (attempt + 1))
            else:
                raise OSError(f"unreadable at byte {done} of {member.name}")
            out.write(data)
            done += len(data)
    partial.replace(destination)


def main():
    failures_path = OUT / "unreadable_members.json"
    failures = json.loads(failures_path.read_text()) if failures_path.exists() else {}
    with open(TAR, "rb", buffering=0) as handle, tarfile.open(fileobj=handle, mode="r:") as archive:
        members = [m for m in archive.getmembers() if m.isfile() and m.name.endswith(".svs")]
    with open(TAR, "rb", buffering=0) as handle:
        for member in members:
            destination = RAW.parent / member.name
            if (destination.exists() and destination.stat().st_size == member.size) or member.name in failures:
                continue
            if destination.exists():
                destination.unlink()  # truncated by the failed tar run
            try:
                copy_member(handle, member, destination)
                print(f"extracted {Path(member.name).name}", flush=True)
            except OSError as error:
                failures[member.name] = str(error)
                save_json(failures, failures_path)
                print(f"UNREADABLE {member.name}: {error}", flush=True)
    print(f"extraction complete; {len(failures)} unreadable", flush=True)


if __name__ == "__main__":
    main()
