"""Install a pinned, isolated Mac runtime for the published HookNet-TLS model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".runtime" / "hooknet_tls"
SOURCES = {
    "hooknet_tls": ("DIAGNijmegen/pathology-hooknet-tls", "573680a5b962b1acae88cdeb6f8b1c9597570539"),
    "hooknet": ("DIAGNijmegen/pathology-hooknet", "3f2a28a4e91c7d4e70502ee11a50fb10c798ef65"),
    "wholeslidedata": ("DIAGNijmegen/pathology-whole-slide-data", "78dd18bae9cc0ff59e572d30c18555794e2ef1b6"),
}
PACKAGES = ["tensorflow==2.18.1", "tf-keras==2.18.0", "tensorflow-metal==1.2.0", "numpy==1.26.4",
            "scipy==1.13.1", "h5py==3.11.0", "shapely==2.0.6", "opencv-python-headless==4.10.0.84",
            "scikit-image==0.24.0", "tifffile==2024.9.20", "openslide-python==1.4.6",
            "openslide-bin==4.0.1.2", "PyYAML==6.0.2", "pandas==2.2.3", "matplotlib==3.9.4", "contourpy==1.3.1"]


def fetch(url, target):
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "HE2CELL"}), timeout=180) as response:
            data = response.read()
        temporary = target.with_suffix(target.suffix + ".part")
        temporary.write_bytes(data)
        temporary.replace(target)
    return hashlib.sha256(target.read_bytes()).hexdigest()


def setup(skip_install=False):
    RUNTIME.mkdir(parents=True, exist_ok=True)
    manifest = {"sources": {}, "packages_requested": PACKAGES}
    # Reuse the initially inspected archives when their git-archive commit matches.
    inspected = {"hooknet_tls": "source.tar.gz", "hooknet": "hooknet.tar.gz", "wholeslidedata": "wholeslidedata.tar.gz"}
    for name, (repository, revision) in SOURCES.items():
        url = f"https://codeload.github.com/{repository}/tar.gz/{revision}"
        archive = RUNTIME / inspected[name]
        sha = fetch(url, archive)
        directory = RUNTIME / "sources" / name
        with tarfile.open(archive) as package:
            if package.pax_headers.get("comment") != revision:
                raise ValueError(f"Unexpected source revision for {name}")
            if not directory.exists():
                unpack = RUNTIME / "source_unpack" / name
                unpack.mkdir(parents=True, exist_ok=True)
                package.extractall(unpack, filter="data")
                extracted = list(unpack.iterdir())
                if len(extracted) != 1:
                    raise ValueError("Unexpected source archive layout")
                directory.parent.mkdir(parents=True, exist_ok=True)
                extracted[0].rename(directory)
        manifest["sources"][name] = {"repository": repository, "revision": revision, "url": url,
            "archive_sha256": sha, "directory": str(directory)}
    weights = RUNTIME / "weights.h5"
    weights_url = "https://zenodo.org/records/10614942/files/weights.h5?download=1"
    sha = fetch(weights_url, weights)
    md5 = hashlib.md5(weights.read_bytes()).hexdigest()
    if weights.stat().st_size != 51063832 or md5 != "72f85daa6112cb5d30794cc64f04cb9d":
        raise ValueError("Pretrained weights do not match the official Zenodo record")
    manifest["weights"] = {"path": str(weights), "url": weights_url, "zenodo_record": 10614942,
                           "bytes": weights.stat().st_size, "published_md5": md5, "sha256": sha}
    python = RUNTIME / "venv" / "bin" / "python"
    if not python.exists():
        subprocess.run([sys.executable, "-m", "venv", str(RUNTIME / "venv")], check=True)
    if not skip_install:
        subprocess.run([str(python), "-m", "pip", "install", *PACKAGES], check=True)
    locked = subprocess.run([str(python), "-m", "pip", "freeze"], check=True, text=True, capture_output=True).stdout
    (RUNTIME / "environment.lock.txt").write_text(locked)
    manifest["python"] = str(python)
    manifest["environment_lock"] = locked
    (RUNTIME / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-install", action="store_true", help="Record already-installed packages and validate sources/weights")
    setup(parser.parse_args().skip_install)
