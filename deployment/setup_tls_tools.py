"""Fetch pinned public sources and install tlsR in an isolated project library."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".runtime" / "tls_tools"
R_SCRIPT = Path("/opt/miniconda3/envs/singlecell/bin/Rscript")
REVISIONS = {
    "TLS-Finder": ("AAKoksoy/TLS-Finder", "cde11ecbd541ec280609f494a9dd9409639359b4"),
    "tlsR": ("labsyspharm/tlsR", "5e6acf7bebc9464635782b37fcf9dc70fc6daafe"),
}


def download(url: str, target: Path) -> dict:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "HE2CELL-research-comparison"})
        with urllib.request.urlopen(request, timeout=120) as response:
            target.write_bytes(response.read())
    return {"url": url, "path": str(target), "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}


def setup(rscript: Path = R_SCRIPT) -> dict:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    lib = RUNTIME / "R-library"
    lib.mkdir(exist_ok=True)
    manifest = {"sources": {}, "downloaded_R_dependencies": {}, "Rscript": str(rscript), "R_library": str(lib)}
    for name, (repo, revision) in REVISIONS.items():
        archive = RUNTIME / "downloads" / f"{name}-{revision}.tar.gz"
        info = download(f"https://codeload.github.com/{repo}/tar.gz/{revision}", archive)
        destination = RUNTIME / "sources" / name
        if not destination.exists():
            unpack = RUNTIME / "unpack" / name
            unpack.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive) as package:
                package.extractall(unpack, filter="data")
            roots = list(unpack.iterdir())
            if len(roots) != 1:
                raise RuntimeError(f"Unexpected archive root for {name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            roots[0].rename(destination)
        manifest["sources"][name] = {**info, "repository": repo, "revision": revision, "directory": str(destination)}
        print(f"Pinned {repo} at {revision[:7]}", flush=True)

    env = os.environ.copy()
    env["R_LIBS_USER"] = str(lib)
    env["PATH"] = str(rscript.parent) + os.pathsep + env.get("PATH", "")
    r_exe = rscript.parent / "R"
    index_path = RUNTIME / "downloads" / "CRAN-PACKAGES.gz"
    download("https://cloud.r-project.org/src/contrib/PACKAGES.gz", index_path)
    package_index = gzip.decompress(index_path.read_bytes()).decode()
    for package in ("dbscan", "fastICA"):
        record = next(item for item in package_index.split("\n\n") if item.startswith(f"Package: {package}\n"))
        fields = dict(line.split(": ", 1) for line in record.splitlines() if ": " in line and not line.startswith(" "))
        version = fields["Version"]
        archive = RUNTIME / "downloads" / f"{package}_{version}.tar.gz"
        info = download(f"https://cloud.r-project.org/src/contrib/{archive.name}", archive)
        manifest["downloaded_R_dependencies"][package] = {**info, "version": version}
        if not (lib / package / "DESCRIPTION").exists():
            subprocess.run([str(r_exe), "CMD", "INSTALL", f"--library={lib}", str(archive)], env=env, check=True)
    if not (lib / "tlsR" / "DESCRIPTION").exists():
        source = Path(manifest["sources"]["tlsR"]["directory"]) / "v0.3.0"
        subprocess.run([str(r_exe), "CMD", "INSTALL", f"--library={lib}", str(source)], env=env, check=True)
    query = 'p <- c("tlsR","dbscan","fastICA","FNN","spatstat.explore","spatstat.geom","ggplot2","rlang"); cat(R.version.string,"\\n"); for (x in p) cat(x,as.character(packageVersion(x)),"\\n"); print(formals(tlsR::detect_TLS))'
    result = subprocess.run([str(rscript), "-e", query], env=env, check=True, text=True, capture_output=True)
    manifest["R_environment"] = result.stdout
    (RUNTIME / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(result.stdout, flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rscript", type=Path, default=R_SCRIPT)
    setup(parser.parse_args().rscript)
