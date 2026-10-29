"""Repeatable entry point to add independent HookNet-TLS to the saved comparison."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".runtime" / "hooknet_tls"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "outputs" / "whole-slide-20261012")
    parser.add_argument("--setup", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    args = parser.parse_args()
    if args.setup:
        subprocess.run([sys.executable, str(ROOT/"deployment/setup_hooknet_tls.py")], check=True)
    if not (RUNTIME/"manifest.json").exists():
        parser.error("Run with --setup to download pinned sources and install the isolated environment")
    python = json.loads((RUNTIME/"manifest.json").read_text())["python"]
    if not args.report_only:
        subprocess.run([python, "-u", str(ROOT/"deployment/run_hooknet_tls.py"), "--run-dir", str(args.run_dir.resolve()), "--device", args.device], check=True)
    subprocess.run([python, str(ROOT/"deployment/report_hooknet_tls.py"), "--run-dir", str(args.run_dir.resolve())], check=True)


if __name__ == "__main__":
    main()
