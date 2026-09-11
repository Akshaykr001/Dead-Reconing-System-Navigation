"""Download and extract the synchronized IO-VNBD archive from GitHub."""
from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

ARCHIVE_URL = "https://media.githubusercontent.com/media/onyekpeu/IO-VNBD/master/Synchronised%20V%20abd%20S%20datasets.zip"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--archive", type=Path, help="Use a previously downloaded archive")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = args.archive or args.output_dir / "iovnbd_synchronised.zip"
    if not archive.exists():
        print(f"Downloading IO-VNBD archive to {archive} ...")
        urllib.request.urlretrieve(ARCHIVE_URL, archive)
    print(f"Extracting {archive} ...")
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(args.output_dir)
    print(f"Dataset extracted under {args.output_dir}")


if __name__ == "__main__":
    main()
