"""
Helper utility to download official Meta SAM 2 release model checkpoints.
"""

import argparse
from pathlib import Path

import requests

from imagetochaste.config import SAM2_CHECKPOINTS


def download_checkpoint(model_type: str = "large", output_dir: str = "checkpoints") -> Path:
    """
    Download a SAM 2 model checkpoint from official Meta releases.

    Args:
        model_type: Key from SAM2_CHECKPOINTS ('tiny', 'small', 'base_plus', 'large', 'sam2.1_large')
        output_dir: Directory where the checkpoint file will be saved.

    Returns:
        Path to the downloaded checkpoint file.
    """
    if model_type not in SAM2_CHECKPOINTS:
        valid_keys = ", ".join(SAM2_CHECKPOINTS.keys())
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from: {valid_keys}")

    meta = SAM2_CHECKPOINTS[model_type]
    url = meta["url"]
    filename = meta["filename"]

    dest_dir = Path(output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / filename

    if dest_file.exists():
        print(f"Checkpoint already exists at {dest_file} ({dest_file.stat().st_size / (1024*1024):.1f} MB). Skipping download.")
        return dest_file

    print(f"Downloading SAM 2 '{model_type}' checkpoint from:\n  {url}\nTo:\n  {dest_file} ...")

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total_bytes = int(response.headers.get("content-length", 0))
    chunk_size = 1024 * 1024  # 1 MB
    downloaded_bytes = 0
    last_reported_pct = 0

    with open(dest_file, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded_bytes += len(chunk)
                if total_bytes > 0:
                    percent = int((downloaded_bytes / total_bytes) * 100)
                    if percent >= last_reported_pct + 25:
                        last_reported_pct = percent
                        mb_downloaded = downloaded_bytes / (1024 * 1024)
                        mb_total = total_bytes / (1024 * 1024)
                        print(f"Downloading checkpoint: {mb_downloaded:.1f} MB / {mb_total:.1f} MB ({percent}%)")

    print(f"Download complete: {dest_file} ({dest_file.stat().st_size / (1024*1024):.1f} MB)")
    return dest_file


def main():
    parser = argparse.ArgumentParser(description="Download Meta SAM 2 model weights.")
    parser.add_argument(
        "--model",
        type=str,
        default="large",
        choices=list(SAM2_CHECKPOINTS.keys()),
        help="Model architecture variant to download (default: large)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints",
        help="Target folder for downloaded weights (default: checkpoints)",
    )
    args = parser.parse_args()
    download_checkpoint(model_type=args.model, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
