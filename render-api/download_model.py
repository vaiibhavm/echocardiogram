from __future__ import annotations

import hashlib
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


MODEL_URL = (
    "https://drive.usercontent.google.com/download"
    "?id=1HOaNCSx0q_vDNlXz2iHSXiqAQQrutvjY&export=download&confirm=t"
)
EXPECTED_SHA256 = "dd4494a66f8cb8e01d930a57d8c66627eede46b1d947d1ebdb5811be79e8f5b1"
MODEL_DIR = Path(__file__).with_name("model")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    destination_root = destination.resolve()
    for item in archive.infolist():
        target = (destination / item.filename).resolve()
        if target != destination_root and destination_root not in target.parents:
            raise RuntimeError(f"Unsafe path in model archive: {item.filename}")
    archive.extractall(destination)


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temporary:
        archive_path = Path(temporary.name)

    try:
        print("Downloading the public Echo LV ONNX model bundle...")
        with urllib.request.urlopen(MODEL_URL, timeout=180) as response:
            with archive_path.open("wb") as destination:
                shutil.copyfileobj(response, destination)

        actual_sha256 = sha256(archive_path)
        if actual_sha256 != EXPECTED_SHA256:
            raise RuntimeError(
                f"Model checksum mismatch: expected {EXPECTED_SHA256}, got {actual_sha256}"
            )

        with zipfile.ZipFile(archive_path) as archive:
            safe_extract(archive, MODEL_DIR)

        if not (MODEL_DIR / "model.onnx").is_file():
            raise RuntimeError("Model archive did not contain model/model.onnx")
        print("Model download and checksum verification complete.")
    finally:
        archive_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
