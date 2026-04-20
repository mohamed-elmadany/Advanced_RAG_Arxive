import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import config

KAGGLE_DATASET = "Cornell-University/arxiv"
SNAPSHOT_NAME = "arxiv-metadata-oai-snapshot.json"


def _print_credentials_help():
    print(
        "\nKaggle credentials not found. Set them up via one of:\n"
        "  1. Environment variables:\n"
        "       set KAGGLE_USERNAME=<your_username>\n"
        "       set KAGGLE_KEY=<your_api_key>\n"
        "  2. Or place kaggle.json at:\n"
        "       Windows: %USERPROFILE%\\.kaggle\\kaggle.json\n"
        "       Linux/macOS: ~/.kaggle/kaggle.json\n"
        "Get credentials from https://www.kaggle.com/settings -> Create New API Token.\n"
    )


def _has_credentials() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    home = Path(os.environ.get("USERPROFILE") or os.path.expanduser("~"))
    return (home / ".kaggle" / "kaggle.json").exists()


def _locate_snapshot(cache_dir: Path) -> Path:
    direct = cache_dir / SNAPSHOT_NAME
    if direct.exists():
        return direct
    for path in cache_dir.rglob(SNAPSHOT_NAME):
        return path
    raise FileNotFoundError(
        f"{SNAPSHOT_NAME} was not found inside {cache_dir} after download."
    )


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"


def main():
    try:
        import kagglehub
    except ImportError:
        print("kagglehub is not installed. Run: pip install kagglehub")
        sys.exit(1)

    if not _has_credentials():
        _print_credentials_help()
        sys.exit(1)

    target_path = config.RAW_DATA_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists() and target_path.stat().st_size > 0:
        print(f"Dataset already present at {target_path} ({_human_size(target_path.stat().st_size)}). Skipping.")
        return

    print(f"Downloading {KAGGLE_DATASET} from Kaggle (this may take a while)...")
    try:
        cache_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    except Exception as exc:
        print(f"Kaggle download failed: {exc}")
        _print_credentials_help()
        sys.exit(1)

    try:
        snapshot_src = _locate_snapshot(cache_dir)
    except FileNotFoundError as exc:
        print(str(exc))
        sys.exit(1)

    print(f"Copying snapshot to {target_path} ...")
    shutil.copy2(snapshot_src, target_path)

    size = target_path.stat().st_size
    print(f"Done. Saved {target_path} ({_human_size(size)}).")


if __name__ == "__main__":
    main()
