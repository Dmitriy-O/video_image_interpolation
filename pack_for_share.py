"""
Створює легкий ZIP для передачі проєкту (без venv, кешу, відео).

Запуск з кореня проєкту:
    python pack_for_share.py
"""

from __future__ import annotations

import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "dist"

# Що включати (відносно кореня)
INCLUDE_FILES = [
    "app.py",
    "requirements.txt",
    "README.md",
    "pack_for_share.py",
    ".gitignore",
    "src/feature_extraction.py",
    "src/clustering.py",
    "src/interpolation.py",
    "src/visualization.py",
    "tests/test_core_logic.py",
]

SKIP_DIR_NAMES = {
    "venv", ".venv", "env", "__pycache__", ".git",
    ".pytest_cache", ".gradio", "dist", ".vscode", ".idea",
}


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file() and not any(part in SKIP_DIR_NAMES for part in p.parts):
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def create_archive() -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d")
    out_zip = OUTPUT_DIR / f"video_interpolation_course_{stamp}.zip"

    missing = [f for f in INCLUDE_FILES if not (ROOT / f).exists()]
    if missing:
        raise FileNotFoundError(f"Відсутні файли: {missing}")

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for rel in INCLUDE_FILES:
            zf.write(ROOT / rel, rel.replace("\\", "/"))

    return out_zip


def main() -> None:
    venv_size = _dir_size(ROOT / "venv")
    archive = create_archive()
    zip_size = archive.stat().st_size

    print("Готово.")
    print(f"  Архів:     {archive}")
    print(f"  Розмір:    {_human_size(zip_size)}")
    if venv_size:
        print(f"  venv (не в архіві): ~{_human_size(venv_size)}")
    print()
    print("Отримувачу:")
    print("  1. Розпакувати ZIP")
    print("  2. python -m venv venv")
    print("  3. venv\\Scripts\\activate   (Windows)")
    print("  4. pip install -r requirements.txt")
    print("  5. python app.py")


if __name__ == "__main__":
    main()