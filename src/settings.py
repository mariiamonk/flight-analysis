# -*- coding: utf-8 -*-
"""
settings.py — общие пути проекта Airlines_project
"""

from pathlib import Path

# === Корень проекта (папка, где лежит settings.py) ===
ROOT = Path(__file__).parent.resolve()

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data_raw"
STAGING = ROOT / "data_staging"
FEATURES = ROOT / "data_features"
MODELS = ROOT / "models"

for d in [RAW, STAGING, FEATURES, MODELS]:
    d.mkdir(exist_ok=True)
# === Создаём папки, если их нет ===


# === Опционально: базовые пути для часто используемых файлов ===
RAW_SQL = RAW / "data.txt"       # или data.sql — см. input_path в скриптах

print(f"[settings] Project root: {ROOT}")
