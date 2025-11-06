import re
import pandas as pd
import chardet
from pathlib import Path
import sys, os
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
import srс.settings as cfg


input_path = cfg.RAW / "data.txt"       # твой дамп
output_dir = cfg.STAGING
output_dir.mkdir(parents=True, exist_ok=True)

# === 1. Определяем кодировку ===
with open(input_path, "rb") as f:
    enc = chardet.detect(f.read(200000))["encoding"] or "utf-8"

# === 2. Читаем весь файл ===
with open(input_path, "r", encoding=enc, errors="ignore") as f:
    text = f.read()

# === 3. Находим все COPY-блоки ===
pattern = re.compile(
    r"COPY\s+([\w\.]+)\s*\(([^)]+)\)\s+FROM\s+stdin;\s*(.*?)\n\\\.",
    re.DOTALL | re.IGNORECASE,
)
matches = pattern.findall(text)

print(f"📦 Найдено таблиц: {len(matches)}")

for table_name, columns, rows in matches:
    cols = [c.strip() for c in columns.split(",")]
    lines = [line.strip() for line in rows.strip().split("\n") if line.strip()]

    parsed_rows = []
    for line in lines:
        parts = line.split("\t")
        parsed_rows.append([None if p == r"\N" else p for p in parts])

    # 💡 Пропускаем пустые таблицы
    if not parsed_rows:
        print(f"⚠️ {table_name}: пропущено (нет данных)")
        continue

    # 💡 Исправляем несоответствие количества колонок и данных
    max_len = max(len(p) for p in parsed_rows)
    if len(cols) < max_len:
        cols += [f"extra_{i}" for i in range(len(cols)+1, max_len+1)]
    elif len(cols) > max_len:
        cols = cols[:max_len]

    df = pd.DataFrame(parsed_rows, columns=cols)
    csv_path = output_dir / f"{table_name.replace('.', '_')}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")

    print(f"✅ {table_name}: {len(df)} строк → {csv_path}")


print("🎉 Готово! Все таблицы сохранены в data_staging/")
