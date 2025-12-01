import re
import pandas as pd
import chardet
from pathlib import Path

input_file = "data.txt"
output_folder = "data_staging"

# читаем файл и определяем кодировку
with open(input_file, 'rb') as f:
    data = f.read(200000)
    enc = chardet.detect(data)['encoding']
    if not enc:
        enc = 'utf-8'

with open(input_file, 'r', encoding=enc, errors='ignore') as f:
    text = f.read()

pattern = r"COPY\s+(\w+\.\w+)\s*\(([^)]+)\)\s+FROM\s+stdin;\s*(.*?)\n\\\."
matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

print(f"нашли таблиц: {len(matches)}")

for table, columns, data_part in matches:
    # колонки
    cols = [c.strip() for c in columns.split(',')]
    
    # строки данных
    lines = data_part.strip().split('\n')
    lines = [line.strip() for line in lines if line.strip()]
    
    if not lines:
        print(f"пропуск {table} - нет данных")
        continue
    
    # парсим данные
    rows = []
    for line in lines:
        parts = line.split('\t')
        row = [None if p == r'\N' else p for p in parts]
        rows.append(row)
    
    # выравниваем количество колонок
    max_cols = max(len(r) for r in rows)
    if len(cols) > max_cols:
        cols = cols[:max_cols]
    elif len(cols) < max_cols:
        for i in range(len(cols), max_cols):
            cols.append(f'col{i+1}')
    
    # создаем датафрейм
    df = pd.DataFrame(rows, columns=cols)
    
    # сохраняем
    out_name = table.replace('.', '_') + '.csv'
    out_path = Path(output_folder) / out_name
    out_path.parent.mkdir(exist_ok=True)
    
    df.to_csv(out_path, index=False)
    
    print(f"{table}: {len(df)} строк -> {out_path}")

print("готово")

