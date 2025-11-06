# -*- coding: utf-8 -*-
import sys
import re
import pandas as pd
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))
import settings as cfg

print(f"[settings] Project root: {cfg.ROOT}")

# ==========================================================
# 1. ЗАГРУЗКА
# ==========================================================
tables = {
    "flights": "wrk_flights.csv",
    "sirena": "airlines_sirena_export.csv",
    "sirena_users": "airlines_sirena_export_users.csv",
    "users": "wrk_users.csv"
}
dfs = {}
for name, file in tables.items():
    path = cfg.STAGING / file
    dfs[name] = pd.read_csv(path, low_memory=False)
    print(f"✅ {name}: {len(dfs[name])} строк, {len(dfs[name].columns)} колонок")

flights, sirena, sirena_users, users = dfs["flights"], dfs["sirena"], dfs["sirena_users"], dfs["users"]

# ==========================================================
# 2. ОЧИСТКА И НОРМАЛИЗАЦИЯ
# ==========================================================
for df_name, df in dfs.items():
    df.columns = df.columns.str.strip().str.replace('"', '').str.replace("'", '')
    for col in df.columns:
        df[col] = df[col].astype(str).str.strip().replace("nan", "")
    print(f"🧩 {df_name} columns: {df.columns[:8].tolist()}...")

def normalize_doc(s):
    """Убираем нецифры и пробелы в документе."""
    if not isinstance(s, str): return ""
    return re.sub(r"\D", "", s or "")

for d in [sirena, sirena_users, users]:
    if "document" in d.columns:
        d["document_norm"] = d["document"].map(normalize_doc)
    elif "travel_doc" in d.columns:
        d["document_norm"] = d["travel_doc"].map(normalize_doc)

# ==========================================================
# 3. MERGE flights ↔ sirena
# ==========================================================
print("\n🔗 Шаг 1: flights ↔ sirena по sirena_id → id")

eticket_col = None
for c in sirena.columns:
    if re.search(r"ticket", c, re.IGNORECASE):
        eticket_col = c
        break

if eticket_col:
    print(f"🔍 Найдено поле для билета: {eticket_col}")
else:
    print("⚠️ В sirena не найдено поле с билетом (eticket / ticket_number).")
    eticket_col = None

# включаем pax_birth_data
cols_for_merge = [
    "id","departure_date","departure_time","arrival_date","arrival_time",
    "fare","baggage","meal","trv_cls","travel_doc","agent_info",
    "pax_name","document_norm","pax_birth_data"
]
if eticket_col and eticket_col not in cols_for_merge:
    cols_for_merge.append(eticket_col)

cols_existing = [c for c in cols_for_merge if c in sirena.columns]

merged = flights.merge(
    sirena[cols_existing],
    left_on="sirena_id",
    right_on="id",
    how="left"
)

if eticket_col and eticket_col in merged.columns:
    merged["eticket"] = merged[eticket_col]
else:
    print(f"⚠️ Колонка {eticket_col or 'eticket'} не найдена после merge, создаём пустую.")
    merged["eticket"] = ""

print(f"✅ После merge 1: {len(merged)} строк, заполнено eticket: {(merged['eticket'] != '').mean():.1%}")

# ==========================================================
# 4. MERGE sirena ↔ sirena_users (имена + дата)
# ==========================================================
print("\n🔗 Шаг 2: sirena ↔ sirena_users по pax_birth_data и именам")

# определяем реальные поля
birth_left = next((c for c in merged.columns if "birth" in c.lower()), None)
birth_right = next((c for c in sirena_users.columns if "birth" in c.lower()), None)

if not birth_left:
    print("⚠️ В merged нет поля с датой рождения (birth_date / pax_birth_data).")
else:
    print(f"📆 Поле даты рождения найдено: {birth_left}")

# нормализуем имена
def normalize_name(s):
    if not isinstance(s, str): return ""
    s = re.sub(r"[^A-Za-zА-Яа-яЁё ]", "", s)
    return s.strip().lower()

if "pax_name" in merged.columns:
    merged["pax_last"]  = merged["pax_name"].map(lambda x: x.split()[0] if isinstance(x, str) and len(x.split()) > 0 else "")
    merged["pax_first"] = merged["pax_name"].map(lambda x: x.split()[1] if isinstance(x, str) and len(x.split()) > 1 else "")
    merged["pax_last_norm"]  = merged["pax_last"].map(normalize_name)
    merged["pax_first_norm"] = merged["pax_first"].map(normalize_name)

sirena_users["last_name_norm"]  = sirena_users["last_name"].map(normalize_name)
sirena_users["first_name_norm"] = sirena_users["first_name"].map(normalize_name)

# объединяем
if birth_left and birth_right:
    print(f"🔁 Объединяем по: фамилии + имени + {birth_left}")
    merged = merged.merge(
        sirena_users[
            ["first_name","last_name","second_name","last_name_norm","first_name_norm",birth_right]
        ],
        left_on=["pax_last_norm","pax_first_norm",birth_left],
        right_on=["last_name_norm","first_name_norm",birth_right],
        how="left",
        suffixes=("", "_su")
    )
    merged["match_reason"] = "name+birth"
else:
    print("⚠️ birth_date не найдено, объединяем только по имени/фамилии")
    merged = merged.merge(
        sirena_users[
            ["first_name","last_name","second_name","last_name_norm","first_name_norm"]
        ],
        left_on=["pax_last_norm","pax_first_norm"],
        right_on=["last_name_norm","first_name_norm"],
        how="left",
        suffixes=("", "_su")
    )
    merged["match_reason"] = "name_only"

found_names = (merged["first_name"] != "").mean()
print(f"✅ После merge 2: {len(merged)} строк, имена найдены: {found_names:.1%}")
print("📋 Примеры найденных пользователей:")
cols_preview = ["first_name", "last_name"]
if birth_left and birth_left in merged.columns:
    cols_preview.append(birth_left)
print(merged.loc[merged['first_name'] != '', cols_preview].head(10).to_string(index=False))

# ==========================================================
# 5. MERGE с wrk_users (по document_norm + birth)
# ==========================================================
print("\n🔗 Шаг 3: добавляем wrk_users (sex, ГОСТ-имена)")
birth_left = next((c for c in merged.columns if "birth" in c.lower()), None)
birth_right = next((c for c in users.columns if "birth" in c.lower()), None)

if not birth_left:
    print("⚠️ В merged нет поля с датой рождения, объединяем только по документу.")
if not birth_right:
    print("⚠️ В users нет поля с датой рождения, объединяем только по документу.")

keys_left = ["document_norm"]
keys_right = ["document_norm"]
if birth_left and birth_right:
    keys_left.append(birth_left)
    keys_right.append(birth_right)
    print(f"🔁 Объединяем по: {keys_left}")
else:
    print("🔁 Объединяем только по документу")

merged = merged.merge(
    users[["first_name_v2","last_name_v2","sex","document_norm"] + ([birth_right] if birth_right else [])],
    left_on=keys_left,
    right_on=keys_right,
    how="left",
    suffixes=("", "_wrk")
)

print(f"✅ После merge 3: {len(merged)} строк, пол заполнен: {(merged['sex'] != '').mean():.1%}")

# ==========================================================
# 7. ВОССТАНОВЛЕНИЕ ИМЁН С ПРИОРИТЕТОМ И ТРАНСЛИТЕРАЦИЕЙ
# ==========================================================
print("\n🧩 Расширенное восстановление имён (приоритет wrk_users > sirena_users > pax_name)")

try:
    from unidecode import unidecode
except ImportError:
    unidecode = None

def normalize_case(s):
    """Приведение регистра и транслитерация."""
    if not isinstance(s, str) or not s.strip():
        return ""
    s = s.strip().capitalize()
    if re.search(r"[А-Яа-яЁё]", s):
        s = unidecode(s) if unidecode else s
    return s

def coalesce(*values):
    """Берёт первое непустое значение."""
    for v in values:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""

# если остались старые first_name_* — удаляем дубликаты
for col in ["first_name", "last_name"]:
    if isinstance(merged.get(col), pd.DataFrame):
        merged[col] = merged[col].iloc[:, 0]

# создаём финальные имена с приоритетом wrk_users > sirena_users > pax_name
merged["first_name_final"] = merged.apply(
    lambda r: coalesce(
        r.get("first_name_v2", ""),
        r.get("first_name_su", ""),
        r.get("first_name", ""),
        r.get("pax_first", "")
    ),
    axis=1
)

merged["last_name_final"] = merged.apply(
    lambda r: coalesce(
        r.get("last_name_v2", ""),
        r.get("last_name_su", ""),
        r.get("last_name", ""),
        r.get("pax_last", "")
    ),
    axis=1
)

merged["first_name_final"] = merged["first_name_final"].map(normalize_case)
merged["last_name_final"]  = merged["last_name_final"].map(normalize_case)

# перезаписываем поля окончательно
merged.drop(columns=[c for c in merged.columns if c in ["first_name","last_name"]], inplace=True, errors="ignore")
merged.rename(columns={"first_name_final": "first_name", "last_name_final": "last_name"}, inplace=True)

# --- вычисляем долю непустых имён ---
mask = merged["first_name"].astype(str).str.strip() != ""
filled_names_ratio = mask.mean()
print(f"✅ После восстановления имён: {filled_names_ratio:.1%}")


# ==========================================================
# 8. СТАТИСТИКА
# ==========================================================
def stat(field): return f"{(merged[field] != '').mean():.1%}" if field in merged else "—"
print("\n📊 Краткая статистика заполненности:")
for f in ["first_name","last_name","sex","pax_birth_data","document_norm","fare","baggage","agent_info"]:
    print(f"   {f:15}: {stat(f)}")

# ==========================================================
# 9. СОХРАНЕНИЕ
# ==========================================================
cols = [
    "flight_code","flight_date","departure","arrival",
    "departure_date","departure_time","arrival_date","arrival_time",
    "fare","baggage","meal","trv_cls","agent_info",
    "first_name","last_name","second_name","sex","pax_birth_data","document_norm","match_reason"
]
cols = [c for c in cols if c in merged.columns]

out = cfg.STAGING / "merged_all_detailed.csv"
merged[cols].fillna("").to_csv(out, index=False, encoding="utf-8")

print(f"\n💾 Сохранено → {out}")
print(f"📈 Всего строк: {len(merged)}")
print("📊 Пример строк:")
print(merged[cols].head(8).to_string(index=False))
