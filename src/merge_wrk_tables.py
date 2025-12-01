import sys
import re
import pandas as pd
import os

sys.path.append('..')
import settings as cfg

print(f"работаем в папке: {cfg.ROOT}")

# 1. загрузка файлов
files = {
    "flights": "wrk_flights.csv",
    "sirena": "airlines_sirena_export.csv",
    "sirena_users": "airlines_sirena_export_users.csv",
    "users": "wrk_users.csv"
}

flights = pd.read_csv(os.path.join(cfg.STAGING, files["flights"]), low_memory=False)
sirena = pd.read_csv(os.path.join(cfg.STAGING, files["sirena"]), low_memory=False)
sirena_users = pd.read_csv(os.path.join(cfg.STAGING, files["sirena_users"]), low_memory=False)
users = pd.read_csv(os.path.join(cfg.STAGING, files["users"]), low_memory=False)

print(f"флайтс: {len(flights)}")
print(f"сирена: {len(sirena)}")
print(f"сирена юзерс: {len(sirena_users)}")
print(f"юзерс: {len(users)}")

# 2. чистим колонки
def clean_df(df):
    df.columns = [c.strip().replace('"', '').replace("'", '') for c in df.columns]
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace('nan', '')
    return df

flights = clean_df(flights)
sirena = clean_df(sirena)
sirena_users = clean_df(sirena_users)
users = clean_df(users)

print("после чистки")
print(flights.columns[:5].tolist())

# 3. нормализация документов
def fix_doc(x):
    if not isinstance(x, str):
        return ''
    return re.sub(r'\D', '', x)

for df in [sirena, sirena_users, users]:
    if 'document' in df.columns:
        df['document_norm'] = df['document'].apply(fix_doc)
    elif 'travel_doc' in df.columns:
        df['document_norm'] = df['travel_doc'].apply(fix_doc)

# 4. мердж флайтс и сирена
print("\nмерджим флайтс и сирену")
ticket_col = None
for col in sirena.columns:
    if 'ticket' in col.lower():
        ticket_col = col
        break

if ticket_col:
    print(f"билеты в колонке {ticket_col}")
else:
    print("нет колонки с билетами")
    ticket_col = ''

cols_to_use = ["id","departure_date","departure_time","arrival_date","arrival_time",
               "fare","baggage","meal","trv_cls","travel_doc","agent_info",
               "pax_name","document_norm","pax_birth_data"]

if ticket_col and ticket_col not in cols_to_use:
    cols_to_use.append(ticket_col)

cols_to_use = [c for c in cols_to_use if c in sirena.columns]

merged = pd.merge(flights, sirena[cols_to_use], left_on="sirena_id", right_on="id", how="left")

if ticket_col and ticket_col in merged.columns:
    merged['eticket'] = merged[ticket_col]
else:
    merged['eticket'] = ''

print(f"после мерджа: {len(merged)} строк")

# 5. добавляем имена из sirena_users
print("\nдобавляем имена")

# ищем колонки с датой рождения
birth_col1 = None
birth_col2 = None
for col in merged.columns:
    if 'birth' in col.lower():
        birth_col1 = col
        break
for col in sirena_users.columns:
    if 'birth' in col.lower():
        birth_col2 = col
        break

# нормализуем имена
def fix_name(n):
    if not isinstance(n, str):
        return ''
    n = re.sub(r'[^A-Za-zА-Яа-я ]', '', n)
    return n.strip().lower()

if 'pax_name' in merged.columns:
    merged['pax_last'] = merged['pax_name'].apply(lambda x: x.split()[0] if isinstance(x, str) and x.split() else '')
    merged['pax_first'] = merged['pax_name'].apply(lambda x: x.split()[1] if isinstance(x, str) and len(x.split()) > 1 else '')
    merged['pax_last_norm'] = merged['pax_last'].apply(fix_name)
    merged['pax_first_norm'] = merged['pax_first'].apply(fix_name)

sirena_users['last_name_norm'] = sirena_users['last_name'].apply(fix_name)
sirena_users['first_name_norm'] = sirena_users['first_name'].apply(fix_name)

# мердж
if birth_col1 and birth_col2:
    print(f"мердж по имени и дате рождения {birth_col1}")
    merged = pd.merge(merged, sirena_users[['first_name','last_name','second_name','last_name_norm','first_name_norm',birth_col2]], 
                     left_on=['pax_last_norm','pax_first_norm',birth_col1],
                     right_on=['last_name_norm','first_name_norm',birth_col2],
                     how='left')
    merged['match_reason'] = 'name+birth'
else:
    print("мердж только по имени")
    merged = pd.merge(merged, sirena_users[['first_name','last_name','second_name','last_name_norm','first_name_norm']],
                     left_on=['pax_last_norm','pax_first_norm'],
                     right_on=['last_name_norm','first_name_norm'],
                     how='left')
    merged['match_reason'] = 'name_only'

print(f"нашли имена: {(merged['first_name'] != '').mean():.1%}")

# 6. добавляем users
print("\nдобавляем users")
birth_col1 = None
birth_col2 = None
for col in merged.columns:
    if 'birth' in col.lower():
        birth_col1 = col
        break
for col in users.columns:
    if 'birth' in col.lower():
        birth_col2 = col
        break

left_keys = ['document_norm']
right_keys = ['document_norm']
if birth_col1 and birth_col2:
    left_keys.append(birth_col1)
    right_keys.append(birth_col2)

merged = pd.merge(merged, users[['first_name_v2','last_name_v2','sex','document_norm'] + ([birth_col2] if birth_col2 else [])],
                 left_on=left_keys, right_on=right_keys, how='left', suffixes=('', '_wrk'))

print(f"пол заполнен: {(merged['sex'] != '').mean():.1%}")

# 7. финальные имена
print("\nфинальные имена")

def fix_case(s):
    if not isinstance(s, str) or not s.strip():
        return ''
    s = s.strip().capitalize()
    if re.search(r'[А-Яа-я]', s):
        try:
            from unidecode import unidecode
            s = unidecode(s)
        except:
            pass
    return s

def get_first(*args):
    for v in args:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ''

merged['first_name_final'] = merged.apply(lambda r: get_first(
    r.get('first_name_v2', ''),
    r.get('first_name', ''),
    r.get('pax_first', '')
), axis=1)

merged['last_name_final'] = merged.apply(lambda r: get_first(
    r.get('last_name_v2', ''),
    r.get('last_name', ''),
    r.get('pax_last', '')
), axis=1)

merged['first_name_final'] = merged['first_name_final'].apply(fix_case)
merged['last_name_final'] = merged['last_name_final'].apply(fix_case)

if 'first_name' in merged.columns:
    merged.drop('first_name', axis=1, inplace=True)
if 'last_name' in merged.columns:
    merged.drop('last_name', axis=1, inplace=True)

merged.rename(columns={'first_name_final': 'first_name', 'last_name_final': 'last_name'}, inplace=True)

print(f"заполнено имён: {(merged['first_name'] != '').mean():.1%}")

# 8. статистика
print("\nстатистика:")
fields = ['first_name','last_name','sex','pax_birth_data','document_norm','fare','baggage','agent_info']
for f in fields:
    if f in merged.columns:
        filled = (merged[f] != '').mean()
        print(f"{f}: {filled:.1%}")

# 9. сохранение
output_cols = ["flight_code","flight_date","departure","arrival",
               "departure_date","departure_time","arrival_date","arrival_time",
               "fare","baggage","meal","trv_cls","agent_info",
               "first_name","last_name","second_name","sex","pax_birth_data","document_norm","match_reason"]

output_cols = [c for c in output_cols if c in merged.columns]

out_path = os.path.join(cfg.STAGING, "merged_all_detailed.csv")
merged[output_cols].fillna('').to_csv(out_path, index=False, encoding='utf-8')

print(f"\nсохранено в {out_path}")
print(f"всего строк: {len(merged)}")
print(merged[output_cols].head(3))
