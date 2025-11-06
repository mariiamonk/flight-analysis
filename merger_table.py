# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# ==================================================
# 0. ЗАГРУЗКА
# ==================================================
df = pd.read_csv(r"/home/mariia/Загрузки/Telegram Desktop/AI2/data_staging/merged_all_detailed.csv", low_memory=False)
print(f"✅ Загружено {len(df):,} строк, {len(df.columns)} колонок")

# Приведение типов
df["flight_date"] = pd.to_datetime(df["flight_date"], errors="coerce")

# ==================================================
# 1. ГРУППИРОВКА ПО ЧЕЛОВЕКУ (а не по документу)
# ==================================================
grouped = df.groupby(["first_name", "last_name", "pax_birth_data"])

# --- Базовые метрики ---
agg = grouped.agg(
    n_flights_total=("flight_code", "count"),
    n_unique_documents=("document_norm", "nunique"),
    n_unique_routes=("flight_code", lambda x: len(set(zip(df.loc[x.index, "departure"], df.loc[x.index, "arrival"])))),
    n_unique_agents=("agent_info", "nunique"),
    baggage_ratio=("baggage", lambda x: (x != "").mean())
).reset_index()

# --- Временные интервалы между рейсами ---
df_sorted = df.sort_values(["first_name", "last_name", "pax_birth_data", "flight_date"])
df_sorted["gap_days"] = df_sorted.groupby(["first_name", "last_name", "pax_birth_data"])["flight_date"].diff().dt.days

gap_stats = (
    df_sorted.groupby(["first_name", "last_name", "pax_birth_data"])["gap_days"]
    .agg(mean_time_between_flights="mean", min_time_between_flights="min")
    .reset_index()
)

# Присоединяем интервалы обратно к agg
agg = agg.merge(gap_stats, on=["first_name", "last_name", "pax_birth_data"], how="left")

# --- Активность во времени ---
agg["days_active"] = grouped["flight_date"].apply(
    lambda x: (x.max() - x.min()).days if len(x.dropna()) > 1 else 0
).values
agg["flights_per_month"] = agg["n_flights_total"] / ((agg["days_active"] / 30).replace(0, 1))

# --- Доля пропусков по ключевым полям ---
def calc_missing_ratio(subdf):
    key_fields = ["fare", "baggage", "agent_info"]
    existing = [f for f in key_fields if f in subdf.columns]
    if not existing:
        return np.nan
    return (subdf[existing] == "").mean().mean()

agg["missing_ratio"] = grouped.apply(calc_missing_ratio).values

# --- Частота маршрутов и агентов ---
route_counts = df.groupby(["departure", "arrival"]).size().rename("route_freq")
df = df.merge(route_counts, on=["departure", "arrival"], how="left")

agent_counts = df["agent_info"].value_counts().to_dict()
agg["avg_route_popularity"] = grouped["flight_code"].apply(
    lambda x: np.mean(df.loc[x.index, "route_freq"])
).values
agg["avg_agent_popularity"] = grouped["agent_info"].apply(
    lambda x: np.mean([agent_counts.get(a, 0) for a in x])
).values

# ==================================================
# 2. МОДЕЛЬ Isolation Forest
# ==================================================
features = agg.drop(columns=["first_name", "last_name", "pax_birth_data"]).fillna(0)

# Нормализация
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Isolation Forest
model = IsolationForest(
    n_estimators=300,
    contamination=0.02,  # ожидаем 2% аномалий
    random_state=42
)
model.fit(features_scaled)

agg["anomaly_score"] = model.decision_function(features_scaled)
agg["is_suspicious"] = model.predict(features_scaled)  # -1 = аномалия

# ==================================================
# 3. ПОЯСНЕНИЕ “ПОЧЕМУ ПОДОЗРИТЕЛЕН”
# ==================================================
def explain(row):
    reasons = []
    if row["n_unique_documents"] > 1:
        reasons.append(f"использовал {int(row['n_unique_documents'])} разных документов")
    if row["flights_per_month"] > 10:
        reasons.append("чрезмерная частота перелётов")
    if row["baggage_ratio"] < 0.2 and row["n_flights_total"] > 5:
        reasons.append("частые рейсы без багажа")
    if row["missing_ratio"] > 0.3:
        reasons.append("много пропусков в данных")
    if row["n_unique_agents"] > 5:
        reasons.append("использует много агентств продаж")
    if not reasons:
        reasons.append("аномалия по совокупности признаков")
    return "; ".join(reasons)

agg["reason"] = agg.apply(explain, axis=1)

# ==================================================
# 4. ВЫВОД И СОХРАНЕНИЕ
# ==================================================
suspects = agg[agg["is_suspicious"] == -1].sort_values("anomaly_score")

print(f"\n🚨 Найдено подозрительных личностей: {len(suspects)} / {len(agg)}")
print(suspects[[
    "first_name", "last_name", "pax_birth_data",
    "n_unique_documents", "n_flights_total", "flights_per_month",
    "n_unique_agents", "baggage_ratio",
    "missing_ratio", "reason", "anomaly_score"
]].head(15).to_string(index=False))

# Сохранение отчёта
out_path = r"/home/mariia/Загрузки/Telegram Desktop/AI2/suspicious_passengers_detailed.csv"
agg.to_csv(out_path, index=False, encoding="utf-8")
print(f"\n💾 Отчёт сохранён → {out_path}")
