# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# загружаем данные
df = pd.read_csv(r"/home/mariia/Загрузки/Telegram Desktop/AI2/data_staging/merged_all_detailed.csv", low_memory=False)
print(f"загрузили {len(df)} строк")

# дату в нормальный формат
df["flight_date"] = pd.to_datetime(df["flight_date"])

# группируем по людям
grouped = df.groupby(["first_name", "last_name", "pax_birth_data"])

# считаем базовые метрики
agg = grouped.agg(
    n_flights_total=("flight_code", "count"),
    n_unique_documents=("document_norm", "nunique"),
    n_unique_agents=("agent_info", "nunique")
).reset_index()

# уникальные маршруты
def count_routes(x):
    routes = set()
    for idx in x.index:
        dep = df.loc[idx, "departure"]
        arr = df.loc[idx, "arrival"]
        routes.add((dep, arr))
    return len(routes)

agg["n_unique_routes"] = grouped["flight_code"].apply(count_routes).values

# багаж
agg["baggage_ratio"] = grouped["baggage"].apply(lambda x: (x != "").mean()).values

# время между рейсами
df_sorted = df.sort_values(["first_name", "last_name", "pax_birth_data", "flight_date"])
df_sorted["gap_days"] = df_sorted.groupby(["first_name", "last_name", "pax_birth_data"])["flight_date"].diff().dt.days

gap_stats = df_sorted.groupby(["first_name", "last_name", "pax_birth_data"])["gap_days"].agg(
    mean_time_between_flights="mean",
    min_time_between_flights="min"
).reset_index()

agg = agg.merge(gap_stats, on=["first_name", "last_name", "pax_birth_data"], how="left")

# активность
def get_days_active(x):
    if len(x) > 1:
        return (x.max() - x.min()).days
    return 0

agg["days_active"] = grouped["flight_date"].apply(get_days_active).values
agg["flights_per_month"] = agg["n_flights_total"] / ((agg["days_active"] / 30).replace(0, 1))

# пропуски в данных
def calc_missing(subdf):
    missing = 0
    if "fare" in subdf.columns:
        missing += (subdf["fare"] == "").mean()
    if "baggage" in subdf.columns:
        missing += (subdf["baggage"] == "").mean()
    if "agent_info" in subdf.columns:
        missing += (subdf["agent_info"] == "").mean()
    return missing / 3

agg["missing_ratio"] = grouped.apply(calc_missing).values

# популярность маршрутов
route_counts = {}
for _, row in df.iterrows():
    route = (row["departure"], row["arrival"])
    route_counts[route] = route_counts.get(route, 0) + 1

route_popularity = []
for name, group in grouped:
    pop_sum = 0
    count = 0
    for idx in group.index:
        route = (df.loc[idx, "departure"], df.loc[idx, "arrival"])
        pop_sum += route_counts.get(route, 0)
        count += 1
    route_popularity.append(pop_sum / count if count > 0 else 0)

agg["avg_route_popularity"] = route_popularity

# популярность агентов
agent_counts = df["agent_info"].value_counts().to_dict()

agent_popularity = []
for name, group in grouped:
    pop_sum = 0
    count = 0
    for agent in group["agent_info"]:
        pop_sum += agent_counts.get(agent, 0)
        count += 1
    agent_popularity.append(pop_sum / count if count > 0 else 0)

agg["avg_agent_popularity"] = agent_popularity

# модель
features = agg.drop(columns=["first_name", "last_name", "pax_birth_data"]).fillna(0)

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

model = IsolationForest(
    n_estimators=300,
    contamination=0.02,
    random_state=42
)
model.fit(features_scaled)

agg["anomaly_score"] = model.decision_function(features_scaled)
agg["is_suspicious"] = model.predict(features_scaled)

# объяснения
reasons_list = []
for i, row in agg.iterrows():
    reasons = []
    if row["n_unique_documents"] > 1:
        reasons.append(f"{int(row['n_unique_documents'])} документов")
    if row["flights_per_month"] > 10:
        reasons.append("часто летает")
    if row["baggage_ratio"] < 0.2 and row["n_flights_total"] > 5:
        reasons.append("нет багажа")
    if row["missing_ratio"] > 0.3:
        reasons.append("пропуски в данных")
    if row["n_unique_agents"] > 5:
        reasons.append("много агентов")
    if len(reasons) == 0:
        reasons.append("странное поведение")
    reasons_list.append("; ".join(reasons))

agg["reason"] = reasons_list

# результаты
suspects = agg[agg["is_suspicious"] == -1].sort_values("anomaly_score")

print(f"нашли {len(suspects)} подозрительных")
print(suspects[[
    "first_name", "last_name", "pax_birth_data",
    "n_unique_documents", "n_flights_total", "flights_per_month",
    "n_unique_agents", "baggage_ratio",
    "missing_ratio", "reason", "anomaly_score"
]].head(15).to_string(index=False))

# сохраняем
agg.to_csv(r"/home/mariia/Загрузки/Telegram Desktop/AI2/suspicious_passengers_detailed.csv", index=False, encoding="utf-8")
print("сохранено в suspicious_passengers_detailed.csv")
