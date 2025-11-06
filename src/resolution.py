import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Настройка стиля
plt.style.use('default')
sns.set_palette("dark:blue")
DARK_BLUE = "#1f4e79"
LIGHT_BLUE = "#4a7bb5"

class FlightMapVisualizer:
    
    def __init__(self):
        self.geolocator = Nominatim(user_agent="flight_analysis")
        self.airport_coords = {}
        self.load_common_airports()
    
    def load_common_airports(self):
        common_airports = {
            'Moscow': (55.7558, 37.6173), 'London': (51.5074, -0.1278), 'Paris': (48.8566, 2.3522),
            'New York': (40.7128, -74.0060), 'Tokyo': (35.6762, 139.6503), 'Dubai': (25.2048, 55.2708),
            'Istanbul': (41.0082, 28.9784), 'Frankfurt': (50.1109, 8.6821), 'Amsterdam': (52.3676, 4.9041),
            'Madrid': (40.4168, -3.7038), 'Rome': (41.9028, 12.4964), 'Barcelona': (41.3851, 2.1734),
            'MCX': (42.8167, 47.6527), 'SVO': (55.9726, 37.4146), 'KRR': (45.0347, 39.1706),
            'VVO': (43.3983, 132.1480), 'KJA': (56.1729, 92.4933), 'KHV': (48.5280, 135.1885),
            'ROV': (47.2582, 39.8181), 'SVX': (56.7431, 60.8027), 'UFA': (54.5575, 55.8744),
            'IKT': (52.2680, 104.3890), 'KZN': (55.6062, 49.2787)
        }
        self.airport_coords.update(common_airports)
    
    def fast_geocode_airport(self, airport_name):
        if airport_name in self.airport_coords:
            return self.airport_coords[airport_name]
        
        for known_airport, coords in self.airport_coords.items():
            if known_airport.lower() in airport_name.lower() or airport_name.lower() in known_airport.lower():
                self.airport_coords[airport_name] = coords
                return coords
        
        try:
            location = self.geolocator.geocode(airport_name + " airport", timeout=10)
            if location:
                coords = (location.latitude, location.longitude)
                self.airport_coords[airport_name] = coords
                return coords
        except (GeocoderTimedOut, GeocoderServiceError):
            pass
        
        return None

def analyze_activity_patterns(passenger_data):
    if len(passenger_data) < 2:
        return {
            'activity_cluster_score': 0, 'sudden_activity_increase': 0,
            'logistic_inconsistency': 0, 'peak_activity_period': 0,
            'avg_flights_per_period': 0
        }
    
    passenger_data = passenger_data.sort_values('flight_date')
    dates = passenger_data['flight_date'].sort_values()
    date_diff = dates.diff().dt.days.fillna(0)
    
    activity_clusters = []
    current_cluster = []
    
    for i, diff in enumerate(date_diff):
        if diff <= 2:
            current_cluster.append(i)
        else:
            if len(current_cluster) > 1:
                activity_clusters.append(current_cluster)
            current_cluster = [i]
    
    if len(current_cluster) > 1:
        activity_clusters.append(current_cluster)
    
    cluster_score = sum(len(cluster) ** 1.5 for cluster in activity_clusters) / len(passenger_data) if len(passenger_data) > 0 else 0
    
    sudden_increase = 0
    if len(passenger_data) >= 4:
        passenger_data_copy = passenger_data.copy()
        passenger_data_copy['week'] = passenger_data_copy['flight_date'].dt.isocalendar().week
        passenger_data_copy['year'] = passenger_data_copy['flight_date'].dt.year
        weekly_activity = passenger_data_copy.groupby(['year', 'week']).size().reset_index(name='flights')
        
        if len(weekly_activity) >= 3:
            weekly_activity = weekly_activity.sort_values(['year', 'week'])
            weekly_flights = weekly_activity['flights'].values
            
            max_spike = 0
            for i in range(2, len(weekly_flights)):
                previous_median = np.median(weekly_flights[i-2:i])
                if previous_median > 0:
                    spike_ratio = weekly_flights[i] / previous_median
                    if spike_ratio > max_spike:
                        max_spike = spike_ratio
            
            sudden_increase = max_spike
    
    logistic_issues = 0
    if 'departure' in passenger_data.columns and 'arrival' in passenger_data.columns:
        daily_activity = passenger_data.groupby(passenger_data['flight_date'].dt.date).agg({
            'departure': list, 'arrival': list
        }).reset_index()
        
        for _, day in daily_activity.iterrows():
            if len(day['departure']) > 1:
                arrivals = set(day['arrival'])
                departures = set(day['departure'])
                if len(departures - arrivals) > 0:
                    logistic_issues += len(departures - arrivals)
    
    logistic_inconsistency = logistic_issues / len(passenger_data) if len(passenger_data) > 0 else 0
    
    if len(dates) > 0:
        total_days = (dates.max() - dates.min()).days + 1
        peak_period = len(passenger_data) / total_days if total_days > 0 else 0
    else:
        peak_period = 0
    
    return {
        'activity_cluster_score': cluster_score, 'sudden_activity_increase': sudden_increase,
        'logistic_inconsistency': logistic_inconsistency, 'peak_activity_period': peak_period,
        'avg_flights_per_period': len(passenger_data) / 30 if len(passenger_data) > 30 else len(passenger_data) / ((dates.max() - dates.min()).days + 1) if len(dates) > 0 else 0
    }

def main_analysis():
    print("Запуск анализа паттернов активности...")

    df = pd.read_csv(r"/home/mariia/Загрузки/Telegram Desktop/AI2/data_staging/merged_all_detailed.csv", low_memory=False)
    print(f"Загружено {len(df):,} строк, {len(df.columns)} колонок")

    print("Предобработка данных...")
    text_columns = ['document_norm', 'first_name', 'last_name', 'pax_birth_data', 'departure', 'arrival', 'agent_info']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).replace('nan', '').replace('None', '')

    mask = (
        (df['first_name'] != '') & (df['last_name'] != '') & 
        (df['pax_birth_data'] != '') & (df['document_norm'] != '')
    )
    valid_df = df[mask].copy()
    print(f"Валидных записей: {len(valid_df):,}")

    print("Анализ документов...")
    doc_stats = valid_df.groupby('document_norm').agg(
        unique_passengers=('first_name', 'nunique'), total_flights=('flight_code', 'count')
    ).reset_index()
    suspicious_docs = doc_stats[doc_stats['unique_passengers'] > 1]
    print(f"Найдено подозрительных документов: {len(suspicious_docs)}")

    print("Группировка данных с анализом паттернов активности...")
    valid_df['passenger_id'] = valid_df['first_name'] + '|' + valid_df['last_name'] + '|' + valid_df['pax_birth_data']
    valid_df['flight_date'] = pd.to_datetime(valid_df['flight_date'], errors='coerce')
    valid_df = valid_df.dropna(subset=['flight_date'])
    print(f"Записей с корректными датами: {len(valid_df)}")

    print("Анализ паттернов активности для каждого пассажира...")
    basic_stats = valid_df.groupby('passenger_id').agg({
        'flight_code': 'count', 'document_norm': 'nunique', 'agent_info': 'nunique',
        'flight_date': ['min', 'max'], 'departure': lambda x: list(x.unique()) if 'departure' in valid_df.columns else [],
        'arrival': lambda x: list(x.unique()) if 'arrival' in valid_df.columns else []
    }).reset_index()

    new_columns = ['passenger_id', 'n_flights_total', 'n_unique_documents', 'n_unique_agents', 'first_flight', 'last_flight']
    if 'departure' in valid_df.columns: new_columns.extend(['departure_airports'])
    if 'arrival' in valid_df.columns: new_columns.extend(['arrival_airports'])
    basic_stats.columns = new_columns

    activity_patterns = []
    for passenger_id in basic_stats['passenger_id']:
        passenger_data = valid_df[valid_df['passenger_id'] == passenger_id]
        patterns = analyze_activity_patterns(passenger_data)
        patterns['passenger_id'] = passenger_id
        activity_patterns.append(patterns)

    activity_df = pd.DataFrame(activity_patterns)
    passenger_stats = basic_stats.merge(activity_df, on='passenger_id', how='left')
    passenger_stats[['first_name', 'last_name', 'pax_birth_data']] = passenger_stats['passenger_id'].str.split('|', expand=True)

    print("Расчет метрик с анализом паттернов...")
    passenger_stats['days_active'] = ((pd.to_datetime(passenger_stats['last_flight']) - pd.to_datetime(passenger_stats['first_flight'])).dt.days.clip(lower=1))
    passenger_stats['avg_activity_frequency'] = passenger_stats['avg_flights_per_period']

    if 'departure_airports' in passenger_stats.columns:
        passenger_stats['n_unique_departures'] = passenger_stats['departure_airports'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    if 'arrival_airports' in passenger_stats.columns:
        passenger_stats['n_unique_arrivals'] = passenger_stats['arrival_airports'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    if 'n_unique_departures' in passenger_stats.columns and 'n_unique_arrivals' in passenger_stats.columns:
        passenger_stats['total_unique_airports'] = passenger_stats['n_unique_departures'] + passenger_stats['n_unique_arrivals']

    print("Быстрый расчет подозрительных документов...")
    suspicious_docs_set = set(suspicious_docs['document_norm'].values)
    passenger_docs = valid_df.groupby(['first_name', 'last_name', 'pax_birth_data'])['document_norm'].apply(list).reset_index()

    def get_suspicious_docs_fast(doc_list):
        suspicious = [doc for doc in doc_list if doc in suspicious_docs_set]
        return suspicious, len(suspicious)

    passenger_docs[['suspicious_documents', 'suspicious_docs_count']] = pd.DataFrame(
        passenger_docs['document_norm'].apply(get_suspicious_docs_fast).tolist(), index=passenger_docs.index
    )

    passenger_stats = passenger_stats.merge(
        passenger_docs[['first_name', 'last_name', 'pax_birth_data', 'suspicious_documents', 'suspicious_docs_count']],
        on=['first_name', 'last_name', 'pax_birth_data'], how='left'
    )
    passenger_stats['has_suspicious_doc'] = passenger_stats['suspicious_docs_count'] > 0
    print(f"Обработано {len(passenger_stats)} уникальных пассажиров")

    print("Расчет уровня риска с анализом паттернов...")
    def calculate_risk_score_with_reasons(row):
        score = 0
        reasons = []
        
        if row['has_suspicious_doc']:
            score += 150
            score += row['suspicious_docs_count'] * 20
            reasons.append(f"Общие документы ({row['suspicious_docs_count']} шт)")
        
        if row['activity_cluster_score'] > 2.0:
            score += 60
            reasons.append("Высокая кластерная активность")
        elif row['activity_cluster_score'] > 1.0:
            score += 30
            reasons.append("Кластерная активность")
        
        if row['sudden_activity_increase'] > 10.0:
            score += 80
            reasons.append("Очень резкий всплеск активности")
        elif row['sudden_activity_increase'] > 5.0:
            score += 60
            reasons.append("Резкий всплеск активности")
        elif row['sudden_activity_increase'] > 3.0:
            score += 40
            reasons.append("Значительный всплеск активности")
        elif row['sudden_activity_increase'] > 2.0:
            score += 20
            reasons.append("Всплеск активности")
        
        if row['logistic_inconsistency'] > 0.3:
            score += 80
            reasons.append("Высокая логистическая несогласованность")
        elif row['logistic_inconsistency'] > 0.1:
            score += 40
            reasons.append("Логистическая несогласованность")
        
        if row['peak_activity_period'] > 2.0:
            score += 50
            reasons.append("Очень высокая пиковая активность")
        elif row['peak_activity_period'] > 1.0:
            score += 25
            reasons.append("Высокая пиковая активность")
        
        if row['n_unique_agents'] >= 10:
            score += 100
            reasons.append("Очень много агентов (10+)")
        elif row['n_unique_agents'] >= 7:
            score += 70
            reasons.append("Много агентов (7-9)")
        elif row['n_unique_agents'] >= 5:
            score += 50
            reasons.append("Несколько агентов (5-6)")
        elif row['n_unique_agents'] >= 3:
            score += 30
            reasons.append("Несколько агентов (3-4)")
        
        if row['n_unique_documents'] > 3:
            score += 50
            reasons.append("Много документов (4+)")
        elif row['n_unique_documents'] > 1:
            score += 25
            reasons.append("Несколько документов (2-3)")
        
        if 'total_unique_airports' in row and row['total_unique_airports'] > 10:
            score += 40
            reasons.append("Очень много аэропортов (10+)")
        elif 'total_unique_airports' in row and row['total_unique_airports'] > 5:
            score += 20
            reasons.append("Много аэропортов (6-10)")
        
        return int(score), "; ".join(reasons)

    risk_results = passenger_stats.apply(calculate_risk_score_with_reasons, axis=1)
    passenger_stats['risk_score'] = [x[0] for x in risk_results]
    passenger_stats['risk_reasons'] = [x[1] for x in risk_results]

    def get_risk_category(score):
        if score >= 200: return "КРИТИЧЕСКИЙ"
        elif score >= 100: return "ВЫСОКИЙ"
        elif score >= 50: return "СРЕДНИЙ"
        elif score >= 20: return "НИЗКИЙ"
        else: return "НОРМА"

    passenger_stats['risk_category'] = passenger_stats['risk_score'].apply(get_risk_category)
    passenger_stats['is_suspicious'] = passenger_stats['risk_score'] >= 50

    print(f"Статистика рисков:")
    print(f"   - Подозрительных пассажиров: {passenger_stats['is_suspicious'].sum()}")
    print(f"   - С общими документами: {passenger_stats['has_suspicious_doc'].sum()}")
    print(f"   - С паттернами кластерной активности: {(passenger_stats['activity_cluster_score'] > 1).sum()}")
    print(f"   - С резкими всплесками активности: {(passenger_stats['sudden_activity_increase'] > 2).sum()}")

    print("Создание визуализаций...")
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.patch.set_alpha(0.0)

    risk_counts = passenger_stats['risk_category'].value_counts()
    axes[0,0].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', colors=[DARK_BLUE, LIGHT_BLUE, '#6ba3d6', '#8fbce8', '#b4d4f0'])
    axes[0,0].set_title('Распределение по категориям риска', fontweight='bold', color=DARK_BLUE)

    suspicious_only = passenger_stats[passenger_stats['is_suspicious'] == True]
    if len(suspicious_only) > 0:
        axes[0,1].scatter(suspicious_only['activity_cluster_score'], suspicious_only['sudden_activity_increase'], c=suspicious_only['risk_score'], cmap='Blues', alpha=0.7, s=50)
        axes[0,1].set_title('Кластерная активность vs Всплески активности', fontweight='bold', color=DARK_BLUE)
        axes[0,1].set_xlabel('Оценка кластерной активности')
        axes[0,1].set_ylabel('Коэффициент всплеска активности')

    if len(suspicious_only) > 0:
        sns.histplot(data=suspicious_only, x='logistic_inconsistency', bins=20, ax=axes[0,2], color=DARK_BLUE)
        axes[0,2].set_title('Логистическая несогласованность', fontweight='bold', color=DARK_BLUE)
        axes[0,2].set_xlabel('Уровень логистической несогласованности')

    sns.boxplot(data=passenger_stats, x='risk_category', y='n_unique_agents', ax=axes[1,0], palette=[DARK_BLUE, LIGHT_BLUE, '#6ba3d6', '#8fbce8'])
    axes[1,0].set_title('Связь агентов и категории риска', fontweight='bold', color=DARK_BLUE)

    risk_reasons_analysis = passenger_stats[passenger_stats['risk_score'] > 0]['risk_reasons'].str.split('; ').explode().value_counts().reset_index()
    risk_reasons_analysis.columns = ['reason', 'count']
    if len(risk_reasons_analysis) > 0:
        top_reasons = risk_reasons_analysis.head(8)
        sns.barplot(y=top_reasons['reason'], x=top_reasons['count'], ax=axes[1,1], color=DARK_BLUE)
        axes[1,1].set_title('Топ-8 причин аномалий', fontweight='bold', color=DARK_BLUE)
        axes[1,1].set_xlabel('Количество случаев')

    if 'peak_activity_period' in passenger_stats.columns:
        sns.histplot(data=passenger_stats[passenger_stats['peak_activity_period'] < 5], x='peak_activity_period', hue='risk_category', ax=axes[1,2], palette=[DARK_BLUE, LIGHT_BLUE, '#6ba3d6', '#8fbce8'])
        axes[1,2].set_title('Распределение пиковой активности', fontweight='bold', color=DARK_BLUE)
        axes[1,2].set_xlabel('Пиковая активность (рейсов/день)')

    for ax in axes.flat: ax.set_facecolor('none')
    plt.tight_layout()
    plt.savefig('/home/mariia/Загрузки/Telegram Desktop/AI2/activity_patterns_analysis.png', dpi=300, bbox_inches='tight', transparent=True)
    plt.close()
    print("Графики паттернов активности сохранены")

    print("Формирование финальной таблицы...")
    final_table = passenger_stats.copy()

    def format_suspicious_activity(row):
        details = []
        if row['has_suspicious_doc']: details.append(f"Общие документы: {row['suspicious_docs_count']} шт")
        if row['activity_cluster_score'] > 1.0: details.append(f"Кластерная активность: {row['activity_cluster_score']:.2f}")
        if row['sudden_activity_increase'] > 2.0: details.append(f"Всплеск активности: {row['sudden_activity_increase']:.1f}x")
        if row['logistic_inconsistency'] > 0.1: details.append(f"Логистические проблемы: {row['logistic_inconsistency']:.2f}")
        if row['n_unique_agents'] >= 5: details.append(f"Много агентов: {row['n_unique_agents']} шт")
        if row['n_unique_documents'] > 1: details.append(f"Несколько документов: {row['n_unique_documents']} шт")
        if 'total_unique_airports' in row and row['total_unique_airports'] > 5: details.append(f"Много аэропортов: {row['total_unique_airports']} шт")
        return "; ".join(details)

    final_table['suspicious_activity_details'] = final_table.apply(format_suspicious_activity, axis=1)

    if 'departure_airports' in final_table.columns:
        final_table['departure_airports_str'] = final_table['departure_airports'].apply(lambda x: ', '.join(str(airport) for airport in x[:5]) + ('...' if len(x) > 5 else '') if isinstance(x, list) else '')
    if 'arrival_airports' in final_table.columns:
        final_table['arrival_airports_str'] = final_table['arrival_airports'].apply(lambda x: ', '.join(str(airport) for airport in x[:5]) + ('...' if len(x) > 5 else '') if isinstance(x, list) else '')
    
    final_table['suspicious_documents_str'] = final_table['suspicious_documents'].apply(lambda x: ', '.join(str(doc) for doc in x[:3]) + ('...' if len(x) > 3 else '') if isinstance(x, list) else '')

    output_columns = ['first_name', 'last_name', 'pax_birth_data', 'n_flights_total', 'n_unique_agents', 'n_unique_documents', 'days_active', 'avg_activity_frequency', 'risk_score', 'risk_category', 'risk_reasons', 'suspicious_activity_details', 'activity_cluster_score', 'sudden_activity_increase', 'logistic_inconsistency']

    if 'n_unique_departures' in final_table.columns: output_columns.extend(['n_unique_departures', 'n_unique_arrivals'])
    if 'total_unique_airports' in final_table.columns: output_columns.append('total_unique_airports')
    if 'departure_airports_str' in final_table.columns: output_columns.append('departure_airports_str')
    if 'arrival_airports_str' in final_table.columns: output_columns.append('arrival_airports_str')
    output_columns.extend(['suspicious_documents_str', 'suspicious_docs_count'])

    final_output = final_table[output_columns].copy()

    print("Сохранение результатов...")
    output_path = r"/home/mariia/Загрузки/Telegram Desktop/AI2/final_results_activity_patterns.csv"
    final_output.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Детальные результаты: {output_path}")

    suspicious_passengers = final_output[final_output['risk_score'] >= 50].sort_values('risk_score', ascending=False)
    suspicious_path = r"/home/mariia/Загрузки/Telegram Desktop/AI2/suspicious_passengers_activity_patterns.csv"
    suspicious_passengers.to_csv(suspicious_path, index=False, encoding='utf-8')
    print(f"Подозрительные пассажиры: {suspicious_path}")

    print(f"АНАЛИЗ ПАТТЕРНОВ АКТИВНОСТИ ЗАВЕРШЕН!")
    print(f"ИТОГИ:")
    print(f"   - Всего пассажиров: {len(passenger_stats)}")
    print(f"   - Подозрительных: {len(suspicious_passengers)}")
    print(f"   - С общими документами: {passenger_stats['has_suspicious_doc'].sum()}")
    print(f"   - С кластерной активностью: {(passenger_stats['activity_cluster_score'] > 1).sum()}")
    print(f"   - С резкими всплесками активности: {(passenger_stats['sudden_activity_increase'] > 2).sum()}")
    print(f"   - С логистическими проблемами: {(passenger_stats['logistic_inconsistency'] > 0.1).sum()}")

    print("ТОП-5 САМЫХ ПОДОЗРИТЕЛЬНЫХ ПАССАЖИРОВ:")
    print("=" * 120)
    for i, (_, row) in enumerate(suspicious_passengers.head(5).iterrows(), 1):
        print(f"{i}. {row['first_name']} {row['last_name']} ({row['pax_birth_data']})")
        print(f"   Рейсов: {row['n_flights_total']} | Агентов: {row['n_unique_agents']} | Документов: {row['n_unique_documents']}")
        print(f"   Активность: {row['days_active']} дней | Частота: {row['avg_activity_frequency']:.2f} рейсов/период")
        print(f"   Risk: {row['risk_score']} ({row['risk_category']})")
        print(f"   Паттерны: Кластеры={row['activity_cluster_score']:.2f}, Всплески={row['sudden_activity_increase']:.1f}x, Логистика={row['logistic_inconsistency']:.2f}")
        print(f"   Детали: {row['suspicious_activity_details']}")
        if 'departure_airports_str' in row and row['departure_airports_str']: print(f"   Аэропорты вылета: {row['departure_airports_str']}")
        if 'arrival_airports_str' in row and row['arrival_airports_str']: print(f"   Аэропорты прилета: {row['arrival_airports_str']}")
        print("-" * 120)

    return valid_df, passenger_stats

def analyze_specific_passenger(first_name, last_name, birth_date, valid_df, passenger_stats):
    print(f"ДЕТАЛЬНЫЙ АНАЛИЗ ПАССАЖИРА: {first_name} {last_name} ({birth_date})")
    print("=" * 100)
    
    passenger_mask = ((passenger_stats['first_name'] == first_name) & (passenger_stats['last_name'] == last_name) & (passenger_stats['pax_birth_data'] == birth_date))
    if not passenger_mask.any():
        print(f"Пассажир не найден в статистике")
        return None
    
    passenger_data = passenger_stats[passenger_mask].iloc[0]
    flight_mask = ((valid_df['first_name'] == first_name) & (valid_df['last_name'] == last_name) & (valid_df['pax_birth_data'] == birth_date))
    passenger_flights = valid_df[flight_mask].copy()
    passenger_flights['flight_date'] = pd.to_datetime(passenger_flights['flight_date'])
    passenger_flights = passenger_flights.sort_values('flight_date')
    
    print(f"ОСНОВНАЯ ИНФОРМАЦИЯ:")
    print(f"   • Всего перелетов: {passenger_data['n_flights_total']}")
    print(f"   • Уникальных агентов: {passenger_data['n_unique_agents']}")
    print(f"   • Уникальных документов: {passenger_data['n_unique_documents']}")
    print(f"   • Период активности: {passenger_data['days_active']} дней")
    print(f"   • Уровень риска: {passenger_data['risk_score']} ({passenger_data['risk_category']})")
    print(f"   • Средняя частота: {passenger_data.get('avg_activity_frequency', 0):.2f} рейсов/период")
    
    if 'activity_cluster_score' in passenger_data: print(f"   • Оценка кластерной активности: {passenger_data['activity_cluster_score']:.2f}")
    if 'sudden_activity_increase' in passenger_data: print(f"   • Всплеск активности: {passenger_data['sudden_activity_increase']:.1f}x")
    if 'logistic_inconsistency' in passenger_data: print(f"   • Логистические проблемы: {passenger_data['logistic_inconsistency']:.2f}")
    
    print(f"ПРИЧИНЫ РИСКА:")
    print(f"   {passenger_data['risk_reasons']}")
    
    if passenger_data['has_suspicious_doc']:
        print(f"ПОДОЗРИТЕЛЬНЫЕ ДОКУМЕНТЫ:")
        for i, doc in enumerate(passenger_data['suspicious_documents'][:5], 1):
            print(f"   {i}. {doc}")
        if len(passenger_data['suspicious_documents']) > 5: print(f"   ... и еще {len(passenger_data['suspicious_documents']) - 5} документов")
    
    print(f"ИНФОРМАЦИЯ ОБ АГЕНТАХ:")
    agents = passenger_flights['agent_info'].value_counts()
    for agent, count in agents.head(10).items():
        print(f"   • {agent}: {count} рейсов")
    
    if 'departure' in passenger_flights.columns:
        print(f"ТОП АЭРОПОРТОВ ВЫЛЕТА:")
        departures = passenger_flights['departure'].value_counts().head(5)
        for airport, count in departures.items():
            print(f"   • {airport}: {count} вылетов")
    
    if 'arrival' in passenger_flights.columns:
        print(f"ТОП АЭРОПОРТОВ ПРИЛЕТА:")
        arrivals = passenger_flights['arrival'].value_counts().head(5)
        for airport, count in arrivals.items():
            print(f"   • {airport}: {count} прилетов")
    
    return {'summary': passenger_data, 'flights': passenger_flights}

def search_passengers(search_term, passenger_stats, max_results=10):
    search_term = search_term.lower()
    mask = (passenger_stats['first_name'].str.lower().str.contains(search_term, na=False) | passenger_stats['last_name'].str.lower().str.contains(search_term, na=False) | passenger_stats['pax_birth_data'].str.lower().str.contains(search_term, na=False))
    results = passenger_stats[mask].head(max_results)
    
    if len(results) == 0:
        print(f"Пассажиры по запросу '{search_term}' не найдены")
        return None
    
    print(f"НАЙДЕНО ПАССАЖИРОВ: {len(results)}")
    print("=" * 80)
    for i, (_, passenger) in enumerate(results.iterrows(), 1):
        print(f"{i}. {passenger['first_name']} {passenger['last_name']} ({passenger['pax_birth_data']})")
        print(f"   Рейсов: {passenger['n_flights_total']} | Агентов: {passenger['n_unique_agents']} | Risk: {passenger['risk_score']} ({passenger['risk_category']})")
        if i < len(results): print("-" * 80)
    
    return results

def analyze_specific_passenger(first_name, last_name, birth_date, valid_df, passenger_stats):
    """Детальный анализ конкретного пассажира с возвратом детальных данных"""
    print(f"ДЕТАЛЬНЫЙ АНАЛИЗ ПАССАЖИРА: {first_name} {last_name} ({birth_date})")
    print("=" * 100)
    
    passenger_mask = ((passenger_stats['first_name'] == first_name) & 
                     (passenger_stats['last_name'] == last_name) & 
                     (passenger_stats['pax_birth_data'] == birth_date))
    
    if not passenger_mask.any():
        print(f"Пассажир не найден в статистике")
        return None
    
    passenger_data = passenger_stats[passenger_mask].iloc[0]
    flight_mask = ((valid_df['first_name'] == first_name) & 
                  (valid_df['last_name'] == last_name) & 
                  (valid_df['pax_birth_data'] == birth_date))
    
    passenger_flights = valid_df[flight_mask].copy()
    passenger_flights['flight_date'] = pd.to_datetime(passenger_flights['flight_date'])
    passenger_flights = passenger_flights.sort_values('flight_date')
    
    # Очищаем данные
    if 'departure' in passenger_flights.columns:
        passenger_flights['departure'] = passenger_flights['departure'].astype(str).replace('nan', '').replace('None', '')
    if 'arrival' in passenger_flights.columns:
        passenger_flights['arrival'] = passenger_flights['arrival'].astype(str).replace('nan', '').replace('None', '')
    if 'agent_info' in passenger_flights.columns:
        passenger_flights['agent_info'] = passenger_flights['agent_info'].astype(str).replace('nan', '').replace('None', '')
    if 'document_norm' in passenger_flights.columns:
        passenger_flights['document_norm'] = passenger_flights['document_norm'].astype(str).replace('nan', '').replace('None', '')
    if 'flight_code' in passenger_flights.columns:
        passenger_flights['flight_code'] = passenger_flights['flight_code'].astype(str).replace('nan', '').replace('None', '')
    
    print(f"ОСНОВНАЯ ИНФОРМАЦИЯ:")
    print(f"   • Всего перелетов: {passenger_data['n_flights_total']}")
    print(f"   • Уникальных агентов: {passenger_data['n_unique_agents']}")
    print(f"   • Уникальных документов: {passenger_data['n_unique_documents']}")
    print(f"   • Период активности: {passenger_data['days_active']} дней")
    print(f"   • Уровень риска: {passenger_data['risk_score']} ({passenger_data['risk_category']})")
    print(f"   • Средняя частота: {passenger_data.get('avg_activity_frequency', 0):.2f} рейсов/период")
    
    if 'activity_cluster_score' in passenger_data: 
        print(f"   • Оценка кластерной активности: {passenger_data['activity_cluster_score']:.2f}")
    if 'sudden_activity_increase' in passenger_data: 
        print(f"   • Всплеск активности: {passenger_data['sudden_activity_increase']:.1f}x")
    if 'logistic_inconsistency' in passenger_data: 
        print(f"   • Логистические проблемы: {passenger_data['logistic_inconsistency']:.2f}")
    
    print(f"ПРИЧИНЫ РИСКА:")
    print(f"   {passenger_data['risk_reasons']}")
    
    if passenger_data['has_suspicious_doc']:
        print(f"ПОДОЗРИТЕЛЬНЫЕ ДОКУМЕНТЫ:")
        for i, doc in enumerate(passenger_data['suspicious_documents'][:5], 1):
            print(f"   {i}. {doc}")
        if len(passenger_data['suspicious_documents']) > 5: 
            print(f"   ... и еще {len(passenger_data['suspicious_documents']) - 5} документов")
    
    print(f"ИНФОРМАЦИЯ ОБ АГЕНТАХ:")
    agents = passenger_flights['agent_info'].value_counts()
    for agent, count in agents.head(10).items():
        print(f"   • {agent}: {count} рейсов")
    
    if 'departure' in passenger_flights.columns:
        print(f"ТОП АЭРОПОРТОВ ВЫЛЕТА:")
        departures = passenger_flights['departure'].value_counts().head(5)
        for airport, count in departures.items():
            if airport and airport != 'nan' and airport != 'None':
                print(f"   • {airport}: {count} вылетов")
    
    if 'arrival' in passenger_flights.columns:
        print(f"ТОП АЭРОПОРТОВ ПРИЛЕТА:")
        arrivals = passenger_flights['arrival'].value_counts().head(5)
        for airport, count in arrivals.items():
            if airport and airport != 'nan' and airport != 'None':
                print(f"   • {airport}: {count} прилетов")
    
    print(f"\nДЕТАЛЬНАЯ ТАБЛИЦА ПЕРЕЛЕТОВ:")
    print("=" * 80)
    
    available_columns = []
    if 'flight_date' in passenger_flights.columns:
        available_columns.append('flight_date')
    if 'flight_code' in passenger_flights.columns:
        available_columns.append('flight_code')
    if 'departure' in passenger_flights.columns:
        available_columns.append('departure')
    if 'arrival' in passenger_flights.columns:
        available_columns.append('arrival')
    if 'agent_info' in passenger_flights.columns:
        available_columns.append('agent_info')
    if 'document_norm' in passenger_flights.columns:
        available_columns.append('document_norm')
    

    display_flights = passenger_flights[available_columns].copy()
    
    if 'flight_date' in display_flights.columns:
        display_flights['flight_date'] = display_flights['flight_date'].dt.strftime('%Y-%m-%d')
    
    flights_displayed = 0
    for _, flight in display_flights.iterrows():
        parts = []
        
        if 'flight_date' in flight:
            parts.append(str(flight['flight_date']))
        
        if 'flight_code' in flight and flight['flight_code']:
            parts.append(str(flight['flight_code']))
        else:
            parts.append("N/A")
        
        if 'departure' in flight and 'arrival' in flight and flight['departure'] and flight['arrival']:
            route = f"{flight['departure']} → {flight['arrival']}"
            parts.append(route)
        else:
            parts.append("N/A → N/A")
        
        if 'agent_info' in flight and flight['agent_info']:
            agent = str(flight['agent_info'])
            if len(agent) > 20:
                agent = agent[:20] + "..."
            parts.append(agent)
        else:
            parts.append("N/A")
        
        if 'document_norm' in flight and flight['document_norm']:
            parts.append(str(flight['document_norm']))
        else:
            parts.append("N/A")
        
        # Выводим строку
        print(" | ".join(parts))
        flights_displayed += 1
        
        if flights_displayed >= 50:  
            remaining = len(display_flights) - 50
            if remaining > 0:
                print(f"... и еще {remaining} перелетов")
            break
    
    print("=" * 80)
    
    return {
        'summary': passenger_data, 
        'flights': passenger_flights,
        'departure_airports': list(passenger_flights['departure'].unique()) if 'departure' in passenger_flights.columns else [],
        'arrival_airports': list(passenger_flights['arrival'].unique()) if 'arrival' in passenger_flights.columns else []
    }


def create_passenger_flight_map(passenger_row, detailed_data=None, output_path=None):
    """Создание карты перелетов для конкретного пассажира"""
    visualizer = FlightMapVisualizer()
    
    print("Создание данных о перелетах пассажира...")
    
    # Используем детальные данные если они предоставлены
    if detailed_data is not None and 'flights' in detailed_data:
        print("Используются детальные данные о перелетах...")
        flight_df = detailed_data['flights'][['departure', 'arrival']].copy()
        
        # Очищаем данные
        flight_df = flight_df.dropna()
        flight_df = flight_df[(flight_df['departure'] != '') & (flight_df['departure'] != 'nan') & 
                             (flight_df['arrival'] != '') & (flight_df['arrival'] != 'nan')]
        
        # Создаем flight_id для каждого перелета
        flight_df['flight_id'] = [f"FL{i+1:03d}" for i in range(len(flight_df))]
        
        print(f"Найдено {len(flight_df)} действительных перелетов")
        
    else:
        # Старый метод с агрегированными данными (для обратной совместимости)
        departure_airports = []
        arrival_airports = []
        
        if 'departure_airports_str' in passenger_row and pd.notna(passenger_row['departure_airports_str']):
            departure_airports = [ap.strip() for ap in str(passenger_row['departure_airports_str']).split(',')]
        
        if 'arrival_airports_str' in passenger_row and pd.notna(passenger_row['arrival_airports_str']):
            arrival_airports = [ap.strip() for ap in str(passenger_row['arrival_airports_str']).split(',')]
        
        print(f"Аэропорты вылета: {departure_airports}")
        print(f"Аэропорты прилета: {arrival_airports}")
        
        flights = []
        for i, dep_airport in enumerate(departure_airports):
            if i < len(arrival_airports):
                arr_airport = arrival_airports[i % len(arrival_airports)]
                flights.append({
                    'departure': dep_airport, 
                    'arrival': arr_airport, 
                    'flight_id': f"FL{i+1:03d}"
                })
        
        flight_df = pd.DataFrame(flights)
    
    if len(flight_df) == 0:
        print("Нет данных о перелетах для построения карты")
        return None
    
    print(f"Создано {len(flight_df)} перелетов")
    
    print("Создание координат аэропортов...")
    all_airports = set()
    if 'departure' in flight_df.columns:
        all_airports.update(flight_df['departure'].unique())
    if 'arrival' in flight_df.columns:
        all_airports.update(flight_df['arrival'].unique())
    
    # Удаляем пустые значения
    all_airports = {ap for ap in all_airports if ap and ap != 'nan' and ap != 'None'}
    
    print(f"Найдено уникальных аэропортов: {len(all_airports)}")
    print(f"Аэропорты: {list(all_airports)}")
    
    airport_data = []
    for airport in all_airports:
        coords = visualizer.fast_geocode_airport(airport)
        if coords:
            dep_count = len(flight_df[flight_df['departure'] == airport])
            arr_count = len(flight_df[flight_df['arrival'] == airport])
            
            airport_data.append({
                'airport': airport,
                'lat': coords[0],
                'lon': coords[1],
                'departures_count': dep_count,
                'arrivals_count': arr_count,
                'flights_count': dep_count + arr_count
            })
        else:
            print(f"Не удалось найти координаты для аэропорта: {airport}")
    
    airport_df = pd.DataFrame(airport_data)
    
    if len(airport_df) == 0:
        print("Не удалось получить координаты аэропортов")
        return None
    
    print("Создание карты перелетов пассажира...")
    if output_path is None:
        output_path = f"/home/mariia/Загрузки/Telegram Desktop/AI2/passenger_{passenger_row['first_name']}_{passenger_row['last_name']}_flights.html"
    
    fig = go.Figure()
    
    # Добавляем аэропорты на карту
    fig.add_trace(go.Scattergeo(
        lon=airport_df['lon'],
        lat=airport_df['lat'],
        text=airport_df['airport'] + '<br>' + 
             'Вылеты: ' + airport_df['departures_count'].astype(str) + '<br>' +
             'Прилеты: ' + airport_df['arrivals_count'].astype(str),
        mode='markers',
        marker=dict(
            size=15,
            color='red',
            opacity=0.8,
            sizemode='area'
        ),
        name='Аэропорты'
    ))
    
    # Добавляем линии перелетов
    flight_lines = []
    for i, (_, flight) in enumerate(flight_df.iterrows()):
        dep_airport = flight['departure']
        arr_airport = flight['arrival']
        
        dep_coords = visualizer.airport_coords.get(dep_airport)
        arr_coords = visualizer.airport_coords.get(arr_airport)
        
        if dep_coords and arr_coords:
            flight_lines.append({
                'dep_lon': dep_coords[1],
                'dep_lat': dep_coords[0],
                'arr_lon': arr_coords[1],
                'arr_lat': arr_coords[0],
                'route': f"{dep_airport} → {arr_airport}"
            })
    
    print(f"Создано {len(flight_lines)} линий перелетов")
    
    # Добавляем линии на карту
    if flight_lines:
        lons = []
        lats = []
        hover_texts = []
        
        for line in flight_lines:
            lons.extend([line['dep_lon'], line['arr_lon'], None])
            lats.extend([line['dep_lat'], line['arr_lat'], None])
            hover_texts.extend([line['route'], line['route'], None])
        
        fig.add_trace(go.Scattergeo(
            lon=lons,
            lat=lats,
            text=hover_texts,
            hoverinfo='text',
            mode='lines',
            line=dict(width=2, color='blue'),
            opacity=0.6,
            name='Перелеты'
        ))
    
    # Добавляем информацию о пассажире в заголовок
    risk_category = passenger_row.get('risk_category', 'НЕИЗВЕСТНО')
    risk_score = passenger_row.get('risk_score', 'НЕИЗВЕСТНО')
    n_flights = passenger_row.get('n_flights_total', 'НЕИЗВЕСТНО')
    
    title = f"КАРТА ПЕРЕЛЕТОВ: {passenger_row['first_name']} {passenger_row['last_name']}<br>"
    title += f"<sub>Категория риска: {risk_category} | Баллы риска: {risk_score} | Всего перелетов: {n_flights}</sub>"
    
    # Настройка карты
    fig.update_layout(
        title_text=title,
        showlegend=True,
        geo=dict(
            scope='world',
            projection_type='equirectangular',
            showland=True,
            landcolor='rgb(243, 243, 243)',
            countrycolor='rgb(204, 204, 204)',
            coastlinecolor='rgb(204, 204, 204)',
            showocean=True,
            oceancolor='rgb(222, 243, 246)'
        )
    )
    
    # Сохраняем карту
    fig.write_html(output_path)
    print(f"Карта перелетов сохранена: {output_path}")
    
    return fig

def create_world_flights_map(sample_size=5000):
    visualizer = FlightMapVisualizer()
    
    try:
        df = pd.read_csv(r"/home/mariia/Загрузки/Telegram Desktop/AI2/data_staging/merged_all_detailed.csv", low_memory=False)
        print(f"Загружено {len(df):,} строк данных")
    except FileNotFoundError:
        print("Файл данных не найден")
        return
    
    print("Подготовка данных...")
    if sample_size and len(df) > sample_size:
        df = df.sample(sample_size)
        print(f"Взята выборка: {sample_size} перелетов")
    
    flight_data = df.copy()
    text_columns = ['departure', 'arrival']
    for col in text_columns:
        if col in flight_data.columns:
            flight_data[col] = flight_data[col].astype(str).replace('nan', '')
    
    if 'departure' in flight_data.columns and 'arrival' in flight_data.columns:
        mask = (flight_data['departure'] != '') & (flight_data['arrival'] != '')
        flight_data = flight_data[mask]
    
    print(f"Обработано перелетов: {len(flight_data)}")
    
    print("Создание координат аэропортов...")
    all_airports = set()
    if 'departure' in flight_data.columns: all_airports.update(flight_data['departure'].unique())
    if 'arrival' in flight_data.columns: all_airports.update(flight_data['arrival'].unique())
    
    print(f"Найдено уникальных аэропортов: {len(all_airports)}")
    if len(all_airports) > 150:
        airport_counts = {}
        for airport in all_airports:
            dep_count = len(flight_data[flight_data['departure'] == airport])
            arr_count = len(flight_data[flight_data['arrival'] == airport])
            airport_counts[airport] = dep_count + arr_count
        top_airports = sorted(airport_counts.items(), key=lambda x: x[1], reverse=True)[:150]
        all_airports = set([airport for airport, count in top_airports])
        print(f"Ограничиваем геокодирование до 150 самых частых аэропортов")
    
    airport_data = []
    for airport in all_airports:
        coords = visualizer.fast_geocode_airport(airport)
        if coords:
            airport_data.append({'airport': airport, 'lat': coords[0], 'lon': coords[1], 'flights_count': len(flight_data[flight_data['departure'] == airport]) + len(flight_data[flight_data['arrival'] == airport])})
    
    airport_df = pd.DataFrame(airport_data)
    print(f"Получены координаты для {len(airport_df)} аэропортов")
    
    print("Создание карты мира...")
    fig = go.Figure()
    fig.add_trace(go.Scattergeo(lon=airport_df['lon'], lat=airport_df['lat'], text=airport_df['airport'] + '<br>Рейсов: ' + airport_df['flights_count'].astype(str), mode='markers', marker=dict(size=10, color='red', opacity=0.8, sizemode='area'), name='Аэропорты'))
    
    print("Добавление перелетов...")
    if len(flight_data) > 5000:
        route_counts = flight_data.groupby(['departure', 'arrival']).size().reset_index()
        route_counts.columns = ['departure', 'arrival', 'count']
        top_routes = route_counts.nlargest(2000, 'count')
        flight_data_to_use = top_routes
    else:
        flight_data_to_use = flight_data
    
    flight_lines = []
    for _, flight in flight_data_to_use.iterrows():
        dep_airport = flight['departure']
        arr_airport = flight['arrival']
        dep_coords = visualizer.airport_coords.get(dep_airport)
        arr_coords = visualizer.airport_coords.get(arr_airport)
        if dep_coords and arr_coords:
            flight_lines.append({'dep_lon': dep_coords[1], 'dep_lat': dep_coords[0], 'arr_lon': arr_coords[1], 'arr_lat': arr_coords[0], 'route': f"{dep_airport} → {arr_airport}"})
    
    print(f"Создано {len(flight_lines)} линий перелетов")
    if flight_lines:
        lons, lats = [], []
        for line in flight_lines:
            lons.extend([line['dep_lon'], line['arr_lon'], None])
            lats.extend([line['dep_lat'], line['arr_lat'], None])
        
        fig.add_trace(go.Scattergeo(lon=lons, lat=lats, mode='lines', line=dict(width=1, color='blue'), opacity=0.2, showlegend=False))
    
    fig.update_layout(title_text='КАРТА ПЕРЕЛЕТОВ - Самые популярные маршруты', showlegend=True, geo=dict(scope='world', projection_type='equirectangular', showland=True, landcolor='rgb(243, 243, 243)', countrycolor='rgb(204, 204, 204)', coastlinecolor='rgb(204, 204, 204)', showocean=True, oceancolor='rgb(222, 243, 246)'))
    output_path = "/home/mariia/Загрузки/Telegram Desktop/AI2/world_flights_map.html"
    fig.write_html(output_path)
    print(f"Карта мировых перелетов сохранена: {output_path}")
    return fig

def run_passenger_analysis(valid_df, passenger_stats):
    print("СИСТЕМА ДЕТАЛЬНОГО АНАЛИЗА ПАССАЖИРОВ")
    print("=" * 60)
    
    while True:
        print("\nВыберите действие:")
        print("1. Поиск пассажира")
        print("2. Анализ подозрительных пассажиров")
        print("3. Возврат в главное меню")
        
        choice = input("Введите номер действия (1-3): ").strip()
        
        if choice == '1':
            search_term = input("Введите имя, фамилию или дату рождения для поиска: ").strip()
            if search_term:
                results = search_passengers(search_term, passenger_stats)
                if results is not None and len(results) > 0:
                    if len(results) == 1:
                        passenger = results.iloc[0]
                        analysis_result = analyze_specific_passenger(passenger['first_name'], passenger['last_name'], passenger['pax_birth_data'], valid_df, passenger_stats)
                        if analysis_result:
                            map_choice = input("Создать карту перелетов для этого пассажира? (y/n): ").strip().lower()
                            if map_choice == 'y':
                                create_passenger_flight_map(passenger, analysis_result)
                    else:
                        try:
                            passenger_num = int(input(f"Выберите пассажира (1-{len(results)}): ")) - 1
                            if 0 <= passenger_num < len(results):
                                passenger = results.iloc[passenger_num]
                                analysis_result = analyze_specific_passenger(passenger['first_name'], passenger['last_name'], passenger['pax_birth_data'], valid_df, passenger_stats)
                                if analysis_result:
                                    map_choice = input("Создать карту перелетов для этого пассажира? (y/n): ").strip().lower()
                                    if map_choice == 'y':
                                        create_passenger_flight_map(passenger)
                            else:
                                print("Неверный номер пассажира")
                        except ValueError:
                            print("Введите корректный номер")
        
        elif choice == '2':
            suspicious = passenger_stats[passenger_stats['is_suspicious'] == True].sort_values('risk_score', ascending=False)
            print("ТОП-10 ПОДОЗРИТЕЛЬНЫХ ПАССАЖИРОВ:")
            print("=" * 80)
            for i, (_, passenger) in enumerate(suspicious.head(10).iterrows(), 1):
                print(f"{i}. {passenger['first_name']} {passenger['last_name']} ({passenger['pax_birth_data']})")
                print(f"   Рейсов: {passenger['n_flights_total']} | Агентов: {passenger['n_unique_agents']} | Risk: {passenger['risk_score']}")
                print(f"   {passenger['risk_reasons']}")
                if i < min(10, len(suspicious)): print("-" * 80)
            
            try:
                passenger_num = int(input("Выберите пассажира для детального анализа (1-10): ")) - 1
                if 0 <= passenger_num < len(suspicious.head(10)):
                    passenger = suspicious.iloc[passenger_num]
                    analysis_result = analyze_specific_passenger(passenger['first_name'], passenger['last_name'], passenger['pax_birth_data'], valid_df, passenger_stats)
                    if analysis_result:
                        map_choice = input("Создать карту перелетов для этого пассажира? (y/n): ").strip().lower()
                        if map_choice == 'y':
                            create_passenger_flight_map(passenger)
                else:
                    print("Неверный номер пассажира")
            except ValueError:
                print("Введите корректный номер")
        
        elif choice == '3':
            print("Возврат в главное меню...")
            break
        
        else:
            print("Неверный выбор. Попробуйте снова.")

def main():
    print("СИСТЕМА АНАЛИЗА ПАССАЖИРСКОЙ АКТИВНОСТИ")
    print("=" * 50)
    
    valid_df = None
    passenger_stats = None
    
    while True:
        print("\nГЛАВНОЕ МЕНЮ:")
        print("1. Запустить полный анализ данных")
        print("2. Детальный анализ пассажиров")
        print("3. Создать карту мировых перелетов")
        print("4. Выход")
        
        choice = input("Выберите действие (1-4): ").strip()
        
        if choice == '1':
            print("ЗАПУСК ПОЛНОГО АНАЛИЗА ДАННЫХ")
            valid_df, passenger_stats = main_analysis()
            
        elif choice == '2':
            if valid_df is None or passenger_stats is None:
                print("Сначала выполните полный анализ данных (пункт 1)")
                continue
            print("ЗАПУСК ДЕТАЛЬНОГО АНАЛИЗА ПАССАЖИРОВ")
            run_passenger_analysis(valid_df, passenger_stats)
            
        elif choice == '3':
            print("СОЗДАНИЕ КАРТЫ МИРОВЫХ ПЕРЕЛЕТОВ")
            print("Выберите размер выборки:")
            print("1. 1000 перелетов (быстро)")
            print("2. 5000 перелетов (баланс)")
            print("3. 10000 перелетов (качество)")
            print("4. Все данные (медленно)")
            
            size_choice = input("Выберите вариант (1-4): ").strip()
            sample_sizes = {'1': 1000, '2': 5000, '3': 10000, '4': None}
            sample_size = sample_sizes.get(choice, 5000)
            create_world_flights_map(sample_size)
                
        elif choice == '4':
            print("Завершение работы программы.")
            break
            
        else:
            print("Неверный выбор. Пожалуйста, выберите от 1 до 4.")

if __name__ == "__main__":
    main()
