import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Настройка стиля графиков
plt.style.use('default')
sns.set_palette("dark:blue")
DARK_BLUE = "#1f4e79"
LIGHT_BLUE = "#4a7bb5"

def main_analysis():
    """Основной анализ паттернов активности"""
    print("🚀 ЗАПУСК УЛУЧШЕННОГО АНАЛИЗА ПАТТЕРНОВ АКТИВНОСТИ...")

    # Загрузка данных
    df = pd.read_csv(r"/home/mariia/Загрузки/Telegram Desktop/AI2/data_staging/merged_all_detailed.csv", low_memory=False)
    print(f"✅ Загружено {len(df):,} строк, {len(df.columns)} колонок")

    # ==================================================
    # 1. ОПТИМИЗИРОВАННАЯ ПРЕДОБРАБОТКА ДАННЫХ
    # ==================================================
    print("🔍 Оптимизированная предобработка данных...")

    # Быстрое преобразование типов и очистка
    text_columns = ['document_norm', 'first_name', 'last_name', 'pax_birth_data', 'departure', 'arrival', 'agent_info']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).replace('nan', '').replace('None', '')

    # Фильтрация только корректных данных
    mask = (
        (df['first_name'] != '') & 
        (df['last_name'] != '') & 
        (df['pax_birth_data'] != '') &
        (df['document_norm'] != '')
    )
    valid_df = df[mask].copy()
    print(f"📊 Валидных записей: {len(valid_df):,}")

    # ==================================================
    # 2. ОПТИМИЗИРОВАННЫЙ АНАЛИЗ ДОКУМЕНТОВ
    # ==================================================
    print("🔍 Оптимизированный анализ документов...")

    # Анализ документов с оптимизацией
    doc_stats = valid_df.groupby('document_norm').agg(
        unique_passengers=('first_name', 'nunique'),
        total_flights=('flight_code', 'count')
    ).reset_index()

    suspicious_docs = doc_stats[doc_stats['unique_passengers'] > 1]
    print(f"📋 Найдено подозрительных документов: {len(suspicious_docs)}")

    # ==================================================
    # 3. УЛУЧШЕННАЯ ГРУППИРОВКА С АНАЛИЗОМ ПАТТЕРНОВ АКТИВНОСТИ
    # ==================================================
    print("🔍 Группировка данных с анализом паттернов активности...")

    # Создаем уникальный ID пассажира
    valid_df['passenger_id'] = (
        valid_df['first_name'] + '|' + 
        valid_df['last_name'] + '|' + 
        valid_df['pax_birth_data']
    )

    # Преобразуем дату в datetime
    valid_df['flight_date'] = pd.to_datetime(valid_df['flight_date'], errors='coerce')
    
    # Удаляем строки с некорректными датами
    valid_df = valid_df.dropna(subset=['flight_date'])
    print(f"📅 Записей с корректными датами: {len(valid_df)}")

    # Функция для анализа паттернов активности пассажира
    def analyze_activity_patterns(passenger_data):
        """Анализирует паттерны активности пассажира с правильным расчетом всплесков"""
        if len(passenger_data) < 2:
            return {
                'activity_cluster_score': 0,
                'sudden_activity_increase': 0,
                'logistic_inconsistency': 0,
                'peak_activity_period': 0,
                'avg_flights_per_period': 0
            }
        
        # Сортируем по дате
        passenger_data = passenger_data.sort_values('flight_date')
        dates = passenger_data['flight_date'].sort_values()
        
        # Анализ временных паттернов
        date_diff = dates.diff().dt.days.fillna(0)
        
        # Находим кластеры активности (перелеты в близкие даты)
        activity_clusters = []
        current_cluster = []
        
        for i, diff in enumerate(date_diff):
            if diff <= 2:  # Перелеты в течение 2 дней считаем одним кластером
                current_cluster.append(i)
            else:
                if len(current_cluster) > 1:
                    activity_clusters.append(current_cluster)
                current_cluster = [i]
        
        if len(current_cluster) > 1:
            activity_clusters.append(current_cluster)
        
        # Оценка кластерной активности
        cluster_score = sum(len(cluster) ** 1.5 for cluster in activity_clusters) / len(passenger_data) if len(passenger_data) > 0 else 0
        
        # ПРАВИЛЬНЫЙ РАСЧЕТ РЕЗКИХ ВСПЛЕСКОВ АКТИВНОСТИ
        sudden_increase = 0
        if len(passenger_data) >= 4:  # Нужно достаточно данных для анализа
            # Группируем по неделям для анализа временных рядов
            passenger_data_copy = passenger_data.copy()
            passenger_data_copy['week'] = passenger_data_copy['flight_date'].dt.isocalendar().week
            passenger_data_copy['year'] = passenger_data_copy['flight_date'].dt.year
            weekly_activity = passenger_data_copy.groupby(['year', 'week']).size().reset_index(name='flights')
            
            if len(weekly_activity) >= 3:  # Нужно хотя бы 3 недели для анализа
                # Сортируем по времени
                weekly_activity = weekly_activity.sort_values(['year', 'week'])
                weekly_flights = weekly_activity['flights'].values
                
                # Ищем самый большой всплеск относительно предыдущего периода
                max_spike = 0
                for i in range(2, len(weekly_flights)):
                    # Сравниваем с медианным значением предыдущих 2 недель
                    previous_median = np.median(weekly_flights[i-2:i])
                    if previous_median > 0:
                        spike_ratio = weekly_flights[i] / previous_median
                        if spike_ratio > max_spike:
                            max_spike = spike_ratio
                
                sudden_increase = max_spike
        
        # Анализ логистической согласованности
        logistic_issues = 0
        if 'departure' in passenger_data.columns and 'arrival' in passenger_data.columns:
            # Группируем по дням и проверяем логистику
            daily_activity = passenger_data.groupby(passenger_data['flight_date'].dt.date).agg({
                'departure': list,
                'arrival': list
            }).reset_index()
            
            for _, day in daily_activity.iterrows():
                if len(day['departure']) > 1:
                    # В один день должно быть: прилет -> вылет из того же аэропорта
                    arrivals = set(day['arrival'])
                    departures = set(day['departure'])
                    
                    # Если есть вылет из аэропорта, куда не было прилета в этот день - логистическая проблема
                    if len(departures - arrivals) > 0:
                        logistic_issues += len(departures - arrivals)
        
        logistic_inconsistency = logistic_issues / len(passenger_data) if len(passenger_data) > 0 else 0
        
        # Период пиковой активности
        if len(dates) > 0:
            total_days = (dates.max() - dates.min()).days + 1
            peak_period = len(passenger_data) / total_days if total_days > 0 else 0
        else:
            peak_period = 0
        
        return {
            'activity_cluster_score': cluster_score,
            'sudden_activity_increase': sudden_increase,
            'logistic_inconsistency': logistic_inconsistency,
            'peak_activity_period': peak_period,
            'avg_flights_per_period': len(passenger_data) / 30 if len(passenger_data) > 30 else len(passenger_data) / ((dates.max() - dates.min()).days + 1) if len(dates) > 0 else 0
        }

    # Основные агрегации
    print("🔍 Анализ паттернов активности для каждого пассажира...")

    # Сначала делаем базовую агрегацию
    basic_stats = valid_df.groupby('passenger_id').agg({
        'flight_code': 'count',
        'document_norm': 'nunique',
        'agent_info': 'nunique',
        'flight_date': ['min', 'max'],
        'departure': lambda x: list(x.unique()) if 'departure' in valid_df.columns else [],
        'arrival': lambda x: list(x.unique()) if 'arrival' in valid_df.columns else []
    }).reset_index()

    # Выравниваем колонки
    new_columns = ['passenger_id', 'n_flights_total', 'n_unique_documents', 'n_unique_agents', 'first_flight', 'last_flight']
    if 'departure' in valid_df.columns:
        new_columns.extend(['departure_airports'])
    if 'arrival' in valid_df.columns:
        new_columns.extend(['arrival_airports'])

    basic_stats.columns = new_columns

    # Анализ паттернов активности для каждого пассажира
    activity_patterns = []
    for passenger_id in basic_stats['passenger_id']:
        passenger_data = valid_df[valid_df['passenger_id'] == passenger_id]
        patterns = analyze_activity_patterns(passenger_data)
        patterns['passenger_id'] = passenger_id
        activity_patterns.append(patterns)

    activity_df = pd.DataFrame(activity_patterns)

    # Объединяем с базовой статистикой
    passenger_stats = basic_stats.merge(activity_df, on='passenger_id', how='left')

    # Разделяем passenger_id обратно на компоненты
    passenger_stats[['first_name', 'last_name', 'pax_birth_data']] = (
        passenger_stats['passenger_id'].str.split('|', expand=True)
    )

    # ==================================================
    # 4. ОПТИМИЗИРОВАННЫЙ РАСЧЕТ МЕТРИК С ПАТТЕРНАМИ
    # ==================================================
    print("🔍 Расчет метрик с анализом паттернов...")

    # Основные метрики
    passenger_stats['days_active'] = (
        (pd.to_datetime(passenger_stats['last_flight']) - 
         pd.to_datetime(passenger_stats['first_flight'])).dt.days.clip(lower=1)
    )

    # Используем среднюю частоту за период вместо ежедневной
    passenger_stats['avg_activity_frequency'] = passenger_stats['avg_flights_per_period']

    # Метрики геолокации
    if 'departure_airports' in passenger_stats.columns:
        passenger_stats['n_unique_departures'] = passenger_stats['departure_airports'].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )

    if 'arrival_airports' in passenger_stats.columns:
        passenger_stats['n_unique_arrivals'] = passenger_stats['arrival_airports'].apply(
            lambda x: len(x) if isinstance(x, list) else 0
        )

    if 'n_unique_departures' in passenger_stats.columns and 'n_unique_arrivals' in passenger_stats.columns:
        passenger_stats['total_unique_airports'] = (
            passenger_stats['n_unique_departures'] + passenger_stats['n_unique_arrivals']
        )

    # ОПТИМИЗАЦИЯ: Быстрый расчет подозрительных документов
    print("🔍 Быстрый расчет подозрительных документов...")

    # Создаем словарь для быстрого поиска подозрительных документов
    suspicious_docs_set = set(suspicious_docs['document_norm'].values)

    # Группируем документы по пассажирам
    passenger_docs = valid_df.groupby(['first_name', 'last_name', 'pax_birth_data'])['document_norm'].apply(list).reset_index()

    # Быстрая функция для проверки подозрительных документов
    def get_suspicious_docs_fast(doc_list):
        suspicious = [doc for doc in doc_list if doc in suspicious_docs_set]
        return suspicious, len(suspicious)

    # Применяем оптимизированную функцию
    passenger_docs[['suspicious_documents', 'suspicious_docs_count']] = pd.DataFrame(
        passenger_docs['document_norm'].apply(get_suspicious_docs_fast).tolist(),
        index=passenger_docs.index
    )

    # Объединяем с основной статистикой
    passenger_stats = passenger_stats.merge(
        passenger_docs[['first_name', 'last_name', 'pax_birth_data', 'suspicious_documents', 'suspicious_docs_count']],
        on=['first_name', 'last_name', 'pax_birth_data'],
        how='left'
    )

    passenger_stats['has_suspicious_doc'] = passenger_stats['suspicious_docs_count'] > 0

    print(f"✅ Обработано {len(passenger_stats)} уникальных пассажиров")

    # ==================================================
    # 5. УЛУЧШЕННАЯ СИСТЕМА ОЦЕНКИ РИСКА С ПАТТЕРНАМИ АКТИВНОСТИ
    # ==================================================
    print("🔍 Расчет уровня риска с анализом паттернов...")

    def calculate_risk_score_with_reasons(row):
        score = 0
        reasons = []
        
        # Критические факторы
        if row['has_suspicious_doc']:
            score += 150
            score += row['suspicious_docs_count'] * 20
            reasons.append(f"Общие документы ({row['suspicious_docs_count']} шт)")
        
        # Паттерны активности
        if row['activity_cluster_score'] > 2.0:
            score += 60
            reasons.append("Высокая кластерная активность")
        elif row['activity_cluster_score'] > 1.0:
            score += 30
            reasons.append("Кластерная активность")
        
        # Внезапное увеличение активности (всплески)
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
        
        # Логистические несоответствия
        if row['logistic_inconsistency'] > 0.3:
            score += 80
            reasons.append("Высокая логистическая несогласованность")
        elif row['logistic_inconsistency'] > 0.1:
            score += 40
            reasons.append("Логистическая несогласованность")
        
        # Пиковая активность
        if row['peak_activity_period'] > 2.0:
            score += 50
            reasons.append("Очень высокая пиковая активность")
        elif row['peak_activity_period'] > 1.0:
            score += 25
            reasons.append("Высокая пиковая активность")
        
        # Агенты
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
        
        # Документы
        if row['n_unique_documents'] > 3:
            score += 50
            reasons.append("Много документов (4+)")
        elif row['n_unique_documents'] > 1:
            score += 25
            reasons.append("Несколько документов (2-3)")
        
        # Геолокация (разнообразие аэропортов)
        if 'total_unique_airports' in row and row['total_unique_airports'] > 10:
            score += 40
            reasons.append("Очень много аэропортов (10+)")
        elif 'total_unique_airports' in row and row['total_unique_airports'] > 5:
            score += 20
            reasons.append("Много аэропортов (6-10)")
        
        return int(score), "; ".join(reasons)

    # Применяем оптимизированную функцию
    risk_results = passenger_stats.apply(calculate_risk_score_with_reasons, axis=1)
    passenger_stats['risk_score'] = [x[0] for x in risk_results]
    passenger_stats['risk_reasons'] = [x[1] for x in risk_results]

    # Категории риска
    def get_risk_category(score):
        if score >= 200:
            return "🚨 КРИТИЧЕСКИЙ"
        elif score >= 100:
            return "🔴 ВЫСОКИЙ"
        elif score >= 50:
            return "🟡 СРЕДНИЙ"
        elif score >= 20:
            return "🔵 НИЗКИЙ"
        else:
            return "✅ НОРМА"

    passenger_stats['risk_category'] = passenger_stats['risk_score'].apply(get_risk_category)
    passenger_stats['is_suspicious'] = passenger_stats['risk_score'] >= 50

    print(f"📊 Статистика рисков:")
    print(f"   - Подозрительных пассажиров: {passenger_stats['is_suspicious'].sum()}")
    print(f"   - С общими документами: {passenger_stats['has_suspicious_doc'].sum()}")
    print(f"   - С паттернами кластерной активности: {(passenger_stats['activity_cluster_score'] > 1).sum()}")
    print(f"   - С резкими всплесками активности: {(passenger_stats['sudden_activity_increase'] > 2).sum()}")

    # ==================================================
    # 6. СОХРАНЕНИЕ ДАННЫХ ДЛЯ ГРАФИКОВ В EXCEL
    # ==================================================
    print("\n📊 Сохранение данных для графиков...")

    # 1. Данные для распределения по категориям риска
    risk_distribution = passenger_stats['risk_category'].value_counts().reset_index()
    risk_distribution.columns = ['risk_category', 'count']
    risk_distribution['percentage'] = (risk_distribution['count'] / len(passenger_stats) * 100).round(1)

    # 2. Данные для паттернов активности
    activity_patterns_data = passenger_stats[['activity_cluster_score', 'sudden_activity_increase', 
                                            'logistic_inconsistency', 'peak_activity_period', 'risk_category']].copy()

    # 3. Данные для распределения агентов
    agent_distribution = passenger_stats['n_unique_agents'].value_counts().sort_index().reset_index()
    agent_distribution.columns = ['n_agents', 'count']

    # 4. Данные для причин аномалий
    risk_reasons_analysis = passenger_stats[passenger_stats['risk_score'] > 0]['risk_reasons'].str.split('; ').explode().value_counts().reset_index()
    risk_reasons_analysis.columns = ['reason', 'count']

    # 5. Данные для документов
    docs_analysis = passenger_stats.groupby('suspicious_docs_count').agg({
        'first_name': 'count',
        'risk_score': 'mean'
    }).reset_index()
    docs_analysis.columns = ['suspicious_docs_count', 'passenger_count', 'avg_risk_score']

    # 6. Данные для анализа паттернов
    patterns_analysis = passenger_stats[['activity_cluster_score', 'sudden_activity_increase', 
                                       'logistic_inconsistency', 'risk_score']].copy()

    # Сохраняем все данные в один Excel файл с разными листами
    with pd.ExcelWriter('/home/mariia/Загрузки/Telegram Desktop/AI2/activity_patterns_analysis.xlsx') as writer:
        risk_distribution.to_excel(writer, sheet_name='Распределение_рисков', index=False)
        activity_patterns_data.to_excel(writer, sheet_name='Паттерны_активности', index=False)
        agent_distribution.to_excel(writer, sheet_name='Распределение_агентов', index=False)
        risk_reasons_analysis.to_excel(writer, sheet_name='Причины_аномалий', index=False)
        docs_analysis.to_excel(writer, sheet_name='Анализ_документов', index=False)
        patterns_analysis.to_excel(writer, sheet_name='Детали_паттернов', index=False)

    print("✅ Данные для графиков сохранены в activity_patterns_analysis.xlsx")

    # ==================================================
    # 7. ВИЗУАЛИЗАЦИЯ ПАТТЕРНОВ АКТИВНОСТИ
    # ==================================================
    print("\n📊 Создание визуализаций паттернов активности...")

    # Создаем фигуры для графиков
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.patch.set_alpha(0.0)  # Прозрачный фон всей фигуры

    # 1. Распределение по категориям риска
    risk_counts = passenger_stats['risk_category'].value_counts()
    axes[0,0].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', 
                  colors=[DARK_BLUE, LIGHT_BLUE, '#6ba3d6', '#8fbce8', '#b4d4f0'])
    axes[0,0].set_title('📊 РАСПРЕДЕЛЕНИЕ ПО КАТЕГОРИЯМ РИСКА', fontweight='bold', color=DARK_BLUE)

    # 2. Распределение кластерной активности
    suspicious_only = passenger_stats[passenger_stats['is_suspicious'] == True]
    if len(suspicious_only) > 0:
        axes[0,1].scatter(suspicious_only['activity_cluster_score'], 
                         suspicious_only['sudden_activity_increase'],
                         c=suspicious_only['risk_score'], cmap='Blues', alpha=0.7, s=50)
        axes[0,1].set_title('🔍 КЛАСТЕРНАЯ АКТИВНОСТЬ vs ВСПЛЕСКИ АКТИВНОСТИ', fontweight='bold', color=DARK_BLUE)
        axes[0,1].set_xlabel('Оценка кластерной активности')
        axes[0,1].set_ylabel('Коэффициент всплеска активности')
        axes[0,1].set_facecolor('none')

    # 3. Логистическая несогласованность
    if len(suspicious_only) > 0:
        sns.histplot(data=suspicious_only, x='logistic_inconsistency', bins=20, ax=axes[0,2], color=DARK_BLUE)
        axes[0,2].set_title('🔄 ЛОГИСТИЧЕСКАЯ НЕСОГЛАСОВАННОСТЬ', fontweight='bold', color=DARK_BLUE)
        axes[0,2].set_xlabel('Уровень логистической несогласованности')
        axes[0,2].set_facecolor('none')

    # 4. Связь агентов и риска
    sns.boxplot(data=passenger_stats, x='risk_category', y='n_unique_agents', ax=axes[1,0], 
                palette=[DARK_BLUE, LIGHT_BLUE, '#6ba3d6', '#8fbce8'])
    axes[1,0].set_title('🏢 СВЯЗЬ АГЕНТОВ И КАТЕГОРИИ РИСКА', fontweight='bold', color=DARK_BLUE)
    axes[1,0].set_facecolor('none')

    # 5. Топ причин аномалий
    if len(risk_reasons_analysis) > 0:
        top_reasons = risk_reasons_analysis.head(8)
        sns.barplot(y=top_reasons['reason'], x=top_reasons['count'], ax=axes[1,1], color=DARK_BLUE)
        axes[1,1].set_title('📋 ТОП-8 ПРИЧИН АНОМАЛИЙ', fontweight='bold', color=DARK_BLUE)
        axes[1,1].set_xlabel('Количество случаев')
        axes[1,1].set_facecolor('none')

    # 6. Распределение пиковой активности
    if 'peak_activity_period' in passenger_stats.columns:
        sns.histplot(data=passenger_stats[passenger_stats['peak_activity_period'] < 5], 
                     x='peak_activity_period', hue='risk_category', ax=axes[1,2], 
                     palette=[DARK_BLUE, LIGHT_BLUE, '#6ba3d6', '#8fbce8'])
        axes[1,2].set_title('📈 РАСПРЕДЕЛЕНИЕ ПИКОВОЙ АКТИВНОСТИ', fontweight='bold', color=DARK_BLUE)
        axes[1,2].set_xlabel('Пиковая активность (рейсов/день)')
        axes[1,2].set_facecolor('none')

    # Устанавливаем прозрачный фон для всех осей
    for ax in axes.flat:
        ax.set_facecolor('none')

    plt.tight_layout()
    plt.savefig('/home/mariia/Загрузки/Telegram Desktop/AI2/activity_patterns_analysis.png', 
                dpi=300, bbox_inches='tight', transparent=True)
    plt.close()

    print("✅ Графики паттернов активности сохранены")

    # ==================================================
    # 8. ФОРМИРОВАНИЕ ФИНАЛЬНОЙ ТАБЛИЦЫ
    # ==================================================
    print("\n📋 Формирование финальной таблицы с детальной информацией...")

    # Создаем копию для финальной таблицы
    final_table = passenger_stats.copy()

    # Добавляем детальную информацию о подозрительной активности
    def format_suspicious_activity(row):
        details = []
        
        if row['has_suspicious_doc']:
            details.append(f"Общие документы: {row['suspicious_docs_count']} шт")
        
        if row['activity_cluster_score'] > 1.0:
            details.append(f"Кластерная активность: {row['activity_cluster_score']:.2f}")
        
        if row['sudden_activity_increase'] > 2.0:
            details.append(f"Всплеск активности: {row['sudden_activity_increase']:.1f}x")
        
        if row['logistic_inconsistency'] > 0.1:
            details.append(f"Логистические проблемы: {row['logistic_inconsistency']:.2f}")
        
        if row['n_unique_agents'] >= 5:
            details.append(f"Много агентов: {row['n_unique_agents']} шт")
        
        if row['n_unique_documents'] > 1:
            details.append(f"Несколько документов: {row['n_unique_documents']} шт")
        
        if 'total_unique_airports' in row and row['total_unique_airports'] > 5:
            details.append(f"Много аэропортов: {row['total_unique_airports']} шт")
        
        return "; ".join(details)

    final_table['suspicious_activity_details'] = final_table.apply(format_suspicious_activity, axis=1)

    # Форматируем списки аэропортов для читаемости
    if 'departure_airports' in final_table.columns:
        final_table['departure_airports_str'] = final_table['departure_airports'].apply(
            lambda x: ', '.join(str(airport) for airport in x[:5]) + ('...' if len(x) > 5 else '') if isinstance(x, list) else ''
        )

    if 'arrival_airports' in final_table.columns:
        final_table['arrival_airports_str'] = final_table['arrival_airports'].apply(
            lambda x: ', '.join(str(airport) for airport in x[:5]) + ('...' if len(x) > 5 else '') if isinstance(x, list) else ''
        )

    # Форматируем списки документов
    final_table['suspicious_documents_str'] = final_table['suspicious_documents'].apply(
        lambda x: ', '.join(str(doc) for doc in x[:3]) + ('...' if len(x) > 3 else '') if isinstance(x, list) else ''
    )

    # Выбираем колонки для финальной таблицы
    output_columns = [
        'first_name', 'last_name', 'pax_birth_data', 
        'n_flights_total', 'n_unique_agents', 'n_unique_documents',
        'days_active', 'avg_activity_frequency', 'risk_score', 'risk_category',
        'risk_reasons', 'suspicious_activity_details',
        'activity_cluster_score', 'sudden_activity_increase', 'logistic_inconsistency'
    ]

    # Добавляем геолокационные колонки если есть
    if 'n_unique_departures' in final_table.columns:
        output_columns.extend(['n_unique_departures', 'n_unique_arrivals'])
    if 'total_unique_airports' in final_table.columns:
        output_columns.append('total_unique_airports')
    if 'departure_airports_str' in final_table.columns:
        output_columns.append('departure_airports_str')
    if 'arrival_airports_str' in final_table.columns:
        output_columns.append('arrival_airports_str')

    output_columns.extend([
        'suspicious_documents_str', 'suspicious_docs_count'
    ])

    # Создаем финальную таблицу
    final_output = final_table[output_columns].copy()

    # ==================================================
    # 9. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
    # ==================================================
    print("\n💾 Сохранение результатов...")

    # Основные результаты
    output_path = r"/home/mariia/Загрузки/Telegram Desktop/AI2/final_results_activity_patterns.csv"
    final_output.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✅ Детальные результаты: {output_path}")

    # Подозрительные пассажиры
    suspicious_passengers = final_output[final_output['risk_score'] >= 50].sort_values('risk_score', ascending=False)
    suspicious_path = r"/home/mariia/Загрузки/Telegram Desktop/AI2/suspicious_passengers_activity_patterns.csv"
    suspicious_passengers.to_csv(suspicious_path, index=False, encoding='utf-8')
    print(f"✅ Подозрительные пассажиры: {suspicious_path}")

    # ==================================================
    # 10. ВЫВОД ИТОГОВЫХ СТАТИСТИК
    # ==================================================
    print(f"\n🎉 АНАЛИЗ ПАТТЕРНОВ АКТИВНОСТИ ЗАВЕРШЕН!")
    print(f"📊 ИТОГИ:")
    print(f"   - Всего пассажиров: {len(passenger_stats)}")
    print(f"   - Подозрительных: {len(suspicious_passengers)}")
    print(f"   - С общими документами: {passenger_stats['has_suspicious_doc'].sum()}")
    print(f"   - С кластерной активностью: {(passenger_stats['activity_cluster_score'] > 1).sum()}")
    print(f"   - С резкими всплесками активности: {(passenger_stats['sudden_activity_increase'] > 2).sum()}")
    print(f"   - С логистическими проблемами: {(passenger_stats['logistic_inconsistency'] > 0.1).sum()}")

    # Вывод топ подозрительных пассажиров
    print(f"\n🚨 ТОП-5 САМЫХ ПОДОЗРИТЕЛЬНЫХ ПАССАЖИРОВ:")
    print("=" * 120)

    for i, (_, row) in enumerate(suspicious_passengers.head(5).iterrows(), 1):
        print(f"{i}. {row['first_name']} {row['last_name']} ({row['pax_birth_data']})")
        print(f"   ⚡ Рейсов: {row['n_flights_total']} | 🏢 Агентов: {row['n_unique_agents']} | 📄 Документов: {row['n_unique_documents']}")
        print(f"   📅 Активность: {row['days_active']} дней | 📈 Частота: {row['avg_activity_frequency']:.2f} рейсов/период")
        print(f"   🎯 Risk: {row['risk_score']} ({row['risk_category']})")
        print(f"   📊 Паттерны: Кластеры={row['activity_cluster_score']:.2f}, Всплески={row['sudden_activity_increase']:.1f}x, Логистика={row['logistic_inconsistency']:.2f}")
        print(f"   📋 Детали: {row['suspicious_activity_details']}")
        
        if 'departure_airports_str' in row and row['departure_airports_str']:
            print(f"   🛫 Аэропорты вылета: {row['departure_airports_str']}")
        
        if 'arrival_airports_str' in row and row['arrival_airports_str']:
            print(f"   🛬 Аэропорты прилета: {row['arrival_airports_str']}")
        
        print("-" * 120)

    return valid_df, passenger_stats

# ==================================================
# ФУНКЦИИ ДЛЯ ДЕТАЛЬНОГО АНАЛИЗА ПАССАЖИРОВ
# ==================================================

def analyze_specific_passenger(first_name, last_name, birth_date, valid_df, passenger_stats):
    """
    Детальный анализ конкретного пассажира с графиками активности
    """
    print(f"🔍 ДЕТАЛЬНЫЙ АНАЛИЗ ПАССАЖИРА: {first_name} {last_name} ({birth_date})")
    print("=" * 100)
    
    # Находим пассажира в статистике
    passenger_mask = (
        (passenger_stats['first_name'] == first_name) & 
        (passenger_stats['last_name'] == last_name) & 
        (passenger_stats['pax_birth_data'] == birth_date)
    )
    
    if not passenger_mask.any():
        print(f"❌ Пассажир не найден в статистике")
        return None
    
    passenger_data = passenger_stats[passenger_mask].iloc[0]
    
    # Находим все перелеты пассажира
    flight_mask = (
        (valid_df['first_name'] == first_name) & 
        (valid_df['last_name'] == last_name) & 
        (valid_df['pax_birth_data'] == birth_date)
    )
    
    passenger_flights = valid_df[flight_mask].copy()
    passenger_flights['flight_date'] = pd.to_datetime(passenger_flights['flight_date'])
    passenger_flights = passenger_flights.sort_values('flight_date')
    
    print(f"📊 ОСНОВНАЯ ИНФОРМАЦИЯ:")
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
    
    print(f"\n📋 ПРИЧИНЫ РИСКА:")
    print(f"   {passenger_data['risk_reasons']}")
    
    # Подозрительные документы
    if passenger_data['has_suspicious_doc']:
        print(f"\n🚨 ПОДОЗРИТЕЛЬНЫЕ ДОКУМЕНТЫ:")
        for i, doc in enumerate(passenger_data['suspicious_documents'][:5], 1):
            print(f"   {i}. {doc}")
        if len(passenger_data['suspicious_documents']) > 5:
            print(f"   ... и еще {len(passenger_data['suspicious_documents']) - 5} документов")
    
    # Агенты
    print(f"\n🏢 ИНФОРМАЦИЯ ОБ АГЕНТАХ:")
    agents = passenger_flights['agent_info'].value_counts()
    for agent, count in agents.head(10).items():
        print(f"   • {agent}: {count} рейсов")
    
    # Аэропорты
    if 'departure' in passenger_flights.columns:
        print(f"\n🛫 ТОП АЭРОПОРТОВ ВЫЛЕТА:")
        departures = passenger_flights['departure'].value_counts().head(5)
        for airport, count in departures.items():
            print(f"   • {airport}: {count} вылетов")
    
    if 'arrival' in passenger_flights.columns:
        print(f"\n🛬 ТОП АЭРОПОРТОВ ПРИЛЕТА:")
        arrivals = passenger_flights['arrival'].value_counts().head(5)
        for airport, count in arrivals.items():
            print(f"   • {airport}: {count} прилетов")
    
    # ==================================================
    # ПОСТРОЕНИЕ ГРАФИКОВ
    # ==================================================
    print(f"\n📈 ПОСТРОЕНИЕ ГРАФИКОВ АКТИВНОСТИ...")
    
    # Создаем фигуру с несколькими графиками
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'ДЕТАЛЬНЫЙ АНАЛИЗ: {first_name} {last_name}\n'
                f'Уровень риска: {passenger_data["risk_score"]} ({passenger_data["risk_category"]})', 
                fontsize=16, fontweight='bold', color=DARK_BLUE)
    
    # 1. График активности по времени
    if len(passenger_flights) > 0:
        # Группируем по дате
        daily_activity = passenger_flights.groupby(passenger_flights['flight_date'].dt.date).size()
        
        axes[0,0].plot(daily_activity.index, daily_activity.values, 
                      marker='o', linewidth=2, markersize=4, color=DARK_BLUE)
        axes[0,0].set_title('АКТИВНОСТЬ ПО ДНЯМ', fontweight='bold', color=DARK_BLUE)
        axes[0,0].set_xlabel('Дата')
        axes[0,0].set_ylabel('Количество рейсов')
        axes[0,0].tick_params(axis='x', rotation=45)
        axes[0,0].grid(True, alpha=0.3)
        
        # Добавляем линию тренда
        if len(daily_activity) > 1:
            x_numeric = np.arange(len(daily_activity))
            z = np.polyfit(x_numeric, daily_activity.values, 1)
            p = np.poly1d(z)
            axes[0,0].plot(daily_activity.index, p(x_numeric), "r--", alpha=0.8, 
                          label=f'Тренд (наклон: {z[0]:.2f})')
            axes[0,0].legend()
    
    # 2. Распределение по месяцам
    if len(passenger_flights) > 0:
        monthly_activity = passenger_flights.groupby(passenger_flights['flight_date'].dt.to_period('M')).size()
        monthly_activity.index = monthly_activity.index.astype(str)
        
        axes[0,1].bar(monthly_activity.index, monthly_activity.values, color=DARK_BLUE, alpha=0.7)
        axes[0,1].set_title('АКТИВНОСТЬ ПО МЕСЯЦАМ', fontweight='bold', color=DARK_BLUE)
        axes[0,1].set_xlabel('Месяц')
        axes[0,1].set_ylabel('Количество рейсов')
        axes[0,1].tick_params(axis='x', rotation=45)
    
    # 3. Распределение по агентам
    if len(agents) > 0:
        top_agents = agents.head(8)
        axes[0,2].barh(range(len(top_agents)), top_agents.values, color=LIGHT_BLUE)
        axes[0,2].set_yticks(range(len(top_agents)))
        axes[0,2].set_yticklabels([str(agent)[:30] + '...' if len(str(agent)) > 30 else str(agent) 
                                 for agent in top_agents.index])
        axes[0,2].set_title('РАСПРЕДЕЛЕНИЕ ПО АГЕНТАМ', fontweight='bold', color=DARK_BLUE)
        axes[0,2].set_xlabel('Количество рейсов')
    
    # 4. Распределение по дням недели
    if len(passenger_flights) > 0:
        weekday_activity = passenger_flights.groupby(passenger_flights['flight_date'].dt.day_name()).size()
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_activity = weekday_activity.reindex(weekday_order, fill_value=0)
        
        axes[1,0].bar(weekday_activity.index, weekday_activity.values, color=DARK_BLUE, alpha=0.7)
        axes[1,0].set_title('АКТИВНОСТЬ ПО ДНЯМ НЕДЕЛИ', fontweight='bold', color=DARK_BLUE)
        axes[1,0].set_xlabel('День недели')
        axes[1,0].set_ylabel('Количество рейсов')
        axes[1,0].tick_params(axis='x', rotation=45)
    
    # 5. Карта маршрутов (если есть данные)
    if 'departure' in passenger_flights.columns and 'arrival' in passenger_flights.columns:
        routes = passenger_flights.groupby(['departure', 'arrival']).size().reset_index()
        routes.columns = ['departure', 'arrival', 'count']
        
        if len(routes) > 0:
            # Создаем таблицу для матрицы маршрутов
            route_matrix = routes.pivot_table(index='departure', columns='arrival', 
                                            values='count', fill_value=0)
            
            if len(route_matrix) > 1:
                im = axes[1,1].imshow(route_matrix.values, cmap='Blues', aspect='auto')
                axes[1,1].set_title('🛫 МАТРИЦА МАРШРУТОВ', fontweight='bold', color=DARK_BLUE)
                axes[1,1].set_xlabel('Аэропорт прилета')
                axes[1,1].set_ylabel('Аэропорт вылета')
                
                # Устанавливаем подписи (ограничиваем количество для читаемости)
                if len(route_matrix) <= 10:
                    axes[1,1].set_xticks(range(len(route_matrix.columns)))
                    axes[1,1].set_xticklabels([str(col) for col in route_matrix.columns], rotation=45, ha='right')
                    axes[1,1].set_yticks(range(len(route_matrix.index)))
                    axes[1,1].set_yticklabels([str(idx) for idx in route_matrix.index])
                
                plt.colorbar(im, ax=axes[1,1], label='Количество рейсов')
            else:
                axes[1,1].text(0.5, 0.5, 'Недостаточно данных\nдля матрицы маршрутов', 
                              ha='center', va='center', transform=axes[1,1].transAxes)
                axes[1,1].set_title('МАТРИЦА МАРШРУТОВ', fontweight='bold', color=DARK_BLUE)
        else:
            axes[1,1].text(0.5, 0.5, 'Нет данных о маршрутах', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('МАТРИЦА МАРШРУТОВ', fontweight='bold', color=DARK_BLUE)
    else:
        axes[1,1].text(0.5, 0.5, 'Нет данных о геолокации', 
                      ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,1].set_title('МАТРИЦА МАРШРУТОВ', fontweight='bold', color=DARK_BLUE)
    
    # 6. Детализация рисков
    risk_factors = []
    if passenger_data['has_suspicious_doc']:
        risk_factors.append(f'Общие документы ({passenger_data["suspicious_docs_count"]})')
    if passenger_data.get('activity_cluster_score', 0) > 1:
        risk_factors.append(f'Кластерная активность ({passenger_data["activity_cluster_score"]:.2f})')
    if passenger_data.get('sudden_activity_increase', 0) > 2:
        risk_factors.append(f'Всплеск активности ({passenger_data["sudden_activity_increase"]:.1f}x)')
    if passenger_data['n_unique_agents'] >= 3:
        risk_factors.append(f'Много агентов ({passenger_data["n_unique_agents"]})')
    if passenger_data['n_unique_documents'] > 1:
        risk_factors.append(f'Несколько документов ({passenger_data["n_unique_documents"]})')
    
    if risk_factors:
        axes[1,2].barh(range(len(risk_factors)), [len(risk_factors)-i for i in range(len(risk_factors))], 
                      color=['#ff6b6b' if 'Общие документы' in factor else LIGHT_BLUE for factor in risk_factors])
        axes[1,2].set_yticks(range(len(risk_factors)))
        axes[1,2].set_yticklabels(risk_factors)
        axes[1,2].set_title('ФАКТОРЫ РИСКА', fontweight='bold', color=DARK_BLUE)
        axes[1,2].set_xlabel('Важность (условная)')
    else:
        axes[1,2].text(0.5, 0.5, 'Нет значимых факторов риска', 
                      ha='center', va='center', transform=axes[1,2].transAxes)
        axes[1,2].set_title('ФАКТОРЫ РИСКА', fontweight='bold', color=DARK_BLUE)
    
    plt.tight_layout()
    
    # Сохраняем график
    filename = f"/home/mariia/Загрузки/Telegram Desktop/AI2/passenger_{first_name}_{last_name}_analysis.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=False)
    plt.show()
    
    print(f"✅ Графики сохранены: {filename}")
    
    # ==================================================
    # ДЕТАЛЬНАЯ ТАБЛИЦА ПЕРЕЛЕТОВ
    # ==================================================
    print(f"\n📋 ДЕТАЛЬНАЯ ТАБЛИЦА ПЕРЕЛЕТОВ (первые 20):")
    print("=" * 120)
    
    # Выбираем колонки для отображения
    display_columns = ['flight_date', 'flight_code']
    if 'departure' in passenger_flights.columns:
        display_columns.append('departure')
    if 'arrival' in passenger_flights.columns:
        display_columns.append('arrival')
    if 'agent_info' in passenger_flights.columns:
        display_columns.append('agent_info')
    if 'document_norm' in passenger_flights.columns:
        display_columns.append('document_norm')
    
    display_flights = passenger_flights[display_columns].head(20).copy()
    display_flights['flight_date'] = display_flights['flight_date'].dt.strftime('%Y-%m-%d')
    
    # Форматируем для красивого вывода с обработкой ошибок
    for _, flight in display_flights.iterrows():
        print(f"📅 {flight['flight_date']} | ✈️ {flight.get('flight_code', 'N/A')} ", end="")
        if 'departure' in flight and 'arrival' in flight:
            print(f"| 🛫 {str(flight['departure'])} → 🛬 {str(flight['arrival'])} ", end="")
        if 'agent_info' in flight:
            agent_str = str(flight['agent_info'])
            if len(agent_str) > 20:
                agent_str = agent_str[:20] + "..."
            print(f"| 🏢 {agent_str} ", end="")
        if 'document_norm' in flight:
            doc_display = str(flight['document_norm'])
            if len(doc_display) > 15:
                doc_display = doc_display[:15] + "..."
            print(f"| 📄 {doc_display}", end="")
        print()
    
    if len(passenger_flights) > 20:
        print(f"... и еще {len(passenger_flights) - 20} перелетов")
    
    # ==================================================
    # СОХРАНЕНИЕ ДАННЫХ В ФАЙЛ
    # ==================================================
    print(f"\n💾 СОХРАНЕНИЕ ДАННЫХ В ФАЙЛ...")
    
    # Сохраняем детальную информацию в CSV
    detail_filename = f"/home/mariia/Загрузки/Telegram Desktop/AI2/passenger_{first_name}_{last_name}_details.csv"
    
    # Создаем сводную таблицу с основной информацией
    summary_data = {
        'Параметр': [
            'Имя', 'Фамилия', 'Дата рождения', 'Всего перелетов', 
            'Уникальных агентов', 'Уникальных документов', 'Дней активности',
            'Уровень риска', 'Категория риска', 'Причины риска'
        ],
        'Значение': [
            passenger_data['first_name'],
            passenger_data['last_name'], 
            passenger_data['pax_birth_data'],
            passenger_data['n_flights_total'],
            passenger_data['n_unique_agents'],
            passenger_data['n_unique_documents'],
            passenger_data['days_active'],
            passenger_data['risk_score'],
            passenger_data['risk_category'],
            passenger_data['risk_reasons']
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(detail_filename, index=False, encoding='utf-8')
    
    # Сохраняем детальные перелеты
    flights_filename = f"/home/mariia/Загрузки/Telegram Desktop/AI2/passenger_{first_name}_{last_name}_flights.csv"
    
    # Обеспечиваем, что все данные являются строками
    flights_export = passenger_flights[display_columns].copy()
    for col in flights_export.columns:
        flights_export[col] = flights_export[col].astype(str)
    
    flights_export.to_csv(flights_filename, index=False, encoding='utf-8')
    
    print(f"✅ Сводная информация: {detail_filename}")
    print(f"✅ Детальные перелеты: {flights_filename}")
    
    return {
        'summary': passenger_data,
        'flights': passenger_flights,
        'graph_filename': filename,
        'detail_filename': detail_filename,
        'flights_filename': flights_filename
    }

def search_passengers(search_term, passenger_stats, max_results=10):
    """
    Поиск пассажиров по имени, фамилии или дате рождения
    """
    search_term = search_term.lower()
    
    # Поиск по разным полям
    mask = (
        passenger_stats['first_name'].str.lower().str.contains(search_term, na=False) |
        passenger_stats['last_name'].str.lower().str.contains(search_term, na=False) |
        passenger_stats['pax_birth_data'].str.lower().str.contains(search_term, na=False)
    )
    
    results = passenger_stats[mask].head(max_results)
    
    if len(results) == 0:
        print(f"❌ Пассажиры по запросу '{search_term}' не найдены")
        return None
    
    print(f"🔍 НАЙДЕНО ПАССАЖИРОВ: {len(results)}")
    print("=" * 80)
    
    for i, (_, passenger) in enumerate(results.iterrows(), 1):
        print(f"{i}. {passenger['first_name']} {passenger['last_name']} ({passenger['pax_birth_data']})")
        print(f"   ⚡ Рейсов: {passenger['n_flights_total']} | 🏢 Агентов: {passenger['n_unique_agents']} | 🎯 Risk: {passenger['risk_score']} ({passenger['risk_category']})")
        if i < len(results):  # Не печатать разделитель после последнего
            print("-" * 80)
    
    return results

def run_passenger_analysis(valid_df, passenger_stats):
    """
    Основная функция для запуска анализа пассажиров
    """
    print("🎯 СИСТЕМА ДЕТАЛЬНОГО АНАЛИЗА ПАССАЖИРОВ")
    print("=" * 60)
    
    while True:
        print("\nВыберите действие:")
        print("1. 🔍 Поиск пассажира")
        print("2. 📊 Анализ подозрительных пассажиров")
        print("3. 🚪 Выход")
        
        choice = input("\nВведите номер действия (1-3): ").strip()
        
        if choice == '1':
            search_term = input("Введите имя, фамилию или дату рождения для поиска: ").strip()
            if search_term:
                results = search_passengers(search_term, passenger_stats)
                if results is not None and len(results) > 0:
                    if len(results) == 1:
                        # Если найден один пассажир - сразу анализируем
                        passenger = results.iloc[0]
                        analyze_specific_passenger(
                            passenger['first_name'], 
                            passenger['last_name'], 
                            passenger['pax_birth_data'],
                            valid_df, 
                            passenger_stats
                        )
                    else:
                        # Если несколько - предлагаем выбрать
                        try:
                            passenger_num = int(input(f"\nВыберите пассажира (1-{len(results)}): ")) - 1
                            if 0 <= passenger_num < len(results):
                                passenger = results.iloc[passenger_num]
                                analyze_specific_passenger(
                                    passenger['first_name'], 
                                    passenger['last_name'], 
                                    passenger['pax_birth_data'],
                                    valid_df, 
                                    passenger_stats
                                )
                            else:
                                print("❌ Неверный номер пассажира")
                        except ValueError:
                            print("❌ Введите корректный номер")
        
        elif choice == '2':
            # Показываем топ подозрительных пассажиров
            suspicious = passenger_stats[passenger_stats['is_suspicious'] == True].sort_values('risk_score', ascending=False)
            
            print(f"\n🚨 ТОП-10 ПОДОЗРИТЕЛЬНЫХ ПАССАЖИРОВ:")
            print("=" * 80)
            
            for i, (_, passenger) in enumerate(suspicious.head(10).iterrows(), 1):
                print(f"{i}. {passenger['first_name']} {passenger['last_name']} ({passenger['pax_birth_data']})")
                print(f"   ⚡ Рейсов: {passenger['n_flights_total']} | 🏢 Агентов: {passenger['n_unique_agents']} | 🎯 Risk: {passenger['risk_score']}")
                print(f"   📋 {passenger['risk_reasons']}")
                if i < min(10, len(suspicious)):
                    print("-" * 80)
            
            try:
                passenger_num = int(input(f"\nВыберите пассажира для детального анализа (1-10): ")) - 1
                if 0 <= passenger_num < len(suspicious.head(10)):
                    passenger = suspicious.iloc[passenger_num]
                    analyze_specific_passenger(
                        passenger['first_name'], 
                        passenger['last_name'], 
                        passenger['pax_birth_data'],
                        valid_df, 
                        passenger_stats
                    )
                else:
                    print("❌ Неверный номер пассажира")
            except ValueError:
                print("❌ Введите корректный номер")
        
        elif choice == '3':
            print("👋 Завершение работы...")
            break
        
        else:
            print("❌ Неверный выбор. Попробуйте снова.")

# ==================================================
# ГЛАВНОЕ МЕНЮ
# ==================================================

def main():
    """Главное меню программы"""
    print("🛫 СИСТЕМА АНАЛИЗА ПАССАЖИРСКОЙ АКТИВНОСТИ")
    print("=" * 50)
    
    valid_df = None
    passenger_stats = None
    
    while True:
        print("\n📋 ГЛАВНОЕ МЕНЮ:")
        print("1. 🚀 Запустить полный анализ данных")
        print("2. 🔍 Детальный анализ конкретного пассажира")
        print("3. 💾 Загрузить ранее сохраненные данные")
        print("4. 🚪 Выход")
        
        choice = input("\nВыберите действие (1-4): ").strip()
        
        if choice == '1':
            print("\n" + "="*50)
            print("🚀 ЗАПУСК ПОЛНОГО АНАЛИЗА ДАННЫХ")
            print("="*50)
            valid_df, passenger_stats = main_analysis()
            
        elif choice == '2':
            if valid_df is None or passenger_stats is None:
                print("❌ Сначала выполните полный анализ данных (пункт 1)")
                continue
            print("\n" + "="*50)
            print("🔍 ЗАПУСК ДЕТАЛЬНОГО АНАЛИЗА ПАССАЖИРОВ")
            print("="*50)
            run_passenger_analysis(valid_df, passenger_stats)
            
        elif choice == '3':
            print("\n💾 ЗАГРУЗКА СОХРАНЕННЫХ ДАННЫХ...")
            try:
                # Попытка загрузить ранее сохраненные данные
                valid_df = pd.read_csv(r"/home/mariia/Загрузки/Telegram Desktop/AI2/final_results_activity_patterns.csv")
                passenger_stats = pd.read_csv(r"/home/mariia/Загрузки/Telegram Desktop/AI2/suspicious_passengers_activity_patterns.csv")
                print("✅ Данные успешно загружены!")
            except FileNotFoundError:
                print("❌ Файлы с данными не найдены. Сначала выполните полный анализ.")
                
        elif choice == '4':
            print("👋 Завершение работы программы. До свидания!")
            break
            
        else:
            print("❌ Неверный выбор. Пожалуйста, выберите от 1 до 4.")

# ==================================================
# ЗАПУСК ПРОГРАММЫ
# ==================================================

if __name__ == "__main__":
    main()
