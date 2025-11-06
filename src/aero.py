import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import warnings
warnings.filterwarnings('ignore')

# Настройка стиля
plt.style.use('default')
sns.set_palette("dark:blue")

class OptimizedFlightMapVisualizer:
    """Оптимизированный класс для визуализации перелетов"""
    
    def __init__(self):
        self.geolocator = Nominatim(user_agent="flight_analysis_optimized")
        self.airport_coords = {}
        # Загружаем заранее известные координаты популярных аэропортов
        self.load_common_airports()
    
    def load_common_airports(self):
        """Предзагружаем координаты популярных аэропортов"""
        common_airports = {
            'Moscow': (55.7558, 37.6173),
            'London': (51.5074, -0.1278),
            'Paris': (48.8566, 2.3522),
            'New York': (40.7128, -74.0060),
            'Tokyo': (35.6762, 139.6503),
            'Dubai': (25.2048, 55.2708),
            'Istanbul': (41.0082, 28.9784),
            'Frankfurt': (50.1109, 8.6821),
            'Amsterdam': (52.3676, 4.9041),
            'Madrid': (40.4168, -3.7038),
            'Rome': (41.9028, 12.4964),
            'Barcelona': (41.3851, 2.1734),
            'Berlin': (52.5200, 13.4050),
            'Prague': (50.0755, 14.4378),
            'Vienna': (48.2082, 16.3738),
            'Warsaw': (52.2297, 21.0122),
            'Budapest': (47.4979, 19.0402),
            'Athens': (37.9838, 23.7275),
            'Lisbon': (38.7223, -9.1393),
            'Zurich': (47.3769, 8.5417),
        }
        self.airport_coords.update(common_airports)
    
    def fast_geocode_airport(self, airport_name):
        """Быстрое геокодирование с использованием кэша и эвристик"""
        if airport_name in self.airport_coords:
            return self.airport_coords[airport_name]
        
        # Пробуем найти в кэше по частичному совпадению
        for known_airport, coords in self.airport_coords.items():
            if known_airport.lower() in airport_name.lower() or airport_name.lower() in known_airport.lower():
                self.airport_coords[airport_name] = coords
                return coords
        
        # Если не нашли в кэше, пробуем геокодировать
        try:
            location = self.geolocator.geocode(airport_name + " airport", timeout=10)
            if location:
                coords = (location.latitude, location.longitude)
                self.airport_coords[airport_name] = coords
                return coords
        except (GeocoderTimedOut, GeocoderServiceError):
            pass
        
        return None
    
    def prepare_optimized_data(self, df, sample_size=None):
        """Оптимизированная подготовка данных"""
        print("🔍 Оптимизированная подготовка данных...")
        
        # Если указан размер выборки, берем случайную выборку
        if sample_size and len(df) > sample_size:
            df = df.sample(sample_size)
            print(f"📊 Взята выборка: {sample_size} перелетов")
        
        flight_data = df.copy()
        
        # Быстрое преобразование текстовых колонок
        text_columns = ['departure', 'arrival']
        for col in text_columns:
            if col in flight_data.columns:
                flight_data[col] = flight_data[col].astype(str).replace('nan', '')
        
        # Фильтруем только корректные данные
        if 'departure' in flight_data.columns and 'arrival' in flight_data.columns:
            mask = (flight_data['departure'] != '') & (flight_data['arrival'] != '')
            flight_data = flight_data[mask]
        
        print(f"📊 Обработано перелетов: {len(flight_data)}")
        return flight_data
    
    def create_optimized_coordinates(self, flight_data, max_airports=200):
        """Оптимизированное создание координат аэропортов"""
        print("🗺️ Оптимизированное создание координат аэропортов...")
        
        # Собираем уникальные аэропорты
        all_airports = set()
        if 'departure' in flight_data.columns:
            all_airports.update(flight_data['departure'].unique())
        if 'arrival' in flight_data.columns:
            all_airports.update(flight_data['arrival'].unique())
        
        print(f"🔍 Найдено уникальных аэропортов: {len(all_airports)}")
        
        # Ограничиваем количество аэропортов для геокодирования
        if len(all_airports) > max_airports:
            print(f"⚠️  Ограничиваем геокодирование до {max_airports} самых частых аэропортов")
            
            # Считаем частоту аэропортов
            airport_counts = {}
            for airport in all_airports:
                dep_count = len(flight_data[flight_data['departure'] == airport])
                arr_count = len(flight_data[flight_data['arrival'] == airport])
                airport_counts[airport] = dep_count + arr_count
            
            # Берем топ-N самых частых аэропортов
            top_airports = sorted(airport_counts.items(), key=lambda x: x[1], reverse=True)[:max_airports]
            all_airports = set([airport for airport, count in top_airports])
        
        print("📍 Быстрое получение координат аэропортов...")
        
        airport_data = []
        for i, airport in enumerate(all_airports):
            if i % 10 == 0:
                print(f"📍 Обработано {i}/{len(all_airports)} аэропортов...")
                
            coords = self.fast_geocode_airport(airport)
            if coords:
                airport_data.append({
                    'airport': airport,
                    'lat': coords[0],
                    'lon': coords[1],
                    'flights_count': len(flight_data[flight_data['departure'] == airport]) + 
                                   len(flight_data[flight_data['arrival'] == airport])
                })
        
        return pd.DataFrame(airport_data)
    
    def create_fast_world_map(self, flight_data, airport_df):
        """Быстрое создание карты мира"""
        print("🌍 Быстрое создание карты мира...")
        
        # Создаем карту
        fig = go.Figure()
        
        # Добавляем аэропорты
        fig.add_trace(go.Scattergeo(
            lon=airport_df['lon'],
            lat=airport_df['lat'],
            text=airport_df['airport'] + '<br>Рейсов: ' + airport_df['flights_count'].astype(str),
            mode='markers',
            marker=dict(
                size=10,
                color='red',
                opacity=0.8,
                sizemode='area'
            ),
            name='Аэропорты'
        ))
        
        # Быстро добавляем перелеты - ограничиваем количество для производительности
        print("✈️  Быстрое добавление перелетов...")
        
        # Группируем маршруты для уменьшения количества линий
        if len(flight_data) > 5000:
            print("🔄 Группируем маршруты для оптимизации...")
            route_counts = flight_data.groupby(['departure', 'arrival']).size().reset_index()
            route_counts.columns = ['departure', 'arrival', 'count']
            # Берем топ маршрутов
            top_routes = route_counts.nlargest(2000, 'count')
            flight_data_to_use = top_routes
        else:
            flight_data_to_use = flight_data
        
        # Добавляем линии перелетов
        flight_lines = []
        for i, (_, flight) in enumerate(flight_data_to_use.iterrows()):
            if i % 1000 == 0:
                print(f"✈️  Обработано {i}/{len(flight_data_to_use)} перелетов...")
                
            dep_airport = flight['departure']
            arr_airport = flight['arrival']
            
            dep_coords = self.airport_coords.get(dep_airport)
            arr_coords = self.airport_coords.get(arr_airport)
            
            if dep_coords and arr_coords:
                flight_lines.append({
                    'dep_lon': dep_coords[1],
                    'dep_lat': dep_coords[0],
                    'arr_lon': arr_coords[1],
                    'arr_lat': arr_coords[0],
                    'route': f"{dep_airport} → {arr_airport}"
                })
        
        print(f"✅ Создано {len(flight_lines)} линий перелетов")
        
        # Используем более эффективный способ добавления линий
        if flight_lines:
            # Создаем массивы координат для всех линий
            lons = []
            lats = []
            
            for line in flight_lines:
                lons.extend([line['dep_lon'], line['arr_lon'], None])
                lats.extend([line['dep_lat'], line['arr_lat'], None])
            
            # Добавляем все линии одним trace
            fig.add_trace(go.Scattergeo(
                lon=lons,
                lat=lats,
                mode='lines',
                line=dict(width=1, color='blue'),
                opacity=0.2,
                showlegend=False
            ))
        
        # Настройка карты
        fig.update_layout(
            title_text='🌍 ОПТИМИЗИРОВАННАЯ КАРТА ПЕРЕЛЕТОВ<br><sub>Самые популярные маршруты</sub>',
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
        output_path = "/home/mariia/Загрузки/Telegram Desktop/AI2/world_flights_OPTIMIZED.html"
        fig.write_html(output_path)
        print(f"✅ Оптимизированная карта сохранена: {output_path}")
        
        return fig

def main():
    """Главная функция программы"""
    print("🌍 ОПТИМИЗИРОВАННАЯ ПРОГРАММА ВИЗУАЛИЗАЦИИ ПЕРЕЛЕТОВ")
    print("=" * 60)
    
    # Загрузка данных
    try:
        df = pd.read_csv(r"/home/mariia/Загрузки/Telegram Desktop/AI2/data_staging/merged_all_detailed.csv", low_memory=False)
        print(f"✅ Загружено {len(df):,} строк данных")
    except FileNotFoundError:
        print("❌ Файл данных не найден")
        return
    
    # Создаем оптимизированный визуализатор
    visualizer = OptimizedFlightMapVisualizer()
    
    # Запрашиваем у пользователя настройки
    print("\n⚡ НАСТРОЙКИ ОПТИМИЗАЦИИ:")
    print("1. 🔥 Максимальная скорость (1000 перелетов)")
    print("2. ⚡ Баланс скорости и качества (5000 перелетов)") 
    print("3. 🎯 Хорошее качество (10000 перелетов)")
    print("4. 🚀 Все данные (может быть медленно)")
    
    choice = input("\nВыберите вариант (1-4): ").strip()
    
    sample_sizes = {
        '1': 1000,
        '2': 5000, 
        '3': 10000,
        '4': None
    }
    
    sample_size = sample_sizes.get(choice, 5000)
    
    if sample_size:
        print(f"📊 Будет использована выборка: {sample_size} перелетов")
    else:
        print("📊 Будут использованы все данные (может занять много времени)")
    
    # Подготавливаем данные
    flight_data = visualizer.prepare_optimized_data(df, sample_size=sample_size)
    
    if len(flight_data) == 0:
        print("❌ Нет данных о перелетах для визуализации")
        return
    
    # Создаем координаты аэропортов
    airport_df = visualizer.create_optimized_coordinates(flight_data, max_airports=150)
    
    if len(airport_df) == 0:
        print("❌ Не удалось получить координаты аэропортов")
        return
    
    print(f"✅ Получены координаты для {len(airport_df)} аэропортов")
    
    # Создаем оптимизированную карту
    print("\n🚀 ЗАПУСК СОЗДАНИЯ ОПТИМИЗИРОВАННОЙ КАРТЫ...")
    visualizer.create_fast_world_map(flight_data, airport_df)
    
    print("\n🎉 ОПТИМИЗИРОВАННАЯ ВИЗУАЛИЗАЦИЯ ЗАВЕРШЕНА!")
    print("📊 СТАТИСТИКА:")
    print(f"   - Обработано перелетов: {len(flight_data)}")
    print(f"   - Уникальных аэропортов: {len(airport_df)}")
    print(f"   - Файл карты: /home/mariia/Загрузки/Telegram Desktop/AI2/world_flights_OPTIMIZED.html")
    print(f"   - Координаты аэропортов в кэше: {len(visualizer.airport_coords)}")

if __name__ == "__main__":
    main()
