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

class PassengerFlightMapVisualizer:
    """Класс для визуализации перелетов конкретного пассажира"""
    
    def __init__(self):
        self.geolocator = Nominatim(user_agent="passenger_flight_analysis")
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
            # Добавляем российские аэропорты из данных
            'MCX': (42.8167, 47.6527),  # Махачкала
            'SVO': (55.9726, 37.4146),   # Москва Шереметьево
            'KRR': (45.0347, 39.1706),   # Краснодар
            'KXK': (50.4094, 136.9342),  # Комсомольск-на-Амуре
            'SGC': (61.3437, 73.4019),   # Сургут
            'VVO': (43.3983, 132.1480),  # Владивосток
            'KJA': (56.1729, 92.4933),   # Красноярск
            'KHV': (48.5280, 135.1885),  # Хабаровск
            'ROV': (47.2582, 39.8181),   # Ростов-на-Дону
            'ASF': (46.2833, 48.0063),   # Астрахань
            'EGO': (50.6438, 36.5901),   # Белгород
            'STW': (45.1092, 42.1128),   # Ставрополь
            'SVX': (56.7431, 60.8027),   # Екатеринбург
            'HMA': (61.0285, 69.0861),   # Ханты-Мансийск
            'UFA': (54.5575, 55.8744),   # Уфа
            'NGK': (52.0875, 113.4781),  # Нижнеангарск
            'SLY': (66.5908, 66.6214),   # Салехард
            'NBC': (55.5647, 52.0884),   # Набережные Челны
            'NUX': (66.0694, 76.5183),   # Новый Уренгой
            'SCW': (61.6764, 50.7739),   # Сыктывкар
            'UUS': (46.8887, 142.7175),  # Южно-Сахалинск
            'IKT': (52.2680, 104.3890),  # Иркутск
            'PKC': (53.1679, 158.4536),  # Петропавловск-Камчатский
            'TJM': (57.1896, 65.3243),   # Тюмень
            'PEE': (57.9145, 56.0219),   # Пермь
            'BAX': (53.3638, 83.5385),   # Барнаул
            'KGD': (54.8901, 20.5926),   # Калининград
            'KZN': (55.6062, 49.2787),   # Казань
            'VOG': (48.7825, 44.3455),   # Волгоград
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
    
    def get_passenger_data(self, passenger_file, first_name, last_name):
        """Получаем данные конкретного пассажира"""
        print(f"🔍 Поиск данных для пассажира: {first_name} {last_name}")
        
        # Загружаем данные подозрительных пассажиров
        suspicious_df = pd.read_csv(passenger_file)
        
        # Ищем пассажира
        passenger_data = suspicious_df[
            (suspicious_df['first_name'] == first_name) & 
            (suspicious_df['last_name'] == last_name)
        ]
        
        if len(passenger_data) == 0:
            print(f"❌ Пассажир {first_name} {last_name} не найден")
            return None
        
        print(f"✅ Найден пассажир: {first_name} {last_name}")
        return passenger_data.iloc[0]
    
    def create_passenger_flight_data(self, passenger_row):
        """Создаем данные о перелетах пассажира на основе аэропортов"""
        print("✈️ Создание данных о перелетах пассажира...")
        
        departure_airports = []
        arrival_airports = []
        
        # Извлекаем аэропорты вылета
        if 'departure_airports_str' in passenger_row and pd.notna(passenger_row['departure_airports_str']):
            departure_airports = [ap.strip() for ap in str(passenger_row['departure_airports_str']).split(',')]
        
        # Извлекаем аэропорты прилета  
        if 'arrival_airports_str' in passenger_row and pd.notna(passenger_row['arrival_airports_str']):
            arrival_airports = [ap.strip() for ap in str(passenger_row['arrival_airports_str']).split(',')]
        
        print(f"📍 Аэропорты вылета: {departure_airports}")
        print(f"📍 Аэропорты прилета: {arrival_airports}")
        
        # Создаем DataFrame с перелетами
        flights = []
        
        # Для простоты создаем перелеты между всеми аэропортами вылета и прилета
        # В реальном сценарии нужно использовать точные данные о маршрутах
        for i, dep_airport in enumerate(departure_airports):
            if i < len(arrival_airports):
                arr_airport = arrival_airports[i % len(arrival_airports)]
                flights.append({
                    'departure': dep_airport,
                    'arrival': arr_airport,
                    'flight_id': f"FL{i+1:03d}"
                })
        
        flight_df = pd.DataFrame(flights)
        print(f"✅ Создано {len(flight_df)} перелетов")
        return flight_df
    
    def create_optimized_coordinates(self, flight_data, max_airports=200):
        """Оптимизированное создание координат аэропортов"""
        print("🗺️ Создание координат аэропортов...")
        
        # Собираем уникальные аэропорты
        all_airports = set()
        if 'departure' in flight_data.columns:
            all_airports.update(flight_data['departure'].unique())
        if 'arrival' in flight_data.columns:
            all_airports.update(flight_data['arrival'].unique())
        
        print(f"🔍 Найдено уникальных аэропортов: {len(all_airports)}")
        
        airport_data = []
        for i, airport in enumerate(all_airports):
            coords = self.fast_geocode_airport(airport)
            if coords:
                # Считаем количество вылетов и прилетов
                dep_count = len(flight_data[flight_data['departure'] == airport])
                arr_count = len(flight_data[flight_data['arrival'] == airport])
                
                airport_data.append({
                    'airport': airport,
                    'lat': coords[0],
                    'lon': coords[1],
                    'departures_count': dep_count,
                    'arrivals_count': arr_count,
                    'flights_count': dep_count + arr_count
                })
        
        return pd.DataFrame(airport_data)
    
    def create_passenger_flight_map(self, flight_data, airport_df, passenger_info, output_path=None):
        """Создание карты перелетов пассажира"""
        print("🌍 Создание карты перелетов пассажира...")
        
        if output_path is None:
            output_path = f"/home/mariia/Загрузки/Telegram Desktop/AI2/passenger_{passenger_info['first_name']}_{passenger_info['last_name']}_flights.html"
        
        # Создаем карту
        fig = go.Figure()
        
        # Добавляем аэропорты с разными цветами для вылетов и прилетов
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
        for i, (_, flight) in enumerate(flight_data.iterrows()):
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
        risk_category = passenger_info.get('risk_category', 'НЕИЗВЕСТНО')
        risk_score = passenger_info.get('risk_score', 'НЕИЗВЕСТНО')
        n_flights = passenger_info.get('n_flights_total', 'НЕИЗВЕСТНО')
        
        title = f"🌍 КАРТА ПЕРЕЛЕТОВ: {passenger_info['first_name']} {passenger_info['last_name']}<br>"
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
        print(f"✅ Карта перелетов сохранена: {output_path}")
        
        return fig

def main():
    """Главная функция программы"""
    print("👤 ПРОГРАММА ВИЗУАЛИЗАЦИИ ПЕРЕЛЕТОВ КОНКРЕТНОГО ПАССАЖИРА")
    print("=" * 60)
    
    # Создаем визуализатор
    visualizer = PassengerFlightMapVisualizer()
    
    # Запрашиваем данные пассажира
    print("\n👤 ВВЕДИТЕ ДАННЫЕ ПАССАЖИРА:")
    first_name = input("Имя: ").strip()
    last_name = input("Фамилия: ").strip()
    
    # Получаем данные пассажира
    passenger_file = "/home/mariia/Загрузки/Telegram Desktop/AI2/suspicious_passengers_activity_patterns.csv"
    passenger_data = visualizer.get_passenger_data(passenger_file, first_name, last_name)
    
    if passenger_data is None:
        print("❌ Не удалось найти данные пассажира")
        return
    
    # Создаем данные о перелетах
    flight_data = visualizer.create_passenger_flight_data(passenger_data)
    
    if len(flight_data) == 0:
        print("❌ Нет данных о перелетах для визуализации")
        return
    
    # Создаем координаты аэропортов
    airport_df = visualizer.create_optimized_coordinates(flight_data)
    
    if len(airport_df) == 0:
        print("❌ Не удалось получить координаты аэропортов")
        return
    
    print(f"✅ Получены координаты для {len(airport_df)} аэропортов")
    
    # Создаем карту перелетов
    print("\n🚀 СОЗДАНИЕ КАРТЫ ПЕРЕЛЕТОВ...")
    visualizer.create_passenger_flight_map(flight_data, airport_df, passenger_data)
    
    print("\n🎉 ВИЗУАЛИЗАЦИЯ ПЕРЕЛЕТОВ ПАССАЖИРА ЗАВЕРШЕНА!")
    print("📊 СТАТИСТИКА:")
    print(f"   - Пассажир: {first_name} {last_name}")
    print(f"   - Категория риска: {passenger_data.get('risk_category', 'НЕИЗВЕСТНО')}")
    print(f"   - Всего перелетов: {len(flight_data)}")
    print(f"   - Уникальных аэропортов: {len(airport_df)}")
    print(f"   - Файл карты: /home/mariia/Загрузки/Telegram Desktop/AI2/passenger_{first_name}_{last_name}_flights.html")

if __name__ == "__main__":
    main()
