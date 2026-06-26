"""
Generator data sintetis untuk demo aplikasi, karena environment build ini
tidak memiliki akses internet untuk mengunduh dataset Kaggle asli
(US Accidents / Air Pollution in China). Struktur data yang dihasilkan
sengaja dibuat menyerupai kriteria yang diminta:

Project 1 (mirip US Accidents):
 - > 20 kolom, banyak baris (default 60.000 baris demo / bisa diubah)
 - Missing value, outlier, fitur kategorikal, butuh transformasi

Project 2 (mirip Air Pollution in China 2015-2025):
 - Time series harian multi-kota dengan tren, musiman, noise

Pengguna dapat mengganti generator ini dengan dataset asli melalui
tombol "Load CSV" pada aplikasi tanpa mengubah kode lain.
"""

import numpy as np
import pandas as pd


def generate_accidents_like_dataset(n_rows=60000, seed=42):
    rng = np.random.RandomState(seed)
    n = n_rows

    severity = rng.choice([1, 2, 3, 4], size=n, p=[0.1, 0.5, 0.3, 0.1])
    start_lat = rng.uniform(25, 49, n)
    start_lng = rng.uniform(-124, -67, n)
    distance_mi = np.abs(rng.normal(0.5, 0.8, n))
    temperature = rng.normal(60, 18, n)
    humidity = rng.uniform(10, 100, n)
    pressure = rng.normal(29.8, 0.6, n)
    visibility = rng.normal(9, 2.5, n)
    wind_speed = np.abs(rng.normal(7, 5, n))
    precipitation = np.abs(rng.exponential(0.05, n))

    states = ["CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI"]
    weather = ["Clear", "Cloudy", "Rain", "Snow", "Fog", "Thunderstorm", None]
    wind_dir = ["N", "S", "E", "W", "NE", "NW", "SE", "SW", None]
    side = ["L", "R"]

    df = pd.DataFrame({
        "ID": np.arange(n),
        "Severity": severity,
        "Start_Lat": start_lat,
        "Start_Lng": start_lng,
        "Distance_mi": distance_mi,
        "Temperature_F": temperature,
        "Humidity_pct": humidity,
        "Pressure_in": pressure,
        "Visibility_mi": visibility,
        "Wind_Speed_mph": wind_speed,
        "Precipitation_in": precipitation,
        "State": rng.choice(states, n),
        "Weather_Condition": rng.choice(weather, n, p=[0.3, 0.2, 0.15, 0.05, 0.1, 0.05, 0.15]),
        "Wind_Direction": rng.choice(wind_dir, n),
        "Side": rng.choice(side, n),
        "Amenity": rng.choice([True, False], n, p=[0.05, 0.95]),
        "Bump": rng.choice([True, False], n, p=[0.02, 0.98]),
        "Crossing": rng.choice([True, False], n, p=[0.15, 0.85]),
        "Junction": rng.choice([True, False], n, p=[0.1, 0.9]),
        "Railway": rng.choice([True, False], n, p=[0.02, 0.98]),
        "Stop": rng.choice([True, False], n, p=[0.05, 0.95]),
        "Traffic_Signal": rng.choice([True, False], n, p=[0.2, 0.8]),
        "Sunrise_Sunset": rng.choice(["Day", "Night", None], n, p=[0.55, 0.4, 0.05]),
        "Hour": rng.randint(0, 24, n),
        "DayOfWeek": rng.randint(0, 7, n),
    })

    # inject missing values
    for col in ["Temperature_F", "Humidity_pct", "Pressure_in", "Visibility_mi",
                "Wind_Speed_mph", "Precipitation_in", "Weather_Condition", "Wind_Direction"]:
        mask = rng.rand(n) < 0.07
        df.loc[mask, col] = np.nan

    # inject outliers
    outlier_idx = rng.choice(n, size=int(n * 0.01), replace=False)
    df.loc[outlier_idx, "Temperature_F"] += rng.choice([-1, 1], len(outlier_idx)) * rng.uniform(80, 150, len(outlier_idx))
    outlier_idx2 = rng.choice(n, size=int(n * 0.01), replace=False)
    df.loc[outlier_idx2, "Distance_mi"] += rng.uniform(20, 100, len(outlier_idx2))

    return df


def generate_air_pollution_like_dataset(start="2015-01-01", end="2025-12-31",
                                         cities=("Beijing", "Shanghai", "Guangzhou"),
                                         seed=7):
    rng = np.random.RandomState(seed)
    dates = pd.date_range(start=start, end=end, freq="D")
    n = len(dates)
    rows = []
    for ci, city in enumerate(cities):
        t = np.arange(n)
        trend = -0.01 * t / n * 50 + 80  # slight downward long-term trend
        seasonal = 25 * np.sin(2 * np.pi * t / 365.25 + ci) + 10 * np.cos(2 * np.pi * t / 7)
        noise = rng.normal(0, 8, n)
        pm25 = np.clip(trend + seasonal + noise + rng.normal(10 * ci, 1), 2, None)
        pm10 = pm25 * rng.uniform(1.2, 1.6, n)
        no2 = np.clip(20 + 10 * np.sin(2 * np.pi * t / 365.25) + rng.normal(0, 5, n), 1, None)
        so2 = np.clip(8 + 4 * np.cos(2 * np.pi * t / 365.25) + rng.normal(0, 2, n), 0.5, None)
        co = np.clip(0.8 + 0.3 * np.sin(2 * np.pi * t / 365.25) + rng.normal(0, 0.1, n), 0.1, None)
        o3 = np.clip(40 + 20 * np.cos(2 * np.pi * t / 365.25 + 1) + rng.normal(0, 6, n), 1, None)
        temp = 15 + 15 * np.sin(2 * np.pi * t / 365.25 - 1.5) + rng.normal(0, 3, n)
        humidity = np.clip(55 + 20 * np.sin(2 * np.pi * t / 365.25) + rng.normal(0, 8, n), 5, 100)

        city_df = pd.DataFrame({
            "Date": dates,
            "City": city,
            "PM2_5": pm25,
            "PM10": pm10,
            "NO2": no2,
            "SO2": so2,
            "CO": co,
            "O3": o3,
            "Temperature": temp,
            "Humidity": humidity,
        })
        rows.append(city_df)

    df = pd.concat(rows, ignore_index=True)

    # missing values
    mask = rng.rand(len(df)) < 0.03
    df.loc[mask, "PM2_5"] = np.nan

    # outliers
    out_idx = rng.choice(len(df), size=int(len(df) * 0.005), replace=False)
    df.loc[out_idx, "PM2_5"] = df.loc[out_idx, "PM2_5"].fillna(50) + rng.uniform(150, 300, len(out_idx))

    return df
