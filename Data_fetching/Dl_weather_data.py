import yfinance as yf
import h5py
from meteostat import Point, Daily
from datetime import datetime
import pandas as pd

locations = [
    # North America
    {"name": "New York", "lat": 40.7128, "lon": -74.0060},
    {"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437},
    {"name": "Houston", "lat": 29.7604, "lon": -95.3698},
    {"name": "Chicago", "lat": 41.8781, "lon": -87.6298},
    {"name": "Toronto", "lat": 43.6511, "lon": -79.3837},
    {"name": "Mexico City", "lat": 19.4326, "lon": -99.1332},
    {"name": "Winnipeg", "lat": 49.8951, "lon": -97.1384},

    # South America
    {"name": "São Paulo", "lat": -23.5505, "lon": -46.6333},
    {"name": "Buenos Aires", "lat": -34.6037, "lon": -58.3816},
    {"name": "Bogotá", "lat": 4.7110, "lon": -74.0721},
    {"name": "Lima", "lat": -12.0464, "lon": -77.0428},
    {"name": "Santiago", "lat": -33.4489, "lon": -70.6693},

    # Europe
    {"name": "London", "lat": 51.5074, "lon": -0.1278},
    {"name": "Paris", "lat": 48.8566, "lon": 2.3522},
    {"name": "Berlin", "lat": 52.5200, "lon": 13.4050},
    {"name": "Madrid", "lat": 40.4168, "lon": -3.7038},
    {"name": "Rome", "lat": 41.9028, "lon": 12.4964},
    {"name": "Warsaw", "lat": 52.2297, "lon": 21.0122},
    {"name": "Oslo", "lat": 59.9139, "lon": 10.7522},
    {"name": "Athens", "lat": 37.9838, "lon": 23.7275},
    {"name": "Reykjavik", "lat": 64.1355, "lon": -21.8954},
    {"name": "Dublin", "lat": 53.3498, "lon": -6.2603},

    # Asia
    {"name": "Tokyo", "lat": 35.6895, "lon": 139.6917},
    {"name": "Beijing", "lat": 39.9042, "lon": 116.4074},
    {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777},
    {"name": "Jakarta", "lat": -6.2088, "lon": 106.8456},
    {"name": "Seoul", "lat": 37.5665, "lon": 126.9780},
    {"name": "Bangkok", "lat": 13.7563, "lon": 100.5018},
    {"name": "Manila", "lat": 14.5995, "lon": 120.9842},
    {"name": "Hanoi", "lat": 21.0285, "lon": 105.8542},

    # Middle East
    {"name": "Dubai", "lat": 25.2760, "lon": 55.2962},
    {"name": "Riyadh", "lat": 24.7136, "lon": 46.6753},
    {"name": "Cairo", "lat": 30.0444, "lon": 31.2357},
    {"name": "Istanbul", "lat": 41.0082, "lon": 28.9784},
    {"name": "Tehran", "lat": 35.6892, "lon": 51.3890},

    # Africa
    {"name": "Cape Town", "lat": -33.9249, "lon": 18.4241},
    {"name": "Lagos", "lat": 6.5244, "lon": 3.3792},
    {"name": "Nairobi", "lat": -1.2864, "lon": 36.8172},
    {"name": "Accra", "lat": 5.6037, "lon": -0.1870},
    {"name": "Algiers", "lat": 36.7372, "lon": 3.0860},

    # Oceania
    {"name": "Sydney", "lat": -33.8688, "lon": 151.2093},
    {"name": "Melbourne", "lat": -37.8136, "lon": 144.9631},
    {"name": "Auckland", "lat": -36.8485, "lon": 174.7633},
    {"name": "Port Moresby", "lat": -9.4438, "lon": 147.1803},

    # Polar regions
    {"name": "Nuuk", "lat": 64.1836, "lon": -51.7214},
    {"name": "McMurdo Station", "lat": -77.8419, "lon": 166.6863}
]

# Time range
start = datetime(2000, 1, 1)
end = datetime(2025, 1, 1)

# Initialize an empty dictionary to hold weather data for each location
weather_data = {}


# Fetch and save data
for loc in locations:
    try:
        # Define location
        location = Point(loc["lat"], loc["lon"])
        
        # Fetch weather data
        data = Daily(location, start, end).fetch()
        
        # Save individual location data as JSON
        if not data.empty:
            filename = f"{loc['name'].replace(' ', '_')}_weather.json"
            data.to_json(filename, orient="split")
            print(f"Saved weather data for {loc['name']} to {filename}.")
            
            # Store data for combining later
            weather_data[loc["name"]] = data
        else:
            print(f"No data found for {loc['name']}.")

    except Exception as e:
        print(f"Failed to fetch data for {loc['name']}: {e}")

# Combine all location data into a single DataFrame
if weather_data:
    combined_data = pd.concat(weather_data.values(), keys=weather_data.keys())
    
    # Save combined data as a single JSON file
    combined_data.to_json("global_weather_data.json", orient="split")
    print("Combined global weather data saved to global_weather_data.json.")
else:
    print("No weather data collected to combine.")

print("Data collection complete!")




