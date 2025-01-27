import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from datetime import datetime
from utile import *

# month_start="1997-07-01"
# month_end="2024-11-30"
# day_start="2000-01-01"
# day_end="2024-11-30"

month_start="2017-07-01"
month_end="2020-11-30"
day_start="2020-01-01"
day_end="2020-11-30"


class GlobalDataLoader:
    def __init__(self, folder_paths):
        self.folder_paths = folder_paths
        self.data = {}
        self.processed_data = {}

    def load_json_data(self):
        """Load JSON data from each folder."""
        for folder_name, path in self.folder_paths.items():
            file_path = os.path.join(path, f"global_{folder_name}_data.json")
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    self.data[folder_name] = json.load(f)
            else:
                print(f"File not found: {file_path}")
        print(f"{self.data.keys() = }")

    def process_finance_data(self, data_name="finance"):
        """Process data into a unified format."""
        forex_data = self.data.get(data_name)
        if not forex_data:
            raise ValueError(f"{data_name} data not loaded.")

        # Extract columns, index, and data
        columns = forex_data['columns']
        raw_index = forex_data['index']
        raw_data = forex_data['data']

        # Create a unified timestamp from 2000/01/01 to 2025/01/01
        unified_timestamps = pd.date_range(start=month_start, end=month_end, freq="MS")

        # Organize columns into a dictionary for easy grouping by currency pair
        column_mapping = {}
        for i, col in enumerate(columns):
            metric, country = col
            if metric not in column_mapping:
                column_mapping[metric] = {}
            column_mapping[metric][country] = i


        # Process data for each currency pair
        processed_frames = {}
        things_list = set(idx[0] for idx in columns)  # Extract unique thing

        for thing in things_list:

            # Filter only the required countries
            countries = list(set(idx for idx in list(column_mapping[thing].keys())))

            selected_columns = [column_mapping[thing][country] for country in countries]
            filtered_data = [[row[col] for col in selected_columns] for row in raw_data]

            # Create a DataFrame for the currency pair
            df = pd.DataFrame(filtered_data, columns=countries, index=raw_index)
            df.index = pd.to_datetime(df.index, unit='ms')  # Convert to datetime
            df = df.reindex(unified_timestamps, method="nearest")  # Align to unified timestamps

            processed_frames[thing] = df

        # Combine all currency pairs into a single DataFrame with MultiIndex
        self.processed_data[data_name] = pd.concat(processed_frames, axis=0, keys=processed_frames.keys())


    def process_data(self, data_name):
        """Process data into a unified format."""
        dictionnary = data_dictionnary[data_name]
        forex_data = self.data.get(data_name)
        if not forex_data:
            raise ValueError(f"{data_name} data not loaded.")

        # Extract columns, index, and data
        columns = forex_data['columns']
        raw_index = forex_data['index']
        raw_data = forex_data['data']

        # Create a unified timestamp from 2000/01/01 to 2025/01/01
        unified_timestamps = pd.date_range(start=day_start, end=day_end, freq="D")

        # Organize columns into a dictionary for easy grouping by currency pair
        column_mapping = {}
        for i, col in enumerate(columns):
            metric, pair = col
            if pair not in column_mapping:
                column_mapping[pair] = {}
            column_mapping[pair][metric] = i

        # Process data for each currency pair
        processed_frames = {}
        things_list = set(idx[0] for idx in raw_index)  # Extract unique thing
        print(column_mapping)

        for thing in things_list:
            short_thing = dictionnary[thing]

            # Extract indices and data for the currency pair
            pair_indices = [i for i, idx in enumerate(raw_index) if idx[0] == thing]
            pair_timestamps = [raw_index[i][1] for i in pair_indices]
            pair_data = [raw_data[i] for i in pair_indices]

            # Filter only the required metrics (Close, High, Low, Open, Volume)
            metrics = list(set(idx[0] for idx in columns))

            selected_columns = [column_mapping[short_thing][metric] for metric in metrics if metric in column_mapping[short_thing]]
            filtered_data = [[row[col] for col in selected_columns] for row in pair_data]

            # Create a DataFrame for the currency pair
            df = pd.DataFrame(filtered_data, columns=metrics, index=pair_timestamps)
            df.index = pd.to_datetime(df.index, unit='ms')  # Convert to datetime
            df = df.reindex(unified_timestamps, method="nearest")  # Align to unified timestamps

            processed_frames[thing] = df

        # Combine all currency pairs into a single DataFrame with MultiIndex
        self.processed_data[data_name] = pd.concat(processed_frames, axis=0, keys=processed_frames.keys())

    def process_weather_data(self, data_name):
        """Process weather data into a unified format."""
        weather_data = self.data.get(data_name)
        if not weather_data:
            raise ValueError("Weather data not loaded.")

        # Extract columns, index, and data
        columns = weather_data['columns']
        raw_index = weather_data['index']
        raw_data = weather_data['data']

        # Create a unified timestamp from 2000/01/01 to 2025/01/01
        unified_timestamps = pd.date_range(start=day_start, end=day_end, freq="D")

        # Convert raw data into a DataFrame
        locations = set(idx[0] for idx in raw_index)  # Extract unique locations
        processed_frames = {}

        for location in locations:
            # Extract data for the location
            loc_indices = [i for i, idx in enumerate(raw_index) if idx[0] == location]
            loc_timestamps = [raw_index[i][1] for i in loc_indices]
            loc_data = [raw_data[i] for i in loc_indices]

            # Create a DataFrame for the location
            df = pd.DataFrame(loc_data, columns=columns, index=loc_timestamps)
            df.index = pd.to_datetime(df.index, unit='ms')  # Convert to datetime
            df = df.reindex(unified_timestamps, method="nearest")  # Align to unified timestamps

            processed_frames[location] = df

        # Combine all locations into a single DataFrame with MultiIndex
        self.processed_data['weather'] = pd.concat(processed_frames, axis=0, keys=processed_frames.keys())

    def call_process_data(self, data_name = None):
        if data_name is None:
            for k in self.data.keys():
                print(f"{k = }")
                if k == "weather":
                    self.process_weather_data(k)
                elif k == "finance":
                    self.process_finance_data(k)
                else:
                    self.process_data(k)
        else:
            if data_name == "weather":
                self.process_weather_data(data_name)
            elif data_name == "finance":
                self.process_finance_data(data_name)
            else:
                self.process_data(data_name)

    def get_processed_data(self):
        """Return processed data."""
        return self.processed_data