import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from datetime import datetime
from utile import *


class UnifiedMultiEntityDataset(Dataset):
    def __init__(self, combined_data, seq_length, stride=1, financial_keys=None, financial_seq_length=10, max_unused_days=4):
        """
        Args:
            combined_data (dict): Dictionary with datasets (e.g., weather, forex, financial) as keys and MultiIndex DataFrames as values.
            seq_length (int): Length of time sequences for daily data.
            stride (int): Step size for sliding window.
            financial_keys (list): Keys in combined_data corresponding to financial datasets.
            financial_seq_length (int): Sequence length for financial data (in months).
        """
        self.combined_data = combined_data
        self.seq_length = seq_length
        self.stride = stride
        self.financial_keys = financial_keys or []
        self.financial_seq_length = financial_seq_length
        self.max_unused_days = max_unused_days

        # Ensure all datasets have the same timestamps
        self.entities_per_dataset = {
            key: sorted(data.index.get_level_values(0).unique()) for key, data in combined_data.items()
        }
        self.time_index = combined_data[next(iter(combined_data))].index.get_level_values(1).unique()
        self.num_time_steps = len(self.time_index)

        # Precompute financial index
        if "finance" in combined_data:
            financial_data = combined_data["finance"]
            self.financial_index = financial_data.index.get_level_values(1).to_period("M")  # Monthly period index
        else:
            # Create dummy financial data
            start_date = "1997-07-01"
            end_date = "2024-11-30"
            financial_index = pd.date_range(start=start_date, end=end_date, freq="MS")
            dummy_data = np.random.rand(len(financial_index), 5)  # Example dummy data with 5 columns (features)

            # Create a dummy DataFrame with financial index
            financial_data = pd.DataFrame(dummy_data, index=financial_index, columns=[f"feature_{i+1}" for i in range(dummy_data.shape[1])])
            self.financial_index = financial_data.index.to_period("M")  # Convert to monthly period index

            print("Financial data not found in combined_data. Dummy financial data created.")
            # raise ValueError("Financial data not found in combined_data.")
        
        for key, data in combined_data.items():
            missing_data = data[data.isnull().any(axis=1)]
            if not missing_data.empty:
                print(f"Missing data in {key}:")
                print(missing_data.tail())


    def __len__(self):
        # Number of sequences depends on time steps, sequence length, and stride
        # Subtract 3 days (max_unused_days) from the total time steps to exclude the last 3 days
        return (self.num_time_steps - self.seq_length - self.max_unused_days) // self.stride + 1

    def __getitem__(self, idx):
        """
        Returns:
            x_dict: Dictionary of inputs for all datasets.
                    Keys are dataset names, values are tensors of shape (seq_length, num_entities, num_features).
            y_dict: Dictionary of targets for all datasets.
                    Keys are dataset names, values are tensors of shape (num_entities, num_features).
        """
        x_dict = {}
        y_dict = {}

        # Calculate end time for the current batch
        end_time = self.time_index[idx + self.seq_length - 1]
        end_month = end_time.to_period("M")
        end_month_idx = self.financial_index.get_loc(end_month)

        if isinstance(end_month_idx, np.ndarray):
            end_month_idx = int(np.where(end_month_idx)[0][0]) - 1
        start_month_idx = end_month_idx - (self.financial_seq_length - 1)

        for key, data in self.combined_data.items():
            entities = self.entities_per_dataset[key]
            x_list = []
            y_list = []

            # for entity in entities:
            #     entity_data = data.loc[entity]

            #     if key == "finance":
            #         # Financial data: Use precomputed monthly indices
            #         x_seq = entity_data.iloc[start_month_idx:end_month_idx + 1].values
            #         y_target = entity_data.iloc[end_month_idx+1].values
            #     else:
            #         # Other data: Use daily sequences
            #         x_seq = entity_data.iloc[idx : idx + self.seq_length].values
            #         y_target = entity_data.iloc[idx + self.seq_length + 1].values

            for entity in entities:
                entity_data = data.loc[entity]

                if key == "finance":
                    # Financial data: Use precomputed monthly indices
                    x_seq = entity_data.iloc[start_month_idx:end_month_idx + 1].values
                    # Calculate y_target
                    current_value = entity_data.iloc[end_month_idx + 1].values
                    previous_value = entity_data.iloc[end_month_idx - 1].values

                    # Binary classification (sign of the change)
                    sign_value = current_value - previous_value

                    direction = [1 if sign > 0 else 0 for sign in sign_value ]

                    # Normalized change
                    magnitude = (current_value - previous_value) / current_value

                    confidence = [1] * len(current_value)

                elif "forex" or "commodities":
                    # Other data: Use daily sequences
                    x_seq = entity_data.iloc[idx : idx + self.seq_length].values
                    # Calculate y_target

                    # # Get the current and previous values using the recursive function
                    current_value = entity_data.iloc[idx + self.seq_length + 1].values[3]
                    previous_value = entity_data.iloc[idx + self.seq_length - 1].values[3]


                    # current_value, previous_value = self.get_valid_values(entity_data, idx)


                    # Binary classification (sign of the change)
                    sign_value = current_value - previous_value

                    direction = [1 if sign_value > 0 else 0] 

                    # Normalized change
                    magnitude = [(current_value - previous_value) / current_value]

                    confidence = [1]
                     
                else:
                    # Other data: Use daily sequences
                    x_seq = entity_data.iloc[idx : idx + self.seq_length].values
                    # Calculate y_target
                    current_value = entity_data.iloc[idx + self.seq_length + 1].values
                    previous_value = entity_data.iloc[idx + self.seq_length - 1].values

                    # Binary classification (sign of the change)
                    sign_value = current_value - previous_value

                    direction = [1 if sign > 0 else 0 for sign in sign_value ]

                    # Normalized change
                    magnitude = (current_value - previous_value) / current_value

                    confidence = [1] * len(current_value)
                
                y_target = np.array([direction, magnitude, confidence])


                x_list.append(x_seq)
                y_list.append(y_target)


            # Convert all sequences for the current dataset to tensors
            x_dict[key] = torch.tensor(x_list).permute(1, 0, 2)
            y_dict[key] = torch.tensor(y_list).squeeze()

        # Metadata preparation
        metadata = {
            "timestamps": [ts.isoformat() for ts in self.time_index[idx : idx + self.seq_length]],
            "timestamps_monthly": [
                ts for ts in self.financial_index[start_month_idx : end_month_idx + 1]
            ],
            "entity_mapping": self.entities_per_dataset,
        }

        return x_dict, y_dict, metadata
    
    def get_valid_values(self, entity_data, idx, max_lookahead=3, max_lookbehind=3):
        """
        This function recursively finds valid values for both the current and previous values
        using linear interpolation if necessary.
        
        Args:
            entity_data (DataFrame): The data frame containing entity data.
            idx (int): The current index in the data.
            seq_length (int): The sequence length used for indexing.
            max_lookahead (int): The number of future data points to check for a valid value.
            max_lookbehind (int): The number of past data points to check for a valid value.

        Returns:
            tuple: (current_value, previous_value)
        """
        # Get the initial current and previous values
        current_value = entity_data.iloc[idx + self.seq_length + 1].values[3]
        previous_value = entity_data.iloc[idx + self.seq_length - 1].values[3]
        i = 0

        # Handle NaN or 0 for current_value (lookahead)
        if np.isnan(current_value) or current_value == 0:
            for i in range(2, max_lookahead + 1):
                next_value = entity_data.iloc[idx + self.seq_length + i].values[3]
                if not np.isnan(next_value) and next_value != 0:
                    current_value = self.linear_interpolate(previous_value, next_value, i - 1, max_lookahead)
                    break
            else:
                current_value = previous_value  # If no valid next value, use the previous value

        # Handle NaN or 0 for previous_value (lookbehind)
        if np.isnan(previous_value) or previous_value == 0:
            for i in range(1, max_lookbehind + 1):
                prev_value = entity_data.iloc[idx + self.seq_length - i].values[3]
                if not np.isnan(prev_value) and prev_value != 0:
                    previous_value = prev_value
                    current_value = self.linear_interpolate(previous_value, current_value, i, max_lookbehind)
                    break
            else:
                previous_value = current_value  # If no valid previous value, use the current value

        return current_value, previous_value
    
    def linear_interpolate(self, previous_value, next_value, distance, max_distance=3):
        """
        Performs linear interpolation between two values based on distance.

        Args:
            previous_value (float): The previous value before the missing point.
            next_value (float): The next valid value after the missing point.
            distance (int): The distance (in terms of number of data points) from the previous value.
            max_distance (int): The maximum number of lookahead steps to consider.

        Returns:
            float: The interpolated value.
        """
        weight = 1 - (distance / max_distance)  # The closer the next value, the higher the weight.
        interpolated_value = previous_value * weight + next_value * (1 - weight)
        return interpolated_value
    
