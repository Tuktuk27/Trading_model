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

                    # print(f"{current_value  = }")
                    # print(f"{previous_value = }")
                    # print(f"{idx = }")
                    # print(f"{entity = }")
                    # print(f"{key = }")
                    # print(f"{end_time = }")


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
    

#     I don't know how the model will do in case of missing data. Do you know? Should it take take nan values and just ignore them, or should we interpolate or duplicate from others, all missing data ? So first, what should we do at best, and second is the model able to do so. Or do you have other solutions? 

#     Project Overview:
# The project focuses on gathering and analyzing a wide range of economic, weather, and market data to predict forex (foreign exchange) and commodities performance. The goal is to build a model that can be used for forecasting price movements and volatility based on factors like economic indicators, weather data, market volatility, and commodity-specific factors (such as demand, supply, geopolitical stability, and regional influences).

# Steps for the Project:
# Data Collection: We’ve collected data across several categories:

# Weather Data: Global weather data for relevant countries and regions, focusing on factors like temperature, precipitation, and drought conditions that can influence agricultural and energy commodities.
# Forex Data: Historical forex pair data for major currencies (e.g., EUR/USD, GBP/USD) obtained from free sources like Yahoo Finance.
# Commodity Data: Historical price data for a selected list of agricultural commodities (wheat, coffee, etc.), energy commodities (ethanol, coal, etc.), and battery metals (lithium, cobalt, etc.), as well as key niche commodities (rubber, pork bellies, etc.).
# Volatility Data: Market volatility indices (e.g., VIX for the U.S., VSTOXX for Europe) to track market sentiment and potential risk factors.
# Economic Indicators: A wide set of economic data like GDP, inflation, unemployment rates, and industrial production data for relevant countries.
# Selected Commodities:

# Agricultural Commodities: Coffee, Soybeans, Corn, Wheat, Sugar.
# Energy Commodities: Ethanol.
# Rare Metals: Lithium, Nickel, Cobalt.
# Niche Commodities: Rubber, Pork Bellies, Wool.
# Coal: Coal price proxies, including the VanEck Vectors Coal ETF (KOL) and companies like Peabody Energy (BTU) and Arch Coal (ARCH).
# Data Sources:

# Weather data obtained from sources like Meteostat and OpenWeather for historical weather data.
# Forex data downloaded from Yahoo Finance.
# Commodity data from Yahoo Finance (for various ETFs, futures, and commodity indexes).
# Economic and volatility indicators collected from various free APIs like Alpha Vantage, NOAA, and World Bank.
# Volatility indices like the VIX and VSTOXX, as well as specific national indices (India VIX, Brazil VXEWZ, etc.).
# Data Formats:

# JSON: All data, including weather, forex, and economic indicators, is stored in JSON format to facilitate easy access and manipulation.
# CSV/Excel: Some economic data, like GDP or CPI, is stored in Excel files for detailed processing and aggregation.
# ETF tickers and Commodities: Data for coal, lithium, and others is gathered through ETFs (e.g., KOL for coal) and related companies (e.g., BTU, ARCH).
# Data Organization:

# Forex Data: Collected in a structured format for each pair (e.g., EUR/USD, GBP/USD) with date-indexed closing price data.
# Commodities Data: Price data for each commodity is stored by ticker symbol (e.g., KC=F for Coffee, ZC=F for Corn).
# Volatility Indices: Collected for each region and country (e.g., VIX for the U.S., INDIAVIX for India, VXEWZ for Brazil) using respective ticker symbols.
# Economic Indicators: Stored by country and indicator, with files including GDP, CPI, unemployment rate, and reserves data.
# Weather Data: Stored by country and region, capturing historical weather patterns that may influence agricultural production and commodity prices.
# Model Building: The data collected will be used to build a forecasting model for forex pairs and commodity prices. The key inputs will include:

# Economic Factors: GDP, inflation, employment, and trade data to identify macroeconomic influences on prices.
# Weather Conditions: Precipitation, temperature, and seasonal patterns to influence agricultural and energy commodity forecasts.
# Market Sentiment: Volatility data to capture broader market sentiment and potential risks.
# Commodity-Specific Factors: Supply and demand, geopolitical stability, and production forecasts for the selected commodities.
# The model will aim to predict price fluctuations, identify correlations between economic indicators and market movements, and assess the risk of price volatility.

# Key Decisions Made:

# Focused on safe, niche commodities that are less likely to be heavily impacted by geopolitical risks (e.g., avoiding crude oil and natural gas).
# Chose weather data as an important factor for agricultural commodities, including coffee, soybeans, and wheat, which are sensitive to growing conditions.
# Volatility data is crucial for understanding market sentiment and will help in predicting risk for forex pairs and commodities.
# Economic indicators like GDP, CPI, and unemployment rates were chosen based on their proven impact on currency and commodity price movements.
# Next Steps:

# Begin data processing by aggregating the weather, forex, volatility, and economic data into a unified format.
# Model design: Use machine learning or statistical modeling techniques to correlate the data and build prediction models.
# Evaluate the model’s ability to predict short-term and long-term price movements in the selected forex pairs and commodities.
# Backtest the model using historical data to validate its performance.

# Why Are We Doing This?
# The primary goal of this project is to make profits by forecasting the price movements of forex pairs and commodities. However, the key focus is on safety and mitigating risk. We want to make money, but we don't want to take unnecessary risks or make decisions that could lead to large, unpredictable losses. Here’s how we’re approaching it:

# Risk Management: We’ve carefully selected niche commodities that are less prone to geopolitical events or other extreme market disruptions. By avoiding highly volatile markets like crude oil and natural gas, we’re limiting exposure to sudden, massive price swings that can be caused by unpredictable geopolitical events. This allows us to trade with more confidence and reduces the likelihood of catastrophic losses.

# Moderate Growth: While the goal is profitability, we don’t want to achieve that in an overly aggressive or speculative way. We are focusing on steady, consistent returns rather than aiming for "get-rich-quick" strategies. We’re looking for stable growth, not volatility that could jeopardize the capital we’ve built up.

# Data-Driven Decisions: The model will be grounded in data, including historical weather patterns, economic indicators, market volatility, and past commodity performance. By understanding what influences prices (such as economic growth, weather conditions, or market sentiment), we can make more informed decisions that minimize risk while still seeking profit.

# Long-Term Viability: Our approach is designed to be sustainable and profitable over time, avoiding the risks that come with high-frequency trading or excessive speculation. It’s about building a model that can adapt to market conditions and evolve with changing data.

# That being said, we don't want neither a slow growth of 1% a year. Profitability is the main goal of this project and if we don't want to take unnecessary risk, we are still aiming for the fastest and best way to make profit.

# What Should you do next (you are overtaking that project)?
# Now that we’ve gathered the necessary data, it’s important for the assistant to understand the next steps. While the direction is fairly clear, there’s room for the assistant to decide how best to proceed within the following framework:

# Data Processing & Cleaning: The next step could involve aggregating and cleaning the data. It’s important to get the data in a format that the model can use. This might mean filling in missing values, dealing with time-series inconsistencies, or normalizing data so that it’s easier to analyze.

# Exploratory Data Analysis (EDA): Once the data is ready, performing some basic analysis can help understand patterns and relationships. The assistant could identify correlations between economic indicators, weather patterns, and commodity prices. This could involve visualizations (e.g., graphs, heatmaps) and basic statistical analysis to identify key drivers of price movements.

# Model Building: Once the data is cleaned and prepared, the assistant could start building the prediction model. This might involve choosing a modeling approach (e.g., machine learning techniques, econometric models) and training the model on the historical data.

# Backtesting the Model: After the model is trained, it needs to be tested against historical data to evaluate how well it would have predicted prices. Backtesting is crucial to ensuring the model works in real-world scenarios and that it’s not overfitted to past data.

# Performance Evaluation: Finally, after backtesting, the model’s performance should be evaluated based on accuracy, reliability, and risk. This might involve adjusting the model to improve predictions or finding ways to fine-tune the risk management features.

# Have you understood the project, and your task now ? Summarize in a very quick way, without any details.

# For info: 
# All entities are included in every batch.
# For example, if I have N entities and I process sequences of 30 time steps, each batch should include the data for all entities over the same set of 30 consecutive time steps. Except for monthly data (finance) that have seqence of 10 month (starting the month prior the current daily data)

# Entity ordering is consistent.
# Each batch ensure that the entities are always in the same order (e.g., A, B, C, ...). This is critical to maintaining entity-specific representations in the model.

# Key Idea
# Fully Connected Branch (Global Branch): This branch processes all the input features together through fully connected layers. It serves to capture potential interdependencies across all the data sources right from the start.
# Specialized Branches: These branches process each data source independently (e.g., forex, weather, unemployment). They focus on extracting specific temporal or domain-relevant features.
# Fusion and Output: At the end, combine the outputs of both the global branch and the specialized branches. Then, pass the combined features through a final set of fully connected layers to predict the target outputs (e.g., forex rates, commodity prices).

# So I have updated my code to do all of this. But I have another problem:
# There will be always missing data. Some are missing because I want tdata from Mexico in 2004 for instance, and there are none avaialbe, but some might be mising from 2024 in Oslo, maybe they didn't record the precipitation or the wind speed, or maybe there is no data on the GDP for Russia in 2022/02... I can be random and a bit everywhere. But my model should be able to handle those missing values, even today if I predict for the currency of tommorw, I might have data missing, it's normal, it can be any data, and at any time. Regardless, I still should be able to predict from the avaialbe data. So is there a way to make a model able to handle missing values or Nan values and that will just rely on the ones avaialbe ? 

# import torch
# import torch.nn as nn

# class HybridModel(nn.Module):
#     def __init__(self, input_sizes, global_hidden_size, branch_hidden_sizes, fusion_hidden_size, output_size):
#         super(HybridModel, self).__init__()
        
#         # Global fully connected branch
#         self.global_fc = nn.Sequential(
#             nn.Linear(sum(input_sizes), global_hidden_size),
#             nn.ReLU(),
#             nn.Linear(global_hidden_size, global_hidden_size // 2),
#             nn.ReLU()
#         )
        
#         # Specialized branches
#         self.specialized_branches = nn.ModuleList([
#             nn.Sequential(
#                 nn.LSTM(input_size, hidden_size, batch_first=True),
#                 nn.Linear(hidden_size, branch_output_size)
#             )
#             for input_size, hidden_size, branch_output_size in branch_hidden_sizes
#         ])
        
#         # Fusion and output layers
#         self.fusion_layer = nn.Sequential(
#             nn.Linear(global_hidden_size // 2 + sum(b[2] for b in branch_hidden_sizes), fusion_hidden_size),
#             nn.ReLU()
#         )
#         self.output_layer = nn.Linear(fusion_hidden_size, output_size)
    
#     def forward(self, inputs):
#         # Global branch
#         global_input = torch.cat(inputs, dim=-1)  # Combine all inputs
#         global_features = self.global_fc(global_input)
        
#         # Specialized branches
#         branch_outputs = []
#         for branch, input_data in zip(self.specialized_branches, inputs):
#             branch_output, _ = branch(input_data)  # LSTM returns (output, (hidden, cell))
#             branch_outputs.append(branch_output[:, -1, :])  # Take the last time step
        
#         # Fusion
#         combined_features = torch.cat([global_features] + branch_outputs, dim=-1)
#         fused_features = self.fusion_layer(combined_features)
        
#         # Final prediction
#         output = self.output_layer(fused_features)
#         return output


# def forward(self, inputs):
#     # Create masks and replace NaNs
#     masks = [~torch.isnan(input_data) for input_data in inputs]
#     inputs = [torch.nan_to_num(input_data, nan=0.0) for input_data in inputs]
    
#     # Global branch
#     global_input = torch.cat(inputs, dim=-1)  # Combine all inputs
#     global_features = self.global_fc(global_input) * torch.cat(masks, dim=-1).float()

#     # Specialized branches
#     branch_outputs = []
#     for branch, input_data, mask in zip(self.specialized_branches, inputs, masks):
#         branch_output, _ = branch(input_data * mask.float())
#         branch_outputs.append(branch_output[:, -1, :] * mask[:, -1, None].float())

#     # Fusion
#     combined_features = torch.cat([global_features] + branch_outputs, dim=-1)
#     fused_features = self.fusion_layer(combined_features)
    
#     # Final prediction
#     output = self.output_layer(fused_features)
#     return output



# class EntityTimeSeriesDataset(Dataset):
#     def __init__(self, data, seq_length, target_col=None):
#         """
#         Dataset for a single entity's time-series data.
#         :param data: DataFrame (Timestamp, Metrics).
#         :param seq_length: Length of input sequences.
#         :param target_col: Target column name. If None, predict all columns.
#         """
#         self.data = data
#         self.seq_length = seq_length
#         self.target_col = target_col
#         self.values = data.values  # Numpy array of data
#         self.columns = data.columns  # Column names

#     def __len__(self):
#         return len(self.data) - self.seq_length

#     def __getitem__(self, idx):
#         x = self.values[idx:idx + self.seq_length]
#         y = (
#             self.values[idx + self.seq_length]
#             if self.target_col is None
#             else self.values[idx + self.seq_length, self.columns.get_loc(self.target_col)]
#         )
#         return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# class MultiEntityTimeSeriesDataset(Dataset):
#     def __init__(self, data_df, seq_length, stride=1):
#         self.data_df = data_df
#         self.seq_length = seq_length
#         self.stride = stride

#         # Ensure consistent entity ordering
#         self.entities = sorted(data_df.index.get_level_values(0).unique())
#         self.num_entities = len(self.entities)

#         # Create a unified timestamp index across all entities
#         self.time_index = list(data_df.loc[self.entities[0]].index)  # Assuming all entities align
#         self.num_time_steps = len(self.time_index)

#     def __len__(self):
#         # Adjust the number of sequences based on the stride
#         return (self.num_time_steps - self.seq_length) // self.stride

#     def __getitem__(self, idx):
#         # Adjust idx to account for the stride
#         idx = idx * self.stride
        
#         x_list = []
#         y_list = []
#         for entity in self.entities:
#             data = self.data_df.loc[entity]
#             x_seq = data.iloc[idx : idx + self.seq_length].values  # Sequence of length `seq_length`
#             y_target = data.iloc[idx + self.seq_length].values     # Target at `t + seq_length`
#             x_list.append(x_seq)
#             y_list.append(y_target)
        
#         # Convert to tensors
#         x = torch.tensor(x_list).permute(1, 0, 2)  # Shape: (seq_length, num_entities, num_features)
#         y = torch.tensor(y_list)  # Shape: (num_entities, num_features)
        
#         return x, y
