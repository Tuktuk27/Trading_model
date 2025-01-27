import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from datetime import datetime
from Load_class import UnifiedMultiEntityDataset #, MultiEntityTimeSeriesDataset
from DualBranchForecastingModel import DualBranchForecastingModel
from HybridForecastingModel import HybridForecastingModel
from GlobalDataLoader import GlobalDataLoader
from utile import *
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from SimpleTransformerForecastingModel import SimpleTransformerForecastingModel

commodities_dict = {
    # Agricultural Commodities
    "Coffee Arabica": "KC=F",  # Coffee futures, Arabica
    "Soybeans": "ZS=F",  # Soybean Futures
    "Corn": "ZC=F",  # Corn Futures
    "Wheat": "ZW=F",  # Wheat Futures
    "Sugar": "SB=F",  # Sugar Futures

    # Energy Commodities
    "Crude Oil (WTI)": "CL=F",  # West Texas Intermediate (WTI) Crude Oil Futures
    "Natural Gas": "NG=F",  # Natural Gas Futures
    "Ethanol": "ETH=F",  # Ethanol Futures (Ethanol is less commonly available, so we use the ETF or future ticker)
    "Coal": "KOL",  # VanEck Vectors Coal ETF
    "Coal Companies Hong Kong": "1088.HK",  # Peabody Energy, Arch Coal, China Shenhua Energy

    # Rare Metals
    "Lithium": "LITH-USD",#"LITH-USD=F",  # Placeholder for Lithium; may need a broader ETF like "LIT" or a different proxy
    "Nickel": "NIK",  # Nickel Futures ^SPGSIK

    # Niche Commodities
    "Pork Bellies": "PB",  # Pork Bellies Futures (Live Hogs Futures)
    "Wool": "WOOL-USD",  # Wool can be hard to find directly, using a general proxy (e.g., Wool ETF, if available)

    # Renewable Energy (via ETFs)
    "Renewable Energy": "ICLN",  # Global Clean Energy ETF
    "Solar Energy": "TAN",  # Solar Energy ETF
    "Wind Energy": "FAN",  # Wind Energy ETF
    
}

volatility_dict = {
    "United States": "^VIX",  # VIX for US
    "Eurozone": "^EVZ",  # VSTOXX for Europe
    "India": "INDIAVIX",  # India VIX
    "Brazil": "^VXEWZ",  # Brazilian Volatility Index
    "Japan": "NKVF.OS",  # Japan Volatility Index
    "China": "VXFXI",  # Example (not always available, fallback can be an ETF like FXI or GXC)
    "South Korea": "SPKLVKP",  # South Korean Volatility Index (example)
    "United Kingdom": "VIXUK",  # UK Volatility Index (example)
    "Australia": "AXVI",  # Australian Volatility Index (example)
    "South Africa": "EZA",  # South Africa ETF as a proxy for volatility
    "Middle East": "^TASI.SR",  # Saudi Arabia index as a proxy for Middle East volatility
    "Argentina": "EEM",  # Emerging Markets ETF as a proxy
    # Major Countries with known volatility indices
    "United States": "^VIX",  # VIX for US
    "Eurozone": "^EVZ",  # VSTOXX for Europe
    "India": "^INDIAVIX",  # India VIX
    "Brazil": "^VXEWZ",  # Brazilian Volatility Index
    "Japan": "^NKVF.OS",  # Japan Volatility Index
    "South Korea": "^SPKLVKP",  # South Korean Volatility Index
    "United Kingdom": "^VIXUK",  # UK Volatility Index
    "Australia": "^AXVI",  # Australian Volatility Index


    "India": "VIXIN",  # India VIX
    "Japan": "JVI",  # Japan Volatility Index
    "China": "VIXCHINA",  # Example (not always available)
    "South Korea": "VIXSK",  # South Korean Volatility Index (example)
    "United Kingdom": "VIXUK",  # UK Volatility Index (example)
    "Australia": "^VIXAUS",  # Australian Volatility Index (example)

    # Countries where volatility indices are not directly available (using ETFs as proxies)
    "South Africa": "EZA",  # South Africa ETF as a proxy for volatility
    "China": "^VXFXI",  # Example (not always available, fallback can be an ETF like FXI or GXC)
    "Argentina": "EEM",  # Emerging Markets ETF as a proxy
    "Mexico": "EWW",  # Mexico ETF as a proxy for volatility
    "Russia": "RSX",  # Russia ETF as a proxy for volatility
    "Turkey": "TUR",  # Turkey ETF as a proxy for volatility
    "Thailand": "THD",  # Thailand ETF as a proxy for volatility
    "Egypt": "EGPT",  # Egypt ETF as a proxy for volatility
    "Saudi Arabia": "KSA",  # Saudi Arabia ETF as a proxy for volatility
    "Israel": "ISRA",  # Israel ETF as a proxy for volatility
    "United Arab Emirates": "UAE",  # UAE ETF as a proxy for volatility
    "Singapore": "ESGD",  # Singapore ETF as a proxy for volatility
    "Vietnam": "VNM",  # Vietnam ETF as a proxy for volatility
    "Chile": "ECH",  # Chile ETF as a proxy for volatility
    "Colombia": "GXG",  # Colombia ETF as a proxy for volatility
    "Peru": "EPU",  # Peru ETF as a proxy for volatility
    "Malaysia": "EWM",  # Malaysia ETF as a proxy for volatility
    "Philippines": "EPHE",  # Philippines ETF as a proxy for volatility
    "Indonesia": "EIDO",  # Indonesia ETF as a proxy for volatility
    "Pakistan": "PAK",  # Pakistan ETF as a proxy for volatility
    "Kazakhstan": "KAZ",  # Kazakhstan ETF as a proxy for volatility

    # More general regional or frontier market proxies
    "Africa": "AFK",  # Africa ETF as a proxy for volatility (includes multiple African countries)
    "Emerging Markets": "EEM",  # MSCI Emerging Markets ETF (covers multiple countries including many in Africa and Asia)
    "Frontier Markets": "FM",  # MSCI Frontier Markets ETF (covers countries in Africa, Central Asia, and others)

    # Additional ETF proxies for broader or regional volatility
    "Europe": "^VSTOXX",  # VSTOXX (Eurozone Volatility Index) or ETF alternatives
    "Latin America": "ILF",  # iShares Latin America 40 ETF
    "Sub-Saharan Africa": "AFK",  # Sub-Saharan Africa ETF
}

currencies_dict = {"EUR/USD": "EURUSD=X", "GBP/USD":"GBPUSD=X", "AUD/USD":"AUDUSD=X", "JPY/USD":"JPYUSD=X", "CAD/USD": "CADUSD=X", "TRY/USD": "TRYUSD=X", "SEK/USD":"SEKUSD=X", "THB/USD":"THBUSD=X", "PLN/USD":"PLNUSD=X", "NZD/USD":"NZDUSD=X"}

finance_keys = [
"Exchange rate, new LCU per USD extended backward, period average.xlsx",
"Exchange rate, old LCU per USD extended forward, period average.xlsx",
"Nominal Effective Exchange Rate.xlsx",
"Official exchange rate, LCU per USD, period average.xlsx",
"Real Effective Exchange Rate.xlsx",

"Core CPI, seas. adj..xlsx",
"Core CPI, not seas. adj..xlsx",
"CPI Price, % y-o-y, nominal, seas. adj..xlsx",
"CPI Price, % y-o-y, median weighted, seas. adj..xlsx",
"CPI Price, nominal, seas. adj..xlsx",

"Exports Merchandise, Customs, current US$, millions, seas. adj..xlsx",
"Imports Merchandise, Customs, current US$, millions, seas. adj..xlsx",
"Exports Merchandise, Customs, Price, US$, seas. adj..xlsx",
"Imports Merchandise, Customs, Price, US$, seas. adj..xlsx",
"Terms of Trade.xlsx",

"Foreign Reserves, Months Import Cover, Goods.xlsx",
"Total Reserves.xlsx",
"GDP at market prices, current US$, millions, seas. adj..xlsx",
"GDP at market prices, constant 2010 US$, millions, seas. adj..xlsx",

"Industrial Production, constant 2010 US$, seas. adj..xlsx",

"Unemployment Rate, seas. adj..xlsx",
"Retail Sales Volume Index, seas. adj..xlsx",
"Stock Markets, US$.xlsx",
"Stock Markets, LCU.xlsx",
]

data_dictionnary = {"commodities": commodities_dict,"volatility": volatility_dict, "forex": currencies_dict}

def nan_aware_mse_loss(predictions, targets):
    # Create mask to identify non-NaN entries
    mask = ~torch.isnan(targets)  # True where target is not NaN
    # Apply mask to the loss calculation
    mse_loss = ((predictions - targets) ** 2) * mask.float()  # Only compute loss for valid targets
    return mse_loss.sum() / mask.sum()  # Normalize by the number of valid targets


# Example function for training loop
def train_model(model, dataloader, num_epochs=10, learning_rate=1e-3, strength = False):
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Scheduler (optional)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.7)

    previous_loss = 0

    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        total_batches = len(dataloader)
        
        for batch_idx, (x_dict, y_dict, metadata) in enumerate(tqdm(dataloader)):
            # Prepare inputs and targets
            inputs = {key: x_dict[key].to(device).float() for key in x_dict}
            targets = {key: y_dict[key].to(device).float() for key in y_dict}

            # print(f"{inputs["forex"] = }")
            # print(f"{targets["forex"] = }")
            # print(f"{metadata = }")
            # print(f"{inputs["forex"].shape = }")
            # print(f"{targets["forex"].shape = }")

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            predictions = model(inputs)

            # Compute the losses for all datasets
            total_loss = 0.0
            for dataset, output in predictions.items():
                # Assume the model output has three components: direction, strength, and confidence
                direction_pred = output["direction"]
                if strength:
                    strength_pred = output["strength"]
                    confidence_pred = output["confidence"]

                # print(f"{direction_pred = }")
                # print(f"{strength_pred = }")
                # print(f"{confidence_pred = }")

                # Get the target values for this dataset
                direction_target = targets[dataset][:,:,0]
                if strength:
                    strength_target = targets[dataset][:,:,1]
                    confidence_target = 1- abs(output["strength"]-targets[dataset][:,:,1])
                # print(f"{direction_target = }")
                # Calculate the individual losses
                loss_direction = criterion_direction(direction_pred, direction_target)
                if strength:
                    loss_strength = criterion_strength(strength_pred, strength_target)
                    loss_confidence = criterion_confidence(confidence_pred, confidence_target)

                # print(f"{loss_direction = }")
                if strength:
                    print(f"{loss_strength = }")
                    print(f"{loss_confidence = }")

                # Sum the individual losses for this dataset
                if strength:
                    dataset_loss = loss_direction.masked_fill(torch.isnan(loss_direction), 0) + loss_strength.masked_fill(torch.isnan(loss_strength), 0) + loss_confidence.masked_fill(torch.isnan(loss_confidence), 0)
                else:
                    dataset_loss = loss_direction.masked_fill(torch.isnan(loss_direction), 0)
                # dataset_loss = loss_direction + loss_strength + loss_confidence
                total_loss += dataset_loss

            # Backward pass
            total_loss.backward()


            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print(f"Layer: {name}")
            #         print(f"Gradient: {param.grad}")


            #         # Check gradient values (after backward pass)
            # total_norm = 0
            # for param in model.parameters():
            #     if param.grad is not None:
            #         total_norm += param.grad.data.norm(2).item() ** 2
            # total_norm = total_norm ** 0.5

            # print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Total Gradient Norm: {total_norm}")
            

            # Optimizer step
            optimizer.step()

            # Update running loss
            running_loss += total_loss.item()

            # Print statistics every 10 batches
            if batch_idx % 10 == 0:
                improve = previous_loss - total_loss.item()
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{total_batches}, Loss: {total_loss.item():.4f}")
                print(f"Loss evolution: {improve = }. Is learning: {np.sign(improve) > 0}")
                print(f"{direction_pred = }")
                print(f"{direction_target = }")
                if strength:
                    print(f"{strength_pred = }")
                    print(f"{strength_target = }")
                    print(f"{confidence_pred = }")
                    print(f"{confidence_target = }")

                previous_loss = total_loss.item()

        # Update learning rate
        scheduler.step()

        # Print epoch loss
        avg_loss = running_loss / total_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    print("Training complete!")



# Example usage
if __name__ == "__main__":
    # Define folder paths
    folder_paths = {
        # "weather": "weather_data",
        "forex": "forex_data",
        "commodities": "commodities_data",
        # "volatility":"volatility_data",
        # "finance":"finance_data",
    }

    # Initialize and load data
    loader = GlobalDataLoader(folder_paths)
    loader.load_json_data()

    # data_type = "volatility"

    # loader.call_process_data(data_type)
    # df = loader.get_processed_data()[data_type]


    # # Display the first few rows
    # print(df.head())
    # # Display the last few rows
    # print(df.tail())
    # # Display a random sample of rows
    # print(df.sample(10))

    # print(df.shape)
    # # print(df.loc["Accra"])
    # print(df.index.get_level_values(0).unique())


    loader.call_process_data()
    all_data = loader.get_processed_data()

    print(type(all_data))

    # Assume `weather_data` is a DataFrame for weather metrics
    seq_length = 30
    seq_length_finance = 30
    stride = 1
    num_epochs=100
    learning_rate=1e-4
    number_in_batch = 2
    batch_size = 2
    strength = False


    # dataset = EntityTimeSeriesDataset(weather_df, seq_length)
    # dataset = MultiEntityTimeSeriesDataset(all_data["weather"], seq_length, stride)

    # # Create a DataLoader for batch processing
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # # Iterate through batches
    # for x, y in dataloader:
    #     print(f"Input batch shape: {x.shape}, Target batch shape: {y.shape}")
    #     break

    # total_batches = len(dataloader)
    # print(f"Total number of batches: {total_batches}")

    # Define the dataset and dataloader
    unified_dataset = UnifiedMultiEntityDataset(all_data, seq_length, stride, finance_keys, seq_length_finance)
    # dataloader = DataLoader(unified_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)

    # Get a subset of the dataset to create 10 batches
    subset_size = number_in_batch * batch_size  # 10 batches with batch size 32
    small_dataset = torch.utils.data.Subset(unified_dataset, range(subset_size))
    dataloader = DataLoader(small_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn, drop_last=False)

    total_batches = len(dataloader)
    print(f"Total number of batches: {total_batches}")

    input_shapes = {}

    # Check shapes
    for x_dict, y_dict, metadata in dataloader:
        print(metadata[0])

        inputs = {key: x_dict[key] for key in x_dict}
        targets = {key: y_dict[key] for key in y_dict}

        for key in x_dict:
            print(f"Dataset: {key}, Input shape: {x_dict[key].shape}, Target shape: {y_dict[key].shape}")
            input_shapes[key] = tuple(x_dict[key].shape[1:])
        break
    print(f"{input_shapes = }")

    input_shapes={
        "forex": (seq_length, 10, 5),
        "commodities": (seq_length, 17, 5),
        # "volatility": (seq_length, 28, 5),
        # "weather" : (seq_length, 46, 10),
        # "finance": (seq_length_finance, 24, 66),
    }
    hidden_sizes = {
        "forex": 128,
        "commodities": 128,
        # "volatility": 128,
        # "weather": 128,
        # "finance": 128,
    }
    num_heads = 8
    fusion_size = 256
    output_sizes = {
        "forex": 10,  # Predicting for 10 entities
        "commodities": 17,  # Predicting for 17 entities
    }

    hidden_size = 32
    num_layers = 4

    models = {"simple transformer" : SimpleTransformerForecastingModel,
              "Hybrid model" : HybridForecastingModel,
              }

    model_name = "simple transformer"

    model = models[model_name](
    input_dim=input_shapes,
    d_model=hidden_size,
    num_heads=num_heads,
    # output_sizes=output_sizes,
    num_layers = num_layers, 
    max_time = 30,
    max_locations = 100,
    )

    # model = models[model_name](
    # input_shapes=input_shapes,
    # hidden_sizes=hidden_sizes,
    # num_heads=num_heads,
    # fusion_size=fusion_size,
    # output_sizes=output_sizes,
    # strength= strength
    # )

    SimpleTransformerForecastingModel

    # inputs = {key: torch.nan_to_num(inputs[key], nan=1.0) for key in inputs}  # Replace NaNs with 0s, or another appropriate value
    predictions = model(inputs)
    for dataset, output in predictions.items():
        print(f"Predictions for {dataset}:")
        for key, value in output.items():
            print(f"{key}: {value.shape}")

    # Define the loss functions
    criterion_direction = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy for direction (0 or 1)
    criterion_strength = nn.MSELoss()  # Mean Squared Error for strength (continuous values)
    criterion_confidence = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy for confidence (0 or 1)

    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for name, param in model.named_parameters():
        print(name, param.data.mean(), param.data.std())

    # Train the model
    train_model(model, dataloader, num_epochs=num_epochs, learning_rate=learning_rate, strength = strength)



        
