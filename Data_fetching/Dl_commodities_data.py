import yfinance as yf
import pandas as pd
from datetime import datetime
import json

# Time range for historical data
start = datetime(2000, 1, 1)
end = datetime(2025, 1, 1)

# Commodities with their Yahoo Finance tickers (if available)
commodities = {
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

# Create a dictionary to store the data for each commodity
commodity_data = {}

# Download and save commodity data
for commodity, ticker in commodities.items():
    try:
        # Fetch historical data for each commodity
        data = yf.download(ticker, start=start, end=end)

        if not data.empty:
            # Save individual commodity data as JSON
            filename = f"{commodity.replace(' ', '_')}_data.json"
            data.to_json(filename, orient="split")
            print(f"Saved {commodity} data to {filename}.")
            
            # Store data for combining later
            commodity_data[commodity] = data
        else:
            print(f"No data found for {commodity}.")

    except Exception as e:
        print(f"Failed to fetch data for {commodity}: {e}")

# Combine all commodity data into a single DataFrame
if commodity_data:
    combined_filename = "global_commodities_data.json"
    combined_data = pd.concat(commodity_data.values(), keys=commodity_data.keys())

    # Save combined data as a single JSON file
    combined_data.to_json(combined_filename, orient="split")
    print(f"Combined commodities data saved to {combined_filename}.")
else:
    print("No commodity data collected to combine.")

print("Data collection complete!")
