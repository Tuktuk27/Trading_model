import yfinance as yf
import h5py
from meteostat import Point, Daily
from datetime import datetime
import pandas as pd


# Time range
start = datetime(2000, 1, 1)
end = datetime(2025, 1, 1)


currencies = ["EUR/USD", "GBP/USD", "AUD/USD", "JPY/USD", "CAD/USD", "TRY/USD", "SEK/USD", "THB/USD", "PLN/USD", "NZD/USD"]

forex_data = {}
for currency in currencies: 
    currency_pair = currency.replace("/","")

    try: 
        data = yf.download(f"{currency_pair}=X", start=start, end=end)


                # Save individual location data as JSON
        if not data.empty:
            filename = f"{currency_pair}_pair.json"
            data.to_json(filename, orient="split")
            print(f"Saved forex history data for {currency} to {filename}.")
            
            # Store data for combining later
            forex_data[currency] = data
        else:
            print(f"No data found for {currency}.")

    except Exception as e:
        print(f"Failed to fetch data for {currency}: {e}")

# Combine all location data into a single DataFrame
if forex_data:
    forex_output_name = "global_forex_data.json"
    combined_data = pd.concat(forex_data.values(), keys=forex_data.keys())
    
    # Save combined data as a single JSON file
    combined_data.to_json(forex_output_name, orient="split")
    print(f"Combined forex data saved to {forex_output_name}.")
else:
    print("No forex data collected to combine.")

print("Data collection complete!")




