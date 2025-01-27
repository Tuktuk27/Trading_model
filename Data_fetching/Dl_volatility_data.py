import yfinance as yf
import pandas as pd
from datetime import datetime

# Time range
start = datetime(2000, 1, 1)
end = datetime(2025, 1, 1)


country_volatility_indices = {
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


# Function to fetch and save volatility data
def fetch_volatility_data():
    volatility_data = {}
    
    # Loop through each country and fetch corresponding volatility index data
    for country, volatility_index in country_volatility_indices.items():
        try:
            print(f"Fetching volatility data for {country} using {volatility_index}...")
            
            # Fetch volatility data from Yahoo Finance
            data = yf.download(volatility_index, start=start, end=end)
            
            # Save data to JSON if not empty
            if not data.empty:
                filename = f"{country}_volatility.json"
                data.to_json(filename, orient="split")
                print(f"Saved volatility data for {country} to {filename}.")
                
                # Store data for combining later
                volatility_data[country] = data
            else:
                print(f"No data found for {country}.")
        
        except Exception as e:
            print(f"Failed to fetch data for {country}: {e}")
    
    return volatility_data

# Fetch and save the data
volatility_data = fetch_volatility_data()

# Combine all location data into a single DataFrame
if volatility_data:
    volatility_output_name = "global_volatility_data.json"
    combined_data = pd.concat(volatility_data.values(), keys=volatility_data.keys())
    
    # Save combined data as a single JSON file
    combined_data.to_json(volatility_output_name, orient="split")
    print(f"Combined volatility data saved to {volatility_output_name}.")
else:
    print("No volatility data collected to combine.")

print("Data collection complete!")
