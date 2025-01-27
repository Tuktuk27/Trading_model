import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

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


def custom_collate_fn(batch):
    x_batch, y_batch, metadata_batch = zip(*batch)  # Unpack the batch

    # Batch x_dict and y_dict using PyTorch's default_collate
    x_batch = torch.utils.data._utils.collate.default_collate(x_batch)
    y_batch = torch.utils.data._utils.collate.default_collate(y_batch)
        
    # Keep metadata as a simple list of lists
    return x_batch, y_batch, metadata_batch