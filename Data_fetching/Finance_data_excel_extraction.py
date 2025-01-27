import os
import pandas as pd
import json
import numpy as np


path = r"C:\Users\tugdu\Desktop\Trading\finance_data\GemDataEXTR"
store_name_file = []

# Get list of all Excel files in the specified directory
for root, folders, files in os.walk(path):
    for file in files:
        name = os.path.basename(file)
        store_name_file.append(name)

forex_commo_indicator_list = [
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

countries_list = ["Bangladesh", "Kenya", "Nigeria", "Egypt", "Pakistan", "Venezuela", "Ukraine", "Iran", "Netherlands", "Thailand", "United Arab Emirates", "Vietnam", "Malaysia", "Switzerland", "Japan", "South Korea", "Hong Kong", "Singapore", "India", "Argentina", "Chile", "South Africa", "Mexico", "China", "Brazil", "Saudi Arabia", "Russia", "Australia", "Canada", "Indonesia",  "Turkey", "Morocco", "United Kingdom", "Spain", "Italy", "France", "Germany", "Belgium", "United States", "Norway", "Denmark","Luxembourg", "Taiwan","Isreal", "Greece", "Colombia","Poland","Peru", "Sweden", "Finland", "Hong Kong SAR, China", "Taiwan, China", "Viet Nam", "Korea, Rep.", "Egypt, Arab Rep.", "Russian Federation", "Kazakhstan", "Uzbekistan", "Qatar", "Ethiopia", "Angola", "Algeria", "Ghana", "Ivory Coast", "Advanced Economies",	"EMDE East Asia & Pacific",	"EMDE Europe & Central Asia",	"Emerging Market and Developing Economies (EMDEs)",	"High Income Countries",	"EMDE Latin America & Caribbean",	"Low-Income Countries (LIC)",	"Middle-Income Countries (MIC)",	"EMDE Middle East & N. Africa",	"EMDE South Asia",	"EMDE Sub-Saharan Africa",	"World (WBG members)",
]

json_data = {}

# # Helper function to convert 'YYYYMM' to Unix timestamp (milliseconds)
# def convert_date_to_timestamp(date_str):
#     try:
#         return int(pd.to_datetime(date_str, format='%YM%m').timestamp() * 1000)
#     except:
#         return None
    
# Helper function to convert 'YYYYMM' or 'YYYYQx' to Unix timestamp (milliseconds)
def convert_date_to_timestamp(date_str):
    try:
        # Check if the date format is quarterly (e.g., 1995Q2)
        if 'Q' in date_str:
            # Extract year and quarter (e.g., 1995Q2 -> 1995-04-01)
            year, quarter = date_str[:4], date_str[5]
            month = (int(quarter) - 1) * 3 + 1  # Convert quarter to month (Q1 -> Jan, Q2 -> Apr, Q3 -> Jul, Q4 -> Oct)
            return int(pd.to_datetime(f"{year}-{month:02d}-01").timestamp() * 1000)
        else:
            # Handle monthly format (e.g., 1995M09)
            return int(pd.to_datetime(date_str, format='%YM%m').timestamp() * 1000)
    except Exception as e:
        print(f"Error converting date {date_str}: {e}")
        return None
    
# Placeholder for JSON structure
columns = []
index = []
data = []

def expand_quarter_to_months(df, master_index):
    """
    Expand quarterly data to monthly rows using linear interpolation.

    Parameters:
        df (pd.DataFrame): The quarterly data DataFrame. Index is assumed to be quarterly strings (e.g., '1995Q1').
        master_index (list): A list of valid Unix timestamps for alignment.

    Returns:
        pd.DataFrame: Monthly data with interpolated values.
    """
    # Convert the quarterly index to datetime (start of the quarter)
    df.index = pd.PeriodIndex(df.index, freq='Q').to_timestamp()

    # Reindex to include all months in the range of the data
    monthly_index = pd.date_range(start=df.index.min(), end=df.index.max(), freq='M')

    # Create a DataFrame with the new index
    df_monthly = df.reindex(monthly_index)

    # Interpolate values linearly for the missing months
    df_monthly = df_monthly.interpolate(method='linear')

    # Convert the index to Unix timestamps (in milliseconds) and filter by master index
    df_monthly.index = (df_monthly.index.astype('int64') // 10**6).astype(int)
    df_monthly = df_monthly[df_monthly.index.isin(master_index)]

    return df_monthly


# Create the master index from the first monthly sheet
master_index = None

for file in store_name_file:
    if file in forex_commo_indicator_list:
        # Load the Excel file
        excel_path = os.path.join(root, file)
        xl = pd.ExcelFile(excel_path)
        quarterly = True

        # Identify sheets
        monthly_sheets = [sheet for sheet in xl.sheet_names if 'month' in sheet.lower()]

        if len(monthly_sheets)< 1 :
            quarterly_sheets = [sheet for sheet in xl.sheet_names if 'quarterly' in sheet.lower()]
        else:
            quarterly_sheets = []

        for sheet in monthly_sheets + quarterly_sheets:
            df = xl.parse(sheet)

            # Ensure the first column is the date and set it as the index
            df.set_index(df.columns[0], inplace=True)

            # Filter valid columns
            valid_columns = [col for col in countries_list if col in df.columns]
            df = df[valid_columns]

            # Drop rows with missing dates
            df = df[df.index.notna()]

            # Convert index to timestamps
            if sheet in quarterly_sheets:
                df.index = df.index.map(lambda x: str(x))  # Ensure index is string for quarterly parsing
                df = expand_quarter_to_months(df, master_index)
                print(f"Processing: {file}")
            else:
                df.index = df.index.map(convert_date_to_timestamp)

            # Initialize the master index from the first monthly sheet
            if master_index is None and sheet in monthly_sheets:
                master_index = df.index.tolist()

            # Align data to master index
            if master_index:
                    # Find any timestamps in df.index that are not in master_index
                missing_in_master = df.index.difference(master_index)

                if not missing_in_master.empty:
                    print(f"Warning: The following timestamps in df.index are not in master_index:\n{missing_in_master}")
                    # Optionally, raise an error if this is considered critical
                    # raise ValueError(f"Timestamps in df.index not found in master_index: {missing_in_master}")

                df = df.reindex(master_index, fill_value=np.nan)

            # Append data to combined structure
            for country in valid_columns:
                column_name = f"{file}"
                if column_name not in columns:
                    columns.append([column_name, country])
                column_data = df[country].tolist()
                data.append(column_data)

            

# Transpose the data to match JSON structure
aligned_data = list(map(list, zip(*data)))

# Create final JSON structure
final_data = {
    "columns": columns,
    "index": [date for date in master_index],
    "data": aligned_data,
}

# Save to a JSON file
output_path = os.path.join(root, "combined_data.json")
with open(output_path, "w") as json_file:
    json.dump(final_data, json_file, indent=4)


