import pandas as pd
from pathlib import Path

# Constants
METERS_TO_FEET = 3.28084

# Input/Output paths
input_file = Path('newlyn.txt')
output_file = Path('newlyn_coops_format.csv')

# Read the Newlyn data
# Note: File has trailing commas, so we specify usecols to ignore extra column
df = pd.read_csv(input_file, usecols=[0, 1, 2])
df.columns = df.columns.str.strip()

# Parse the "Time GMT" column to extract date and time
# Format: "2026/01/27 00:08"
df['DateTime'] = pd.to_datetime(df['Time GMT'], format='%Y/%m/%d %H:%M')
df['Date'] = df['DateTime'].dt.strftime('%Y/%m/%d')
df['Time (GMT)'] = df['DateTime'].dt.strftime('%H:%M')

# Convert tide height from meters to feet
df['Verified (ft)'] = (df['tide height in metres to ACD'] * METERS_TO_FEET).round(2)

# Create the output dataframe in CO-OPS format
output_df = pd.DataFrame({
    'Date': df['Date'],
    'Time (GMT)': df['Time (GMT)'],
    'Predicted (ft)': '-',
    'Preliminary (ft)': '-',
    'Verified (ft)': df['Verified (ft)']
})

# Write to CSV with proper formatting (quoted strings like CO-OPS format)
output_df.to_csv(output_file, index=False, quoting=1)  # quoting=1 means QUOTE_ALL

print(f'✓ Converted {len(output_df)} records from Newlyn format to CO-OPS format')
print(f'✓ Converted tide heights from meters to feet (×{METERS_TO_FEET})')
print(f'✓ Saved to: {output_file}')
print(f'\nSample output (first 3 rows):')
print(output_df.head(3).to_string(index=False))
