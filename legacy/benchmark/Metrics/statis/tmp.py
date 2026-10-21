import pandas as pd

# Load the CSV file
file_path = 'method_comparison_wodiffsbdd.csv'  # 替换为你的CSV文件路径
df = pd.read_csv(file_path)



# Remove variance parts from specific columns
columns_to_clean = ["fsscore", "rascore", "sa", "scscore", "syba"]
for col in columns_to_clean:
    if col in df.columns:
        # Remove the variance part (anything after ±)
        df[col] = df[col].str.split("±").str[0].str.strip()

# Convert numeric columns to float for ranking
for col in df.columns[1:]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Generate ranks with sorting direction
ranked_df = pd.DataFrame()
ranked_df['Method'] = df['Method']  # Retain the method names

for column in df.columns[1:]:
    if column in ["sa", "scscore"]:  # Sort these columns in ascending order
        ranked_df[column] = df[column].rank(ascending=True)
    else:  # Sort other columns in descending order
        ranked_df[column] = df[column].rank(ascending=False)

# Save the ranked dataframe to a new CSV
output_path = 'ranked_method_comparison.csv'
ranked_df.to_csv(output_path, index=False)

print(f"Ranked CSV saved to {output_path}")
