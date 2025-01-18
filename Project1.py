import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import subprocess

# Load the dataset
try:
    df = pd.read_csv('Amazon_Sale_Report.csv')
except FileNotFoundError:
    print("CSV File not found. Please check the file path.")
    exit()


# Display the first few rows of the DataFrame
print(df.head)


# Show dataset info
print(df.info)
print(df.shape)


# Drop the column 'Unnamed:22' if it exists
print(df.columns)  # This will print the exact column names


if 'Unnamed: 22' in df.columns:
    df.drop(['Unnamed: 22'], axis=1, inplace=True)
if 'promotion-ids' in df.columns:
    df.drop(['promotion-ids'], axis=1, inplace=True)

# Check the cleaned DataFrame
print("Columns after cleaning:")
print(df.columns)

null=pd.isnull(df).sum()
print(null)

df.dropna(inplace=True)
print(df.shape)
print(df.head(8))


#check that the data set is null
print("\nNull values after cleaning:")
print(df.isnull().sum())


# Remove duplicate rows
df.drop_duplicates(inplace=True)
print(df.columns)
df.to_csv('Amazon_Sale_Report.csv', index=False)


# Display basic statistics for numerical columns
print(df.describe())

# Check for unique values in each column (useful for categorical columns)
print(df.nunique())

# Check for any constant columns (columns with the same value throughout)
constant_columns = [col for col in df.columns if df[col].nunique() == 1]
print("Constant columns:", constant_columns)


# Standardize column names (remove spaces, special characters, and convert to lowercase)
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(r'[^\w]', '')
print("Standardized column names:", df.columns)


# Convert columns to appropriate data types if necessary
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    df[col] = df[col].astype('category')


df.to_csv('Amazon_Sale_Report.csv', index=False)
# Reset index to have sequential values starting from 0
df.reset_index(drop=True, inplace=True)

# Now, update the index to start from 1 instead of 0
df.index = df.index + 1

# Check the result
print(df.head())





#Keep Track of the changes
metadata = {
    "initial_shape": (14151, 24),
    "final_shape": df.shape,
    "removed_columns": ['Unnamed: 22', 'promotion-ids'],
    "missing_values_filled": df.isnull().sum().to_dict()
}
print(metadata)


numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=df, x=col)
    plt.title(f"Boxplot for {col}")
    plt.show()

print("Final Dataset Shape:", df.shape)
print("Final Missing Values:", df.isnull().sum().sum())
print("Sample Rows:")
print(df.head())


# Visualize the distribution of the 'Price' column
plt.figure(figsize=(10, 6))
plt.hist(df['amount'], bins=30, edgecolor='r')
plt.title('Distribution of Price')
plt.xlabel('amount')
plt.ylabel('Frequency')
plt.show()
print(df.dtypes)

#Group for sales and fulfillment analysis
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='date', y='amount', alpha=0.6, color='blue', edgecolor='black')
plt.title('Scatter Plot of Amount vs Date', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Amount', fontsize=12)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Group data by SKU or ASIN and calculate total sales
if 'sku' in df.columns:
    sales_by_sku = df.groupby('sku',observed=True)['amount'].sum().sort_values(ascending=False)

    # Plot the top 10 SKUs by sales
    plt.figure(figsize=(12, 6))
    sales_by_sku.head(10).plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Top 10 SKUs by Total Sales', fontsize=16)
    plt.xlabel('SKU', fontsize=12)
    plt.ylabel('Total Sales (Amount)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# For ASIN
if 'asin' in df.columns:
    sales_by_asin = df.groupby('asin',observed=True)['amount'].sum().sort_values(ascending=False)

    # Plot the top 10 ASINs by sales
    plt.figure(figsize=(12, 6))
    sales_by_asin.head(10).plot(kind='bar', color='orange', edgecolor='black')
    plt.title('Top 10 ASINs by Total Sales', fontsize=16)
    plt.xlabel('ASIN', fontsize=12)
    plt.ylabel('Total Sales (Amount)', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


# Launch Power BI after processing
pbix_file_path = r"C:\Users\ankur\OneDrive\Documents\for python\Project.pbix" 

if os.path.exists(pbix_file_path):
    try:
        subprocess.run(["start", "Power BI", pbix_file_path], shell=True)
        print(f"Power BI file '{pbix_file_path}' opened successfully!")
    except Exception as e:
        print(f"Error launching Power BI: {e}")
else:
    print(f"Power BI file not found at {pbix_file_path}")