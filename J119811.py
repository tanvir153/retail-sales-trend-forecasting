# J119811.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# Task 1: Data Importation, Inspection, and Merging
# Import sales data (CSV files)
store_a = pd.read_csv('store_a_sales_data.csv')
store_b = pd.read_csv('store_b_sales_data.csv')
store_c = pd.read_csv('store_c_sales_data.csv')

# Import pricing data (CSV file)
pricing = pd.read_csv('product_pricing_data.csv')

# Add store identifiers
store_a['store'] = 'Store_A'
store_b['store'] = 'Store_B'
store_c['store'] = 'Store_C'

# Merge sales data
sales_data = pd.concat([store_a, store_b, store_c], ignore_index=True)

# Auto-fix column names (lowercase, strip spaces)
sales_data.rename(columns=lambda x: x.strip().lower(), inplace=True)
pricing.rename(columns=lambda x: x.strip().lower(), inplace=True)

# Melt sales data to long format
sales_data_melted = sales_data.melt(id_vars=['date', 'store'], 
                                    value_vars=['product_1', 'product_2', 'product_3', 'product_4', 'product_5', 'product_6'],
                                    var_name='product', 
                                    value_name='sales')

# Standardize product names in pricing
pricing['product'] = pricing['product'].str.lower().str.replace(' ', '_')

# Merge sales and pricing data
data = pd.merge(sales_data_melted, pricing, on='product', how='left')

# Inspect data
print("\nData Info:")
print(data.info())
print("\nSample Data:")
print(data.head())

# Check for missing values
print("\nMissing values per column:")
print(data.isnull().sum())

# Task 2: Distribution Analysis
print("\nTask 2: Distribution Analysis")

products = data['product'].unique()

for product in products:
    product_sales = data[data['product'] == product]['sales']
    stat, p_value = shapiro(product_sales)
    print(f"\nShapiro-Wilk Test for {product}: Statistic={stat:.4f}, p-value={p_value:.4f}")
    if p_value < 0.05:
        print(f"{product} sales are **not normally distributed** (p < 0.05)")
    else:
        print(f"{product} sales are **normally distributed** (p ≥ 0.05)")
    
    plt.figure(figsize=(6, 4))
    sns.histplot(product_sales, kde=True, color=sns.color_palette("tab10")[products.tolist().index(product)])
    plt.title(f'Sales Distribution for {product}')
    plt.xlabel('Sales')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'{product}_distribution.png')
    plt.show()
    plt.close()

# Task 3: Summary Statistics
print("\nTask 3: Summary Statistics")

summary_store_product = data.groupby(['store', 'product'])['sales'].agg(['sum', 'mean']).reset_index()
print("\nTotal & Average Sales per Product per Store:")
print(summary_store_product)

summary_daily = data.groupby('date')['sales'].agg(['sum', 'mean']).reset_index()
print("\nTotal & Average Sales per Day Across All Stores:")
print(summary_daily)

# Pivot Table
pivot = data.pivot_table(index='store', columns='product', values='sales', aggfunc=['sum', 'mean'])
print("\nPivot Table of Sales (Sum & Mean):")
print(pivot)

# Profitability
data['profit'] = data['sales'] * (data['selling price'] - data['cost price'])
profit_summary = data.groupby('product')['profit'].sum().reset_index().sort_values(by='profit', ascending=False)
most_profitable = profit_summary.iloc[0]
least_profitable = profit_summary.iloc[-1]
print(f"\nMost Profitable Product: {most_profitable['product']} with Profit: {most_profitable['profit']}")
print(f"Least Profitable Product: {least_profitable['product']} with Profit: {least_profitable['profit']}")

# Visualization: Total Sales per Store
store_sales = data.groupby('store')['sales'].sum().reset_index()
plt.figure(figsize=(6, 4))
sns.barplot(x='store', y='sales', data=store_sales, hue='store', dodge=False, palette='viridis')
plt.title('Total Sales per Store')
plt.ylabel('Total Sales')
plt.tight_layout()
plt.savefig('total_sales_per_store.png')
plt.show()
plt.close()

# Visualization: Total Profit per Product
plt.figure(figsize=(6, 4))
sns.barplot(x='product', y='profit', data=profit_summary, hue='product', dodge=False, palette='rocket')
plt.title('Total Profit per Product')
plt.ylabel('Total Profit')
plt.tight_layout()
plt.savefig('total_profit_per_product.png')
plt.show()
plt.close()

# Task 4: Profitability Analysis & Predictive Modelling
print("\nTask 4: Profitability Analysis & Predictive Modelling")

daily_profit = data.groupby(['store', 'date', 'product'])['profit'].sum().reset_index()
daily_profit['cumulative_profit'] = daily_profit.groupby(['store', 'product'])['profit'].cumsum()

print("\nDaily Profit Sample:")
print(daily_profit.head())

# Predictive Modeling (Linear Regression with enhanced features)
product_to_model = products[0]
product_data = data[data['product'] == product_to_model].copy()

product_data['date'] = pd.to_datetime(product_data['date'])
product_data['date_ordinal'] = product_data['date'].map(pd.Timestamp.toordinal)
product_data['day_of_week'] = product_data['date'].dt.dayofweek
product_data['month'] = product_data['date'].dt.month
product_data['week_of_year'] = product_data['date'].dt.isocalendar().week
product_data['is_weekend'] = product_data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

X = product_data[['date_ordinal', 'day_of_week', 'month', 'week_of_year', 'is_weekend']]
y = product_data['sales']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)

# Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nLinear Regression for {product_to_model}: MSE={mse:.2f}, R2={r2:.2f}")

# Visualization: Actual vs Predicted Sales
plt.figure(figsize=(6, 4))
plt.scatter(range(len(y_test)), y_test, label='Actual', color='blue')
plt.plot(range(len(y_test)), y_pred, color='red', label='Predicted')
plt.title(f'Sales Prediction for {product_to_model}')
plt.xlabel('Test Data Index')
plt.ylabel('Sales')
plt.legend()
plt.tight_layout()
plt.savefig(f'{product_to_model}_prediction.png')
plt.show()
plt.close()

# Forecast Future Sales (next 30 days)
future_dates = pd.date_range(start=product_data['date'].max() + pd.Timedelta(days=1), periods=30, freq='D')
future_features = pd.DataFrame({
    'date_ordinal': future_dates.map(pd.Timestamp.toordinal),
    'day_of_week': future_dates.dayofweek,
    'month': future_dates.month,
    'week_of_year': future_dates.isocalendar().week,
    'is_weekend': future_dates.dayofweek >= 5
})
future_scaled = scaler.transform(future_features)
future_preds = model.predict(future_scaled)

# Visualization: Future Sales Forecast
plt.figure(figsize=(8, 5))
plt.plot(future_dates, future_preds, label='Future Predictions', color='green')
plt.title(f'Future Sales Forecast for {product_to_model} (Next 30 Days)')
plt.xlabel('Date')
plt.ylabel('Predicted Sales')
plt.legend()
plt.tight_layout()
plt.savefig(f'{product_to_model}_future_forecast.png')
plt.show()
plt.close()

# Visualization: Cumulative Profit Over Time (Top 3 Products)
top_products = profit_summary['product'].head(3).tolist()
cumulative_data = data[data['product'].isin(top_products)].copy()
cumulative_data['date'] = pd.to_datetime(cumulative_data['date'])
cumulative_profit = cumulative_data.groupby(['product', 'date'])['profit'].sum().groupby(level=0).cumsum().reset_index()

plt.figure(figsize=(8, 5))
sns.lineplot(x='date', y='profit', hue='product', data=cumulative_profit, palette='tab10')
plt.title('Cumulative Profit Over Time (Top 3 Products)')
plt.ylabel('Cumulative Profit')
plt.tight_layout()
plt.savefig('cumulative_profit_top_products.png')
plt.show()
plt.close()

# Task 5: Insights & Recommendations
print("\nTask 5: Insights & Recommendations")

print("\nTop-Performing Store(s):")
top_store = summary_store_product.groupby('store')['sum'].sum().reset_index().sort_values(by='sum', ascending=False)
print(top_store)

print("\nTop-Performing Product(s):")
print(profit_summary)

print("\nRecommendations:")
print("- Increase stock levels for high-performing products, especially in top-performing stores.")
print("- Reconsider pricing for least profitable products.")
print("- Monitor sales trends of each product and adjust marketing strategies accordingly.")
print("- Based on the sales forecast for product_1, monitor for potential growth opportunities and prepare inventory accordingly.")
