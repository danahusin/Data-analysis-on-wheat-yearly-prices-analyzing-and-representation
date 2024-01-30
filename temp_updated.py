import pandas as pd
import statistics as s
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


project_file=pd.read_excel(r"C:\Users\INTEL\Desktop\new data\wheat_prices.xlsx");
print(project_file.head())
print("---------------------------------------------------------------------")
print(project_file.tail())
print("---------------------------------------------------------------------")
plt.plot(project_file['year'], project_file['avg_closing_price'],linewidth=2,color='g')
plt.xlabel('year')
plt.ylabel('Average')
plt.title('Scatter Plot: year & Average')
plt.show()
print("---------------------------------------------------------------------")
plt.bar(project_file['year'], project_file['produced_amount'], color="green",width=0.55)
plt.xlabel('Year')
plt.ylabel('Produced Amount')
plt.title('Produced Amount by Year')
plt.show()
print("---------------------------------------------------------------------")
sns.lineplot(x='year', y='avg_closing_price', data=project_file, marker='o', color='green')
plt.xlabel('Year')
plt.ylabel('Average')
plt.title('avg closing price with year')
plt.show()
print("---------------------------------------------------------------------")
print(project_file.isnull())
print("---------------------------------------------------------------------")
mean_avg=s.mean(project_file['avg_closing_price'])
print("average of closing price is = ",mean_avg)
print("---------------------------------------------------------------------")
mean_annual=s.mean(project_file['annual_perc_change'])
print("average of annual percentage price change is = ",(mean_annual*100),"%")
print("---------------------------------------------------------------------")
max_annual=project_file['annual_perc_change'].max()
print("max percentage is = ",(max_annual*100),"%")
print("---------------------------------------------------------------------")
min_annual=project_file['annual_perc_change'].min()
print("min percentage is = ",(min_annual*100),"%")
print("---------------------------------------------------------------------")
max_avg=project_file['avg_closing_price'].max()
print("max closing price is = ",max_avg)
print("---------------------------------------------------------------------")
min_avg=project_file['avg_closing_price'].min()
print("min closing price is = ",min_avg)
print("---------------------------------------------------------------------")
max_pr=project_file['produced_amount'].max()
print("max produced amount is = ",max_pr)
print("---------------------------------------------------------------------")
min_pr=project_file['produced_amount'].min()
print("min produced amount is = ",min_pr)
print("---------------------------------------------------------------------")
mean_pr=project_file['produced_amount'].mean()
print("average of produced amount is = ",mean_pr)
print("---------------------------------------------------------------------")
project_file['produced_amount'].fillna(max_pr,inplace=True)
mean_pr=s.mean(project_file['produced_amount'])
print("average of produced amount after cleaning = ",mean_pr)
print("---------------------------------------------------------------------")
frequency_table = project_file['factors_affect_price'].value_counts()
print(frequency_table)
print("---------------------------------------------------------------------")
mode_factor=project_file['factors_affect_price'].mode()
print(mode_factor)
print("---------------------------------------------------------------------")
max_open=project_file['year_open'].max()
print("max open year price is = ",max_open)
print("---------------------------------------------------------------------")
min_open=project_file['year_open'].min()
print("min open year price is = ",min_open)
print("---------------------------------------------------------------------")
mean_open=project_file['year_open'].mean()
print("average of year_open is = ",mean_open)
print("---------------------------------------------------------------------")
max_close=project_file['year_close'].max()
print("max year_close price is = ",max_close)
print("---------------------------------------------------------------------")
min_close=project_file['year_close'].min()
print("min year_close price is = ",min_close)
print("---------------------------------------------------------------------")
mean_close=project_file['year_close'].mean()
print("average of year_close is = ",mean_close)
print("---------------------------------------------------------------------")
max_high=project_file['year_high'].max()
print("max year_high price is = ",max_high)
print("---------------------------------------------------------------------")
min_high=project_file['year_high'].min()
print("min year_high price is = ",min_high)
print("---------------------------------------------------------------------")
mean_high=project_file['year_high'].mean()
print("average of year_high is = ",mean_high)
print("---------------------------------------------------------------------")
max_low=project_file['year_low'].max()
print("max year_low price is = ",max_low)
print("---------------------------------------------------------------------")
min_low=project_file['year_low'].min()
print("min year_low price is = ",min_low)
print("---------------------------------------------------------------------")
mean_low=project_file['year_low'].mean()
print("average of year_low is = ",mean_low)
print("---------------------------------------------------------------------")
x1 = project_file['year']
y1 = project_file['avg_closing_price']

fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('Year')
ax1.set_ylabel('Average Closing Price', color=color)
ax1.plot(x1, y1, label="avg_closing_price_per_year", color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  

x2 = project_file['year']
y2 = project_file['produced_amount']

color = 'tab:red'
ax2.set_ylabel('produced amount', color=color)  
ax2.plot(x2, y2, label="produced_amount_per_year", color=color)
ax2.tick_params(axis='y', labelcolor=color)

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
plt.title("relation between produced amount and avg closing price per year")
ax2.legend(lines + lines2, labels + labels2, loc='upper left')

plt.show()
print("---------------------------------------------------------------------")
x1 = project_file['year']
y1 = project_file['avg_closing_price']

fig, ax1 = plt.subplots()

color = 'tab:purple'
ax1.set_xlabel('Year')
ax1.set_ylabel('Average Closing Price', color=color)
ax1.plot(x1, y1, label="avg_closing_price_per_year", color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  

x2 = project_file['year']
y2 = project_file['annual_perc_change']

color = 'tab:orange'
ax2.set_ylabel('Annual Percentage Change', color=color)  
ax2.plot(x2, y2, label="annual_percentage_change_per_year", color=color)
ax2.tick_params(axis='y', labelcolor=color)


lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')
plt.title("relation between annual change and avg closing price per year")
plt.show()
print("---------------------------------------------------------------------")

fc_d_sum=project_file['factors_affect_price'].duplicated().sum()
an_d_sum=project_file['annual_perc_change'].duplicated().sum()
avg_d_sum=project_file['avg_closing_price'].duplicated().sum()
yrlow_d_sum=project_file['year_low'].duplicated().sum()
yrhigh_d_sum=project_file['year_high'].duplicated().sum()
open_d_sum=project_file['year_open'].duplicated().sum()
close_d_sum=project_file['year_close'].duplicated().sum()
pr_d_sum=project_file['produced_amount'].duplicated().sum()

print(f"duplicate values of factors_affect_price is {fc_d_sum} ")
print(f"duplicate values of annual_perc_change is {an_d_sum} ")
print(f"duplicate values of avg_closing_price is {avg_d_sum} ")
print(f"duplicate values of year_low is {yrlow_d_sum} ")
print(f"duplicate values of year_high is {yrhigh_d_sum} ")
print(f"duplicate values of year_open is {open_d_sum} ")
print(f"duplicate values of year_close is {close_d_sum} ")
print(f"duplicate values of produced_amount is {pr_d_sum} ")

print("---------------------------------------------------------------------")

plt.plot(project_file['year'],project_file['year_open'],label="year_open price per year ",color='r')
plt.plot(project_file['year'],project_file['year_close'],label="year_close price per year ",color='g')
plt.legend()
plt.title("relation between open and close price per year")
plt.show()

plt.plot(project_file['year'],project_file['year_high'],label="year_high_price per year ",color='b',marker='o',markersize=3)
plt.plot(project_file['year'],project_file['year_low'],label="year_low_price per year ",color='r',marker='o',markersize=3)
plt.legend()
plt.title("relation between low and high price per year")
plt.show()

label_encoder = LabelEncoder()

project_file['factors_affect_price'] = label_encoder.fit_transform(project_file['factors_affect_price'])

all_columns_correlation = project_file.corr()

print("\nAll Columns Correlation:")
print(all_columns_correlation)

csv_file_path = 'correlation_matrix.csv'
all_columns_correlation.to_csv(csv_file_path, index=False)

print(f"\nCorrelation matrix saved to: {csv_file_path}")

X = project_file[['produced_amount']]  
y = project_file['avg_closing_price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Regression
model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.xlabel('Produced Amount')
plt.ylabel('Average Closing Price')
plt.title('Linear Regression: Produced Amount vs. Average Closing Price')
plt.show()
print("---------------------------------------------------------------------")
#prediction
last_year = 2024

expected_avg_percentage_change = 4

years_to_predict = [2025,2030]
predictions = []

additional_factor_2025 = 8
additional_factor_2027 = 12

for year in years_to_predict:
    
    predicted_closing_price = project_file[project_file['year'] == last_year]['year_close'].values[0] * (1 + expected_avg_percentage_change / 100)

    additional_factor = additional_factor_2025 if year == 2025 else additional_factor_2027

    actual_percentage_change = expected_avg_percentage_change + additional_factor

    predictions.append({'year': year, 'predicted_closing_price': predicted_closing_price, 'actual_percentage_change': actual_percentage_change})

for prediction in predictions:
    print(f'Year: {prediction["year"]}, annual Percentage Change: {prediction["actual_percentage_change"]:.2f}%')
