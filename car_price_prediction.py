import pandas as pd
import re
df = pd.read_csv('car_web_scraped_dataset.csv')
print(df.columns)
df.info()

df["miles"]=(
    df["miles"].str.replace("miles", "").str.replace(",", "").astype(float))
df["price"]=(
    df["price"].str.replace(",", "").str.replace("$", "").astype(float))

print(df[["miles", "price"]].head())

X = pd.get_dummies(df[["miles", "year", "condition", "color", "name"]], drop_first=True)
Y = df["price"]

print(X.head())
print(Y.head())

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

from sklearn.ensemble import ExtraTreesRegressor
model = ExtraTreesRegressor(n_estimators=300, random_state=42)
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
print(y_pred[:5])

from sklearn.metrics import mean_absolute_error, r2_score
mae = mean_absolute_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)

print("Mae:", mae)
print("R2 score:", r2)
print("\n--- Used Car Price Prediction ---")

name = input("Enter Car Name: ")
year = int(input("Enter Car Year: "))
miles = float(input("Enter Car Mileage: "))
color = input("Enter Car Color: ")
condition = input("Enter Car Condition: ")

# New car data
new_car = pd.DataFrame([{
    "miles": miles,
    "year": year,
    "condition": condition,
    "color": color,
    "name": name
}])

# Convert categorical features into numerical columns
new_car = pd.get_dummies(new_car, drop_first=True)

# Match new data columns with training data columns
new_car = new_car.reindex(columns=X.columns, fill_value=0)

# Predict car price
predicted_price = model.predict(new_car)

print(f"\nEstimated Car Price: ${predicted_price[0]:,.2f}")
