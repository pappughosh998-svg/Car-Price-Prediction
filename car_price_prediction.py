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
