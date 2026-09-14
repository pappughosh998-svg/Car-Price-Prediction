# Used Car Price Prediction
A Machine Learning project that predicts the price of used cars using features such as car name, manufacturing year, mileage, color, and condition.

## Project Overview
This project uses regression algorithms to estimate the selling price of a used car.
The dataset contains information about different used cars, and the model learns the relationship between car features and their prices.

## Technologies Used
* Pandas
* Scikit-learn
## Dataset Features
The dataset contains the following columns:
* `name` — Name of the car
* `year` — Manufacturing year
* `miles` — Distance travelled by the car
* `color` — Car color
* `condition` — Condition of the car
* `price` — Selling price of the car

## Machine Learning Algorithm
The following regression algorithm was used:
* Extra Trees Regressor

## Project Workflow
1. Load the dataset using Pandas
2. Explore the dataset
3. Clean the `miles` and `price` columns
4. Select relevant features
5. Convert categorical features into numerical values using One-Hot Encoding
6. Split the dataset into training and testing data
7. Train the Extra Trees Regressor model
8. Predict car prices
9. Evaluate the model using MAE and R² Score

## Model Evaluation
The model achieved approximately:
* **R² Score:** 0.76
The model performance may vary depending on the dataset split and training process.

## Evaluation Metrics
### Mean Absolute Error — MAE
MAE measures the average difference between the actual car prices and the predicted car prices.

### R² Score
R² Score measures how well the model explains the variation in car prices.

A higher R² Score generally indicates better model performance.

## Author

**Pappu Ghosh**

B.Tech Computer Science and Engineering Student
