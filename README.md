# Car Price Prediction — Linear Regression, Ridge, Lasso

Predicting car price using Linear Regression, then comparing it against Ridge and Lasso regularization on the CarPrice_Assignment dataset (205 cars, 26 original features).

## What's in this notebook

- Cleaned and encoded the data (dropped high-cardinality CarName and car_ID, one-hot encoded categorical features)
- Checked outliers on price and key predictors, kept them since they reflect real premium/performance cars
- Trained Linear Regression, Ridge, and Lasso models on an 80/20 train-test split
- Compared coefficients across all three models
- Evaluated all three using MAE, MSE, RMSE and R²

## Result

Linear Regression had the best test performance (R² ≈ 0.89), with Lasso close behind. Ridge lagged slightly. Regularization didn't improve much here since Linear Regression wasn't overfitting to begin with, but Lasso still simplified the model by zeroing out a few weak features.

## Files

- `CarPrice_LinearRegression_Ridge_Lasso.ipynb` - the notebook
- `CarPrice_Assignment.csv` - the dataset
