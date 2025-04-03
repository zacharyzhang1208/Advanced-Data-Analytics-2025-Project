# PART I

## Linear Regression vs. **Random Forest Regressor**

## Installation

To run this analysis, please ensure you have the necessary Python libraries installed. You can install the required packages using the following command:

```
pip install pandas numpy seaborn matplotlib plotly scikit-learn
```

Ensure that you also have Jupyter Notebook or JupyterLab to run the code interactively.

## Data Loading and Preprocessing

## Dataset

The dataset should in CSV format named `life_expectancy_dataset.csv` with the following columns:

- `country_code`
- `country_name`
- `year`
- `region`
- `sub-region`
- `intermediate-region`
- `life_expectancy_women`
- `life_expectancy_men`

1. **Loading the Data:**
   - The dataset `life_expectancy_dataset.csv` is loaded using `pandas`.
   - We handle missing values by filling them with “Unknown” in the `intermediate-region` column.
2. **Preprocessing:**
   - We removed records where life expectancy is zero, assuming these are erroneous entries.
   - Categorical variables are encoded using `LabelEncoder` from `scikit-learn`.

## Data Visualization

We use `Plotly` and `Seaborn` for data visualization. Below are some of the visualizations created:

1. **Histograms:**
   - Distribution of life expectancy for women and men.
   
   ![image-20250403162015133](C:\Users\I\AppData\Roaming\Typora\typora-user-images\image-20250403162015133.png)
   
2. **Boxplots:**
   
   - Boxplots representing spreads of life expectancy for each gender.
   
     ![image-20250403162050440](C:\Users\I\AppData\Roaming\Typora\typora-user-images\image-20250403162050440.png)
   
3. **Line Charts:**
   
   - Line charts representing the trend of life expectancy over years and by regions.
   
   ![image-20250403162112935](C:\Users\I\AppData\Roaming\Typora\typora-user-images\image-20250403162112935.png)
   
   ![image-20250403162134344](C:\Users\I\AppData\Roaming\Typora\typora-user-images\image-20250403162134344.png)

## Predictive Modeling

We employ two machine learning models for predictive analysis:

1. **Linear Regression**
2. **Random Forest Regressor**

We used `train_test_split` from `scikit-learn` to split the data into training and testing sets.

## Model Evaluation

​	The approximate performance of `Linear Regression` and `Random Forests` in predicting female life expectancy can be compared by drawing scatter plots of actual and model-predicted values, as well as by comparing the deviations from each model's predictions.

​	By observing the scatter plot:

​	Prediction accuracy: If the scatter points are closer to the red dashed line, it indicates that the model's predictions are more accurate.

​	Distribution status: The concentration and distribution of points reflect the degree of deviation between predicted values and actual values.

​	Error range: The random forest model may form a denser distribution around more data points, closer to the perfect prediction line, and may visually perform better than linear regression.

​	We can infer that the analyzed metrics, such as the average vertical distance to the perfect line, should be smaller for the random forest model than for linear regression, thus indicating its superior performance.

![image-20250403175703810](C:\Users\I\AppData\Roaming\Typora\typora-user-images\image-20250403175703810.png)

![image-20250403175722271](C:\Users\I\AppData\Roaming\Typora\typora-user-images\image-20250403175722271.png)

![image-20250403175735047](C:\Users\I\AppData\Roaming\Typora\typora-user-images\image-20250403175735047.png)

![image-20250403175745640](C:\Users\I\AppData\Roaming\Typora\typora-user-images\image-20250403175745640.png)

​	To determine which model actually performs better, statistical metrics are needed to further quantify the differences in model performance. 

​	Evaluation metrics used include Mean Absolute Error (MAE), Mean Squared Error (MSE), Mean Absolute Percentage Error (MAPE) and R-squared.

![image-20250403165719805](C:\Users\I\AppData\Roaming\Typora\typora-user-images\image-20250403165719805.png)

## Summary

​	From the results of the model assessment, the Random Forest model was significantly better than the Linear Regression model in all indicators. Specifically, the Random Forest model has a mean absolute error (MAE) of 0.529451 for Women and 0.565260 for men, compared to 6.409589 and 6.086330 for the Linear Regression model, respectively. At the same time, the Random Forest model also outperforms the Linear Regression model in terms of the coefficient of determination (R2), the mean squared error (MSE), and the mean absolute percentage error (MAPE). Compared to the Random Forest model the R2 reached 0.991176 for Women and 0.986648 for men, compared to only 0.546828 and 0.526948 for the Linear Regression model.

​	In this prediction, the Random Forest model outperformed the linear regression model in the global life expectancy forecast. The random forest model is particularly adept at handling complex data sets consisting of multiple variables and non-linear relationships. This ability allows it to identify complex patterns in life expectancy more accurately than Linear Regression models. Random Forest reduces the risk of overfitting by averaging over multiple decision trees due to its integrated learning approach.



# PART II

## RF_Prediction 

This part aims to predict life expectancy for men and women globally, using a Random Forest Regressor. It utilizes historical data on life expectancy, covering various countries and regions worldwide, and makes predictions into the future (2023-2032).

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.6 or higher is installed on your system.
- You have installed the required Python packages: pandas, scikit-learn, numpy, and joblib.
- You need the seaborn and matplotlib packages for visualization.

You can install the required packages using:

```
pip install pandas scikit-learn numpy joblib seaborn matplotlib 
```

## Installation

1. Clone the repository to your local computer.
2. Ensure the `life_expectancy_dataset.csv` is in the same directory as your script.

## Execution

To run the life expectancy prediction and visualization code, follow these steps:

1. **Model Training**

   The code reads the dataset, encodes categorical variables,and then splits the data into training and test sets, trains a `RandomForest Regressor` model for both men and women, and saves these models using joblib.

   ![image-20250403165416568](C:\Users\I\AppData\Roaming\Typora\typora-user-images\image-20250403165416568.png)

2. **Future Prediction**

   After training, the code predicts life expectancy for the years 2023 through 2032. It uses unique combinations of country and regions to forecast future life expectancies, then saves these predictions to a CSV file `future_predictions.csv`.

3. **Visualization**

   ![image-20250403173646508](C:\Users\I\AppData\Roaming\Typora\typora-user-images\image-20250403173646508.png)
   
   ![image-20250403173658187](C:\Users\I\AppData\Roaming\Typora\typora-user-images\image-20250403173658187.png)
   
   ![image-20250403173711642](C:\Users\I\AppData\Roaming\Typora\typora-user-images\image-20250403173711642.png)
   
   ![image-20250403173723879](C:\Users\I\AppData\Roaming\Typora\typora-user-images\image-20250403173723879.png)

## Files

- `life_expectancy_dataset.csv` - Input dataset.
- `model_women_rf.joblib` - Trained model for predicting women’s life expectancy.
- `model_men_rf.joblib` - Trained model for predicting men’s life expectancy.
- `future_predictions.csv` - CSV file containing future life expectancy predictions.
- `Real-Predicted.csv` - A combined dataset of real and predicted life expectancies.
