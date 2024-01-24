import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt


class Moons:
    def __init__(self):
        # Initialize attributes for results and data
        self.test_predict_y = None
        self.estimated_mass_jupiter = None
        self.model_y = None
        self.y_test = None
        self.y_train = None
        self.x_test = None
        self.x_train = None
        self.r2 = None
        self.mse = None

        # Connect to the database and retrieve data from the Moons table
        database_service = "sqlite"
        database = "jupiter.db"
        connectable = f"{database_service}:///{database}"
        query = "SELECT * FROM Moons"
        data = pd.read_sql(query, connectable)
        self.data = data

    def extract_row(self, moon_name):
        # Extract a specific row based on the moons name
        return self.data.loc[self.data["moon"] == moon_name]

    def extract_col(self, col_name):
        # Extract a specific column from the dataset
        return self.data[col_name]

    def correlation(self, full, x_var=None, y_var=None):
        # Calculate correlation either for the entire dataset or between specific columns
        if full:
            return self.data.corr()
        elif not full:
            return self.data[x_var].corr(self.data[y_var])

    def plot(self, x_var, y_var):
        # Plot a scatter plot between two variables
        sns.relplot(data=self.data, x=x_var, y=y_var)
        pass

    def max_col(self, col):
        # Get the maximum value in a specific column
        return self.data[col].max()

    def min_col(self, col):
        # Get the minimum value in a specific column
        return self.data[col].min()

    def mean_col(self, col):
        # Get the mean value of a specific column
        return self.data[col].mean()

    def sd_col(self, col):
        # Get the standard deviation of a specific column
        return self.data[col].std()

    def full_summary(self):
        # Generate a summary of the entire dataset
        return self.data.describe()

    def information(self):
        # Display information about the dataset
        self.data.info()
        pass

    def show(self):
        # Display the entire dataset
        print(self.data)
        pass

    def peek(self, rows):
        # Display the first rows rows of the dataset
        return self.data.head(rows)

    def training(self):
        # Perform training by transforming the variables
        self.data['T_sq'] = (self.data['period_days'] * 86400) ** 2
        self.data['a_cu'] = (self.data['distance_km'] * 1000) ** 3

        X = self.data[['T_sq']]
        Y = self.data['a_cu']

        # Split data into training and testing
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=0.3)

        # Create model and fit it
        self.model_y = linear_model.LinearRegression(fit_intercept=True)
        self.model_y.fit(self.x_train, self.y_train)
        return self.model_y

    def testing(self):
        # Perform testing by making predictions and evaluating model performance
        self.test_predict_y = self.model_y.predict(self.x_test)
        self.mse = mean_squared_error(self.y_test, self.test_predict_y)
        self.r2 = r2_score(self.y_test, self.test_predict_y)
        return self.r2, self.mse

    def predict(self):
        # Calculate the estimated mass of Jupiter based on the linear regression model gradient
        G = 6.67e-11
        self.estimated_mass_jupiter = (4 * np.pi ** 2 / G) * self.model_y.coef_[0]
        return self.estimated_mass_jupiter

    def evaluation(self):
        # Visualize model evaluation with scatter plot and histogram of residuals
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.scatter(self.y_test, self.test_predict_y)
        plt.title('Actual vs Predicted')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')

        plt.subplot(1, 2, 2)
        residuals = self.y_test - self.test_predict_y
        sns.histplot(residuals, kde=True)
        plt.title('Residuals Frequency')
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.show()
        pass
