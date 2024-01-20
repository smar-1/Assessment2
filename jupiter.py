import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error


class Moons:
    def __init__(self):
        self.estimated_mass_jupiter = None
        self.model_y = None
        self.y_test = None
        self.y_train = None
        self.x_test = None
        self.x_train = None
        database_service = "sqlite"
        database = "jupiter.db"
        connectable = f"{database_service}:///{database}"
        query = "SELECT * FROM Moons"
        data = pd.read_sql(query, connectable)

        self.data = data

    def extract_row(self, moon_name):
        return self.data.loc[self.data["moon"] == moon_name]

    def extract_col(self, col_name):
        return self.data[col_name]

    def correlation(self, full, x_var, y_var):
        if full:
            return self.data.corr()
        elif not full:
            return self.data[x_var].corr(self.data[y_var])

    def plot(self, x_var, y_var):
        sns.relplot(data=self.data, x=x_var, y=y_var)
        pass

    def max_col(self, col):
        return self.data[col].max()

    def min_col(self, col):
        return self.data[col].min()

    def mean_col(self, col):
        return self.data[col].mean()

    def sd_col(self, col):
        return self.data[col].std()

    def full_summary(self):
        return self.data.describe()

    def information(self):
        self.data.info()
        pass

    def show(self):
        print(self.data)
        pass

    def training(self):
        self.data['T_sq'] = (self.data['period_days'] * 86400) ** 2
        self.data['a_cu'] = (self.data['distance_km'] * 1000) ** 3

        X = self.data[['T_sq']]
        Y = self.data['a_cu']

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=0.3)

        self.model_y = linear_model.LinearRegression(fit_intercept=True)
        self.model_y.fit(self.x_train, self.y_train)
        return self.model_y

    def testing(self):
        test_predict_y = self.model_y.predict(self.x_test)
        return r2_score(self.y_test, test_predict_y)

    def predict(self):
        G = 6.67e-11
        self.estimated_mass_jupiter = (4 * np.pi ** 2 / G) * self.model_y.coef_[0]
        return self.estimated_mass_jupiter
