import pandas as pd
import numpy as np
import seaborn as sns


class Moons:
    def __init__(self):
        database_service = "sqlite"
        database = "jupiter.db"
        connectable = f"{database_service}:///{database}"
        query = "SELECT * FROM Moons"
        data = pd.read_sql(query, connectable)

        self.data = data

    def extract(self, moon_name):
        return self.data.loc[self.data["moon"] == moon_name]

    def correlation(self, x_var, y_var):
        return self.data[x_var].corr(self.data[y_var])

    def plot(self, x_var, y_var):
        sns.relplot(self.data, x=x_var, y=y_var)
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
