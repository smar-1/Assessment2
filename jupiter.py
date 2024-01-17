import pandas as pd
import numpy as np
class Moons:
    def __init__(self):

        database_service = "sqlite"
        database = "jupiter.db"
        connectable = f"{database_service}:///{database}"
        query = "SELECT * FROM Moons"
        data = pd.read_sql(query, connectable)

        self.data = data

        def extract(self, moon_name):
            return self.data.loc[data["moon"] == moon_name]



