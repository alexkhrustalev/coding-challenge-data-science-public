# import os

# from src.data_handler import load_sql_file_to_dataframe

# df = load_sql_file_to_dataframe(os.path.join("data","tickets.db"))

# print(df.head())

from forecast import forecast

if __name__ == "__main__":
    forecast()
    