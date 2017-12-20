import pandas  as pd

def get_data(file):
    df = pd.read_excel(file)
    print(df.head())

if __name__ == "__main__":
    get_data("./totalNPdata.xlsx")