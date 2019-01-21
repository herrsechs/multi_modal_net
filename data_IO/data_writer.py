import pandas as pd


def write_csv(data, header, path):
    df = pd.DataFrame(data, columns=header)
    df.to_csv(path, index=False)
