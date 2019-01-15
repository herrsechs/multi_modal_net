import pandas as pd


def write_csv(data, header, path):
    if len(data) != len(header):
        print('Length of data and header don\'t match')
    df = pd.DataFrame(data, columns=header)
    df.to_csv(path, index=False)
