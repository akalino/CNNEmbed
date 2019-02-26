import os
import pandas as pd


if __name__ == "__main__":
    files = os.listdir('source')
    for f in files:
        label = f.split('_')[-1].split('.')[0]
        fn = label + '.txt'
        df = pd.read_csv('source/' + f)
        url_list = df['url'].tolist()
        with open(fn, 'w') as f:
            f.writelines(["%s\n" % item for item in url_list])
