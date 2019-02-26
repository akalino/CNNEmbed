import pandas as pd


if __name__ == "__main__":
    df = pd.read_csv('january_insurance_urls.csv')
    for lab in list(set(df.classification)):
        url_list = df[df['classification'] == lab]['url'].tolist()
        fn = lab.split('_')[1] + '.txt'
        with open(fn, 'w') as f:
            f.writelines(["%s\n" % item for item in url_list])
