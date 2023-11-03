from scrape_reviews import scrape_reviews
import pandas as pd

df = pd.read_csv('../output/v2/merged.csv')
output_file = '../output/v2/review.csv'
pages = 3
index = df[df['Original Restaurant Name'] == 'Vong'].index[0]
urls = df['URL']
names = df['Original Restaurant Name']
iterList = list(zip(names[index:], urls[index:]))

def worker(args):
    name, url = args
    if isinstance(url, str) and url != '0':
        scrape_reviews(name, url, output_file, pages)

if __name__ == "__main__":
    for args in iterList:
        worker(args)
