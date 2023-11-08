import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import pandas as pd

def generate_word_cloud_with_stopwords(text, title, stopwords, font_path=None):
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        stopwords=stopwords,
        min_font_size=10,
        font_path=font_path  
    ).generate(text)                    
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.title(title, fontsize=24)
    plt.show()
michelin_reviews_path = '../output/v2/michelinReviews.csv'
normal_reviews_path = '../output/v2/review.csv'
michelin_reviews = pd.read_csv(michelin_reviews_path)
normal_reviews = pd.read_csv(normal_reviews_path, on_bad_lines='skip')

michelin_text = " ".join(review for review in michelin_reviews.review)
normal_text = " ".join(review for review in normal_reviews.review)
extended_stopwords = set(STOPWORDS).union({
    'restaurant', 'food', 'place', 'one', 'service', 'good', 'great', 'like', 'go', 'get', 'really', 'also', 'back'
})

# font_path = 'Godshomedemo-K794e.ttf'

generate_word_cloud_with_stopwords(michelin_text, 'Michelin Reviews Word Cloud', extended_stopwords)
generate_word_cloud_with_stopwords(normal_text, 'Normal Reviews Word Cloud', extended_stopwords)
