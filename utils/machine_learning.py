import pandas as pd

data = pd.read_csv('../output/v2/merged_with_sentiment.csv')

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score

data_cleaned = pd.read_csv('../output/v2/merged_with_sentiment.csv')
# data_cleaned = data.drop(['name', 'address', 'url', 'restaurant_url'], axis=1)
imputer = SimpleImputer(strategy='median')
data_cleaned[['decor', 'sentiment_polarity', 'sentiment_subjectivity']] = imputer.fit_transform(
    data_cleaned[['decor', 'sentiment_polarity', 'sentiment_subjectivity']])
X = data_cleaned.drop('InMichelin', axis=1)
y = data_cleaned['InMichelin']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline = make_pipeline(StandardScaler(), LogisticRegression())
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
