from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImblearnPipeline
from sklearn.metrics import classification_report, accuracy_score
from joblib import dump
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

data = pd.read_csv('merged_with_sentiment.csv')
X = data.drop('InMichelin', axis=1)
y = data['InMichelin']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
imputer = SimpleImputer(strategy='median')
smote = SMOTE(random_state=42)
pipeline_smote = ImblearnPipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('smote', smote),
    ('logisticregression', LogisticRegression())
])

param_grid = {
    'logisticregression__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'logisticregression__penalty': ['l1', 'l2'],
    'logisticregression__solver': ['liblinear', 'saga'],
    'logisticregression__max_iter': [100, 200, 300],
    'logisticregression__class_weight': [None, 'balanced', {0: 1, 1: 10}, {0: 10, 1: 1}, {0: 5, 1: 1}, {0: 1, 1: 5}]
}

scorer = make_scorer(accuracy_score)
grid_search = GridSearchCV(pipeline_smote, param_grid, scoring=scorer, cv=5)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_score = grid_search.best_score_
best_estimator = grid_search.best_estimator_

y_pred_best = best_estimator.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
class_report_best = classification_report(y_test, y_pred_best)

print("Best Parameters:", best_params)
print("Best CV score:", best_score)
print("Test Accuracy with best estimator:", accuracy_best)
print("Classification Report on Test Data:\n", class_report_best)

coefficients = best_estimator.named_steps['logisticregression'].coef_[0]
features = X_train.columns
feature_importance = pd.Series(coefficients, index=features)
odds_ratios = np.exp(coefficients)
odds_series = pd.Series(odds_ratios, index=features)

print("Feature coefficients:\n", feature_importance)
print("Odds Ratios:\n", odds_series)
dump(best_estimator, 'best_logistic_model.joblib')
dump(imputer, 'imputer.joblib')

"""
Best Parameters: {'logisticregression__C': 10, 'logisticregression__class_weight': {0: 1, 1: 5}, 'logisticregression__max_iter': 100, 'logisticregression__penalty': 'l2', 'logisticregression__solver': 'liblinear'}
Best CV score: 0.8709090909090909
Test Accuracy with best estimator: 0.7941176470588235
Classification Report on Test Data:
               precision    recall  f1-score   support

           0       0.86      0.50      0.63        24
           1       0.78      0.95      0.86        44

    accuracy                           0.79        68
   macro avg       0.82      0.73      0.74        68
weighted avg       0.81      0.79      0.78        68

Feature coefficients:
 food                      3.122730
service                  -2.431013
price                     3.125602
decor                     0.757411
number_of_reviews         0.191465
excellent                 0.609696
very_good                -4.332165
average                   5.107123
poor                     -0.680944
terrible                 -1.619325
sentiment_polarity       -0.939182
sentiment_subjectivity   -0.123464
dtype: float64
Odds Ratios:
 food                       22.708290
service                     0.087948
price                      22.773609
decor                       2.132747
number_of_reviews           1.211022
excellent                   1.839872
very_good                   0.013139
average                   165.194383
poor                        0.506139
terrible                    0.198032
sentiment_polarity          0.390948
sentiment_subjectivity      0.883853
"""