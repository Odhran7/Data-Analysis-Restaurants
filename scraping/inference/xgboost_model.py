import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_pipeline_imblearn
from joblib import dump

data = pd.read_csv('merged_with_sentiment.csv')
X = data.drop('InMichelin', axis=1)
y = data['InMichelin']
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
pipeline_xgb_smote = make_pipeline_imblearn(
    StandardScaler(),
    smote,
    XGBClassifier(random_state=42, use_label_encoder=False,
                  eval_metric='logloss')
)

param_grid = {
    'xgbclassifier__n_estimators': [100, 200, 300],
    'xgbclassifier__learning_rate': [0.01, 0.1, 0.3],
    'xgbclassifier__max_depth': [4, 6, 8],
    'xgbclassifier__min_child_weight': [1, 2, 3],
    'xgbclassifier__subsample': [0.8, 0.9, 1.0],
    'xgbclassifier__colsample_bytree': [0.8, 0.9, 1.0]
}
grid_search = GridSearchCV(
    pipeline_xgb_smote, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_score = grid_search.best_score_
best_estimator = grid_search.best_estimator_
y_pred_best = best_estimator.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
class_report_best = classification_report(y_test, y_pred_best)
dump(best_estimator, 'best_xgb_pipeline_with_smote.joblib')

# Print results
print(f"Best parameters: {best_params}")
print(f"Best CV score: {best_score:.2f}")
print(f"Test Accuracy with the best XGBoost: {accuracy_best:.2f}")
print(class_report_best)
feature_importances = best_estimator.named_steps['xgbclassifier'].feature_importances_
feature_importance_series = pd.Series(
    feature_importances, index=X_train.columns)
print("Feature importances:")
print(feature_importance_series.sort_values(ascending=False))
"""
Best parameters: {'xgbclassifier__colsample_bytree': 0.8, 'xgbclassifier__learning_rate': 0.1, 'xgbclassifier__max_depth': 6, 'xgbclassifier__min_child_weight': 3, 'xgbclassifier__n_estimators': 100, 'xgbclassifier__subsample': 1.0}
Best CV score: 0.88
Test Accuracy with the best XGBoost: 0.85
              precision    recall  f1-score   support

           0       0.85      0.71      0.77        24
           1       0.85      0.93      0.89        44

    accuracy                           0.85        68
   macro avg       0.85      0.82      0.83        68
weighted avg       0.85      0.85      0.85        68

Feature importances:
decor                     0.228611
food                      0.173511
service                   0.144522
price                     0.097031
sentiment_polarity        0.057289
very_good                 0.056997
poor                      0.048782
number_of_reviews         0.046622
terrible                  0.040920
sentiment_subjectivity    0.040538
excellent                 0.033260
average                   0.031916
"""