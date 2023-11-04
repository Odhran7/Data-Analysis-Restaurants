import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_pipeline_imblearn
from joblib import dump

data = pd.read_csv('merged_with_sentiment.csv')
X = data.drop('InMichelin', axis=1)
y = data['InMichelin']
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
smote = SMOTE(random_state=42)
pipeline_rf_smote = make_pipeline_imblearn(StandardScaler(), smote, RandomForestClassifier(random_state=42))
param_grid = {
    'randomforestclassifier__n_estimators': [100, 200, 300],
    'randomforestclassifier__max_depth': [None, 10, 20, 30],
    'randomforestclassifier__min_samples_split': [2, 5, 10],
    'randomforestclassifier__min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(pipeline_rf_smote, param_grid, cv=5, scoring='recall', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_recall = grid_search.best_score_
best_model = grid_search.best_estimator_
y_pred_best = grid_search.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
class_report_best = classification_report(y_test, y_pred_best)
dump(grid_search.best_estimator_, 'best_rf_pipeline_with_smote.joblib')
print(f"Best parameters: {best_params}")
print(f"Best recall from cross-validation: {best_recall:.2f}")
print(f"Accuracy with the best Random Forest: {accuracy_best:.2f}")
print(class_report_best)

rf_model = best_model.named_steps['randomforestclassifier']
importances = rf_model.feature_importances_
features = X_train.columns
feature_importance = pd.Series(importances, index=features)

# Print the feature importances
print("Feature importances:")
print(feature_importance)




"""
Best parameters: {'randomforestclassifier__max_depth': None, 'randomforestclassifier__min_samples_leaf': 2, 'randomforestclassifier__min_samples_split': 2, 'randomforestclassifier__n_estimators': 300}
Best recall from cross-validation: 0.93
Accuracy with the best Random Forest: 0.81
              precision    recall  f1-score   support

           0       0.82      0.58      0.68        24
           1       0.80      0.93      0.86        44

    accuracy                           0.81        68
   macro avg       0.81      0.76      0.77        68
weighted avg       0.81      0.81      0.80        68

Feature importances:
food                      0.201062
service                   0.121230
price                     0.108168
decor                     0.167856
number_of_reviews         0.057126
excellent                 0.039888
very_good                 0.071395
average                   0.037378
poor                      0.035090
terrible                  0.038694
sentiment_polarity        0.073760
sentiment_subjectivity    0.048353
"""