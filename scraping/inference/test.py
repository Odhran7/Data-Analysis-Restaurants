from joblib import load
import numpy as np
import shap
import pandas as pd
from sklearn.impute import SimpleImputer

df = pd.read_csv('merged_with_sentiment.csv', encoding='ISO-8859-1')
df['excellent_to_terrible_ratio'] = df['excellent'] / (df['terrible'] + 1)
df['value'] = df['food'] / (df['price'] + 1)

X = df.drop('InMichelin', axis=1)
y = df['InMichelin']
imputer = SimpleImputer(strategy='median')
imputer.fit(X)
pipeline = load('../../models/best_xgb_pipeline_with_smote.joblib')
test_values_nm = np.array([45, 45, 40, 45, 4510, 2889, 510, 229, 125, 97, 0.47121718000552615, 0.6489165831112074, 45/40, 2889/97]).reshape(1, -1)
test_values_m = np.array([45, 45, 40, np.nan, 272, 186, 35, 24, 16, 2, 0.3094743777964221, 0.6027621192813654, 45/40, 186/2]).reshape(1, -1)
test_values_m_imputed = imputer.transform(test_values_m)
predicted_class_nm = pipeline.predict(test_values_nm)
predicted_class_m = pipeline.predict(test_values_m_imputed)

print(f"Predicted class for non-Michelin values: {predicted_class_nm[0]}")
print(f"Predicted class for Michelin values: {predicted_class_m[0]}")

explainer = shap.Explainer(pipeline.named_steps['xgbclassifier'])
preprocessed_nm = pipeline.named_steps['columntransformer'].transform(test_values_nm)
preprocessed_m = pipeline.named_steps['columntransformer'].transform(test_values_m_imputed)

shap_values_nm = explainer.shap_values(preprocessed_nm)
shap_values_m = explainer.shap_values(preprocessed_m)
feature_names = pipeline.named_steps['columntransformer'].get_feature_names_out()

shap.summary_plot(shap_values_nm, feature_names=feature_names)
shap.summary_plot(shap_values_m, feature_names=feature_names)
