from joblib import load
import numpy as np
import shap
import pandas as pd 
from sklearn.impute import SimpleImputer


df = pd.read_csv('merged_with_sentiment.csv', encoding='ISO-8859-1')
X = df.drop('InMichelin', axis=1)
y = df['InMichelin']
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

pipeline = load('../../models/best_xgb_pipeline_with_smote.joblib')
test = [40, 40, 35, 40, 483, 210, 72, 66, 32, 27, 0.33051122981308556, 0.5633983268978834]
test_reshaped = np.array(test).reshape(1, -1)  # Reshape the data to have one sample

predicted_class_m = pipeline.predict(test_reshaped)
print(f"Predicted class: {predicted_class_m[0]}")

explainer = shap.Explainer(pipeline.named_steps['xgbclassifier'], X_imputed)
shap_values_m = explainer(test_reshaped)

# Summary plot for the given test values
shap.summary_plot(shap_values_m.values, X_imputed.columns, plot_type="bar")
