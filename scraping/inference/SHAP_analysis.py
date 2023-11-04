import pandas as pd
import xgboost
import shap
from joblib import load

test_values = [45, 45, 40, 45, 800, 481, 162, 74, 22, 23, 0.32935320530390216, 0.6187026747192107]
column_names = ["food","service","price","decor","number_of_reviews","excellent","very_good","average","poor","terrible","sentiment_polarity","sentiment_subjectivity"]  
test_values_df = pd.DataFrame([test_values], columns=column_names)
model = load('../../models/best_xgb_pipeline_with_smote.joblib')
explainer = shap.TreeExplainer(model.named_steps['xgbclassifier'])
shap_values = explainer.shap_values(test_values_df)
force_plot = shap.force_plot(explainer.expected_value, shap_values[0], test_values_df.iloc[0])

shap.summary_plot(shap_values, test_values_df, plot_type="dot")
shap.summary_plot(shap_values, test_values_df, plot_type="bar")
shap.decision_plot(explainer.expected_value, shap_values, test_values_df)
shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0], test_values_df.iloc[0])
shap.save_html('shap_beeswarm_plot.html', shap.summary_plot(shap_values, test_values_df, plot_type="dot", show=False))
shap.save_html('shap_bar_plot.html', shap.summary_plot(shap_values, test_values_df, plot_type="bar", show=False))
shap.save_html('shap_decision_plot.html', shap.decision_plot(explainer.expected_value, shap_values, test_values_df, show=False))
shap.save_html('shap_waterfall_plot.html', shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0], test_values_df.iloc[0], show=False))
shap.save_html('shap_force_plot.html', force_plot) 
