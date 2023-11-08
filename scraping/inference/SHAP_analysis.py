import pandas as pd
import xgboost
import shap
from joblib import load
import matplotlib.pyplot as plt
import numpy as np

# test_values = [45, 45, 40, 45, 800, 481, 162, 74, 22, 23, 0.32935320530390216, 0.6187026747192107]
# test_values = [45, 45, 45, 45, 863, 630, 150, 50, 20, 13, 0.31540499004884764, 0.6048444836111179]
# test_values_m = [45, 45, 40, None, 272, 186, 35, 24, 16, 2, 0.3094743777964221, 0.6027621192813654]
# test_values = [45, 45, 40, 45, 4510, 2889, 510, 229, 125, 97, 0.47121718000552615, 0.6489165831112074]
# column_names = ["food","service","price","decor","number_of_reviews","excellent","very_good","average","poor","terrible","sentiment_polarity","sentiment_subjectivity"]  
# test_values_df = pd.DataFrame([test_values], columns=column_names)
model = load('../../models/best_xgb_pipeline_with_smote.joblib')
explainer = shap.TreeExplainer(model.named_steps['xgbclassifier'])
# shap_values = explainer.shap_values(test_values_df)
# force_plot = shap.force_plot(explainer.expected_value, shap_values[0], test_values_df.iloc[0])

# shap.summary_plot(shap_values, test_values_df, plot_type="dot")
# shap.summary_plot(shap_values, test_values_df, plot_type="bar")
# shap.decision_plot(explainer.expected_value, shap_values, test_values_df)
# shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0], test_values_df.iloc[0])
# shap.save_html('shap_beeswarm_plot.html', shap.summary_plot(shap_values, test_values_df, plot_type="dot", show=False))
# shap.save_html('shap_bar_plot.html', shap.summary_plot(shap_values, test_values_df, plot_type="bar", show=False))
# shap.save_html('shap_decision_plot.html', shap.decision_plot(explainer.expected_value, shap_values, test_values_df, show=False))
# shap.save_html('shap_waterfall_plot.html', shap.plots._waterfall.waterfall_legacy(explainer.expected_value, shap_values[0], test_values_df.iloc[0], show=False))
# shap.save_html('shap_force_plot.html', force_plot) 

df = pd.read_csv('merged_with_sentiment.csv')
michelin_df = df[df['InMichelin'] == 1]
non_michelin_df = df[df['InMichelin'] == 0]
features = ["food", "service", "price", "decor", "number_of_reviews", "excellent", "very_good", "average", "poor", "terrible", "sentiment_polarity", "sentiment_subjectivity"]

X_michelin = michelin_df[features]
X_non_michelin = non_michelin_df[features]
shap_values_michelin = explainer.shap_values(X_michelin)
shap_values_non_michelin = explainer.shap_values(X_non_michelin)
shap.summary_plot(shap_values_michelin, X_michelin, show=True)
# shap.plots.beeswarm(explainer, show=True)
# shap.save_html('shap_summary_michelin.html', shap.summary_plot(shap_values_michelin, X_michelin, plot_type="dot", show=False))
shap.summary_plot(shap_values_non_michelin, X_non_michelin, show=True)
# shap.save_html('shap_summary_non_michelin.html', shap.summary_plot(shap_values_non_michelin, X_non_michelin, plot_type="dot", show=False))
explainer_michelin = shap.Explanation(values=shap_values_michelin,
                                      base_values=explainer.expected_value,
                                      data=X_michelin.values,  
                                      feature_names=features)

explainer_non_michelin = shap.Explanation(values=shap_values_non_michelin,
                                          base_values=explainer.expected_value,
                                          data=X_non_michelin.values,  
                                          feature_names=features)
predictions = model.predict(df[features])

df_michelin_inclusion = df[features][predictions == 1] 
df_michelin_non_inclusion = df[features][predictions == 0]  
shap_values_inclusion = explainer.shap_values(df_michelin_inclusion)
shap_values_non_inclusion = explainer.shap_values(df_michelin_non_inclusion)

mean_shap_values_inclusion = np.abs(shap_values_inclusion).mean(axis=0)
mean_shap_values_non_inclusion = np.abs(shap_values_non_inclusion).mean(axis=0)

plt.figure(figsize=(10, 8))
y_pos = np.arange(len(features))
plt.barh(y_pos, mean_shap_values_inclusion, align='center', color='green')
plt.yticks(y_pos, features)
plt.xlabel('Mean |SHAP value| (average impact on model output magnitude)')
plt.title('Mean Absolute SHAP Values for Michelin Inclusion')
plt.show()
plt.figure(figsize=(10, 8))
y_pos = np.arange(len(features))
plt.barh(y_pos, mean_shap_values_non_inclusion, align='center', color='red')
plt.yticks(y_pos, features)
plt.xlabel('Mean |SHAP value| (average impact on model output magnitude)')
plt.title('Mean Absolute SHAP Values for Michelin Non-Inclusion')
plt.show()