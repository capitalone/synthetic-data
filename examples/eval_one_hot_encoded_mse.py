import sys
import dataprofiler as dp
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import seaborn as sns
# import re
# from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
sys.path.append('..')
from synthetic_data.synthetic_data import make_data_from_report

#%%

dp.__version__

#%% md

### Inject nulls into the feature "petal length (cm)" based on values for sepal length and sepal width

#%%

# data = dp.Data("./sample_datasets/diamonds.csv")
data = dp.Data("./examples/sample_datasets/diamonds.csv")
data.head()

#%%

### Create profile of (iris + null)

#%%

profile_options = dp.ProfilerOptions()
profile_options.set({
    "data_labeler.is_enabled": False,
    "correlation.is_enabled": False,
    "structured_options.multiprocess.is_enabled": False,
    "null_replication_metrics.is_enabled": False
})
profile = dp.Profiler(data, options=profile_options)
report = profile.report()
#report

#%%

report['data_stats'][2]

#%%

from sklearn import preprocessing

# encoding_type = 'label'
encoding_type = 'one-hot'

text_cat_name_list = []
for col_stat in report["data_stats"]:
    if col_stat["categorical"] and col_stat["data_type"] not in ["int", "float"]:
        col_name = col_stat["column_name"]
        text_cat_name_list.append(col_name)
        if encoding_type == 'label':
            encoder = preprocessing.LabelEncoder()
            # encoder.fit(data.data[col_name])
            data.data[col_name + "_encoded"] = encoder.fit_transform(
                data.data[col_name])
        elif encoding_type == 'one-hot':
            one_hot_data= pd.get_dummies(data.data[col_name])
            one_hot_data.columns = col_name + '_' + one_hot_data.columns
            data._data = pd.concat([data.data, one_hot_data], axis=1)
        else:
            raise ValueError("Can't handle encoder")

df_fixed = data.drop(*[text_cat_name_list], axis=1)

#%%

df_fixed.head()

#%%

profile_options = dp.ProfilerOptions()
profile_options.set({
    "data_labeler.is_enabled": False,
    "correlation.is_enabled": True,
    "structured_options.multiprocess.is_enabled": False,
    "null_replication_metrics.is_enabled": True
})
profile = dp.Profiler(df_fixed, options=profile_options, samples_per_update=len(df_fixed))
report_encoded = profile.report()


#%%

# sns.heatmap(report_encoded["global_stats"]["correlation_matrix"])

#%%



#%% md

### Generate synthetic data for (iris + null)

#%%

synthetic_data = make_data_from_report(report_encoded)
#synthetic_data

#%% md

# data["petal length (cm)"].values.T

#%%

# synthetic_data["petal length (cm)"].values.T

#%%

# sys.exit()

#%%

# I think this is here as debugging - to validate that the null properties of the report from synthetic data
# match the original report
synthetic_data_profile = dp.Profiler(synthetic_data, options=profile_options, samples_per_update=len(synthetic_data))
synthetic_data_report = synthetic_data_profile.report()
synthetic_data_report

#%%

col_id = 8
col_name = synthetic_data_report['data_stats'][col_id]["column_name"]
fig, ax = plt.subplots()
orig_counts = report_encoded["data_stats"][-1]["statistics"]["categorical_count"]
synth_counts = synthetic_data_report["data_stats"][-1]["statistics"]["categorical_count"]
orig_counts = {float(k): v for k, v in orig_counts.items()}
synth_counts = {float(k): v for k, v in synth_counts.items()}
ax.bar( orig_counts.keys(), orig_counts.values(), alpha=0.5, label='orig')
ax.bar( synth_counts.keys(), synth_counts.values(), alpha=0.5, label='synth')
ax.set_title(col_name)
ax.legend()

# report_encoded['data_stats'][2]['null_replication_metrics']

#%%

# synthetic_data_report['data_stats'][2]['null_replication_metrics']

#%% md

#%%

X = df_fixed.drop(columns="price").astype(float)
y = data.data['price'].astype(float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# model = RandomForestRegressor()
model = clf = MLPRegressor(solver='lbfgs', alpha=1e-5,
                            hidden_layer_sizes=(15,), random_state=1)
clf = model.fit(X_train, y_train)
original_mse = np.average((y_test - clf.predict(X_test))) ** 2
print(f"MSE of RandomForestRegressor on predicting price in original dataset: {original_mse}")

X_synthetic = synthetic_data.drop(columns="price").astype(float)
y_synthetic = synthetic_data['price'].astype(float)
synthetic_mse = np.average((y_synthetic - clf.predict(X_synthetic))) ** 2
print(f"MSE of RandomForestRegressor on predicting price in synthetic dataset: {synthetic_mse}")

print()
print()
print()

X_train, X_test, y_train, y_test = train_test_split(X_synthetic, y_synthetic, test_size=0.1, random_state=42)
model = clf = MLPRegressor(solver='lbfgs', alpha=1e-5,
                            hidden_layer_sizes=(15,), random_state=1)
clf = model.fit(X_train, y_train)
synthetic_mse = np.average((y_test - clf.predict(X_test))) ** 2
print(f"MSE of RandomForestRegressor on predicting price in synthetic dataset: {synthetic_mse}")


original_mse = np.average((y - clf.predict(X))) ** 2
print(f"MSE of RandomForestRegressor on predicting price in original dataset: {original_mse}")
