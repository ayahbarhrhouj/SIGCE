import dice_ml
from dice_ml import Dice
from dice_ml.utils import helpers 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler,PowerTransformer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import r2_score,mean_squared_error
from sklearn import metrics
import random
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


data = pd.read_csv('boston.csv')
outcome_name = "MEDV"
continuous_features = data.drop(outcome_name, axis=1).columns.tolist()
target = data[outcome_name]

# Split data into train and test
datasetX = data.drop(outcome_name, axis=1)
x_train, x_test, y_train, y_test = train_test_split(datasetX,
                                                    target,
                                                    test_size=0.3,
                                                    random_state=0)

categorical_features = x_train.columns.difference(continuous_features)

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

transformations = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, continuous_features),
        ('cat', categorical_transformer, categorical_features)])

xgboost = XGBRegressor(colsample_bytree=1, learning_rate=0.1, max_depth=4, min_child_weight=1, n_estimators=200, 
                        reg_alpha=0, reg_lambda=0.5, subsample=1, random_state=42)
regr_wind = Pipeline(steps=[('preprocessor', transformations),
                               ('regressor', xgboost)])
model_wind = regr_wind.fit(x_train, y_train)

d_wind = dice_ml.Data(dataframe=data, continuous_features=continuous_features, outcome_name=outcome_name)
m_wind = dice_ml.Model(model=model_wind, backend="sklearn", model_type='regressor')

exp_genetic = Dice(d_wind, m_wind, method="genetic")



def create_dynamic_permitted_range(instance, percentage=0.5):
    permitted_range = {}
    for col in instance.columns:
        val = instance[col].values[0]
        
        # Appliquer contraintes 
        lower = val * (1 - percentage)
        upper = val * (1 + percentage)
        
        permitted_range[col] = [lower, upper]
    
    return permitted_range


def evaluate_dice_on_all_test_instances(X_test, y_test, model, exp, percentage=0.5):
    results = []
    found = 0
    not_found = 0

    for i in range(len(X_test)):
        query_instance = X_test.iloc[i:i+1]
        current_target = y_test.iloc[i]

        permitted_range = create_dynamic_permitted_range(query_instance, percentage=percentage)

        try:
            cf = exp.generate_counterfactuals(
                query_instances=query_instance,
                total_CFs=2,
                desired_range=[25.0, 35.0],
                permitted_range=permitted_range
            )

            cf_df = cf.cf_examples_list[0].final_cfs_df

            if cf_df is not None and not cf_df.empty:
                original_pred = model.predict(query_instance)[0]
                cf_pred = model.predict(cf_df.iloc[[0]])[0]
                
                cf_values = cf_df[query_instance.columns].iloc[0].values.flatten()
                query_values = query_instance.iloc[0].values.flatten()
                
                sparsity = np.sum(cf_values != query_values)
                proximity = np.sum(np.abs(cf_values - query_values))            
                results.append({
                    'instance_index': i,
                    'sparsity': sparsity,
                    'proximity': proximity
                })
                found += 1
            else:
                not_found += 1
        
        except Exception as e:
            not_found += 1
            continue

    results_df = pd.DataFrame(results)
    
    summary = {
        'total_instances': len(X_test),
        'found_counterfactuals': found,
        'not_found_counterfactuals': not_found,
        'success_rate(%)': round(100 * found / len(X_test), 2)
    }

    return results_df, summary


import time
start = time.time()
dice_results, dice_summary = evaluate_dice_on_all_test_instances(x_test, y_test, regr_wind, exp_genetic)
end = time.time()
t = end-start
print("time : ", t, "sec")



print(dice_summary)


mean_sparsity = dice_results['sparsity'].mean()
mean_proximity = dice_results['proximity'].mean()

print(f"- Sparsity moyenne   : {mean_sparsity:.3f}")
print(f"- Proximity moyenne  : {mean_proximity:.3f}")








