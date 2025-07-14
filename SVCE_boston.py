from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import shap
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
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




def svce(f, x_test, y_test, feature_names, gamma_min, gamma_max, max_iter=1000):
    preprocessor = f.named_steps["preprocessor"]
    model = f.named_steps["regressor"]
    results = []
    found = 0
    not_found = 0
    y_target_min = gamma_min
    y_target_max = gamma_max
    
    for i in range(len(x_test)):
        x = x_test.iloc[i:i+1]
        x_transformed = preprocessor.transform(x).flatten()
        
        explainer = shap.Explainer(model)
        shap_values = explainer.shap_values(x_transformed.reshape(1, -1))
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        phi = explainer.shap_values(x.values.reshape(1, -1))
        if isinstance(phi, list):
            phi = phi[0]
        phi_abs = np.abs(phi)
        phi_all = [(i, phi_abs[0, i]) for i in range(len(feature_names))]
        phi_all.sort(key=lambda x: -x[1])
        
        y_pred = model.predict(x.values.reshape(1, -1))[0]
        

        feature_std = np.std(x, axis=0)
        feature_mean = np.mean(x, axis=0)
        for m in range(1, len(feature_names) + 1):
            selected_indices = [i for i, _ in phi_all[:m]]
            x_prime = x_transformed.copy()
            iteration = 0
            success = False
        
            while iteration < max_iter:
                for idx in selected_indices:
                    noise = np.random.uniform(-0.1, 0.1) * feature_std[idx]
                    x_prime[idx] += noise
                    x_prime[idx] = np.clip(x_prime[idx], feature_mean[idx] - 3 * feature_std[idx], feature_mean[idx] + 3 * feature_std[idx])
        
                y_pred_prime = model.predict(x_prime.reshape(1, -1))[0]
                
                if y_target_min <= y_pred_prime <= y_target_max:
                    success = True
                    break
                    
                iteration += 1
            
            if success==True:
                break

        x_cf=x_prime
        y_cf=y_pred_prime
        y_original=y_pred
    
        if y_target_min <= y_pred_prime <= y_target_max:
            success = True
        if success:
            sparsity = np.sum(x_transformed  != x_cf)
            proximity = np.abs(x_transformed  - x_cf).sum()
            results.append({
                'instance_index': i,
                'sparsity': sparsity,
                'proximity': proximity
            })
            found += 1
        else:
            not_found += 1

    results_df = pd.DataFrame(results)

    summary = {
        'total_instances': len(x_test),
        'found_counterfactuals': found,
        'not_found_counterfactuals': not_found,
        'success_rate(%)': round(100 * found / len(x_test), 2)
    }

    return results_df, summary




import time
start = time.time()
svce_results, svce_summary = svce(
    f=regr_wind,
    x_test=x_test,
    y_test=y_test,    
    feature_names=x_train.columns.tolist(), 
    gamma_min=25.0,
    gamma_max=35.0,
    max_iter=1000
)
end = time.time()
t = end-start
print("time : ", t, "sec")



print(svce_summary)


mean_sparsity = svce_results['sparsity'].mean()
mean_proximity = svce_results['proximity'].mean()

print(f"- Sparsity moyenne   : {mean_sparsity:.3f}")
print(f"- Proximity moyenne  : {mean_proximity:.3f}")








