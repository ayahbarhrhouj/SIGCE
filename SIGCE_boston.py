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




#******************************************************************************************************
def sigce(f, X_test, y_test, X_train, feature_names, gamma_min, gamma_max, max_iter=1000):

    preprocessor = f.named_steps["preprocessor"]
    model = f.named_steps["regressor"]
    results = []
    interaction_pairs = []
    found = 0
    not_found = 0
    y_target_min = gamma_min
    y_target_max = gamma_max

    explainer = shap.TreeExplainer(model)
    X_train_transformed = preprocessor.transform(X_train)
    shap_interaction_values = explainer.shap_interaction_values(X_train_transformed)
    if isinstance(shap_interaction_values, list):
        shap_interaction_values = shap_interaction_values[0]
    print("*shap_interaction_values calculated*")

    interaction_strength = np.abs(shap_interaction_values).mean(axis=0)
    for i in range(interaction_strength.shape[0]):
        for j in range(i + 1, interaction_strength.shape[1]):
            interaction_pairs.append(((i, j), interaction_strength[i, j]))

    interaction_pairs = sorted(interaction_pairs, key=lambda x: -x[1])
    top_k = 78
    all_groups = [pair[0] for pair in interaction_pairs[:top_k]]
    print("*top groups selected*")

    for i in range(len(X_test)):
        x = X_test.iloc[i:i+1]
        x_transformed = preprocessor.transform(x).flatten()
        y_pred = model.predict(x.values.reshape(1, -1))[0]

        feature_std = np.std(X_train_transformed, axis=0)
        feature_mean = np.mean(X_train_transformed, axis=0)
        success = False

        cumulative_indices = set()

        for group in all_groups:
           
            selected_indices = group 

            x_prime = x_transformed.copy()
            iteration = 0

            while iteration < max_iter:
                for idx in group:
                    noise = np.random.uniform(-0.1, 0.1) * feature_std[idx]
                    x_prime[idx] += noise
                    x_prime[idx] = np.clip(
                        x_prime[idx],
                        feature_mean[idx] - 3 * feature_std[idx],
                        feature_mean[idx] + 3 * feature_std[idx]
                    )

                    y_pred_prime = model.predict(x_prime.reshape(1, -1))[0]
                    if y_target_min <= y_pred_prime <= y_target_max:
                        success = True
                        sparsity = (x_transformed != x_prime).sum()
                        proximity = np.abs(x_transformed - x_prime).sum()
                        results.append({
                            'instance_index': i,
                            'sparsity': sparsity,
                            'proximity': proximity
                        })
                        found += 1
                        break
                    if success:
                        break
                if success:
                    break
                iteration += 1

            if success:
                break
      
        if not success:
            not_found += 1

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

sigce_results, sigce_summary = sigce(
    f=regr_wind,
    X_test=x_test,
    y_test=y_test,
    X_train=x_train,
    feature_names=x_train.columns.tolist(),
    gamma_min=25.0, 
    gamma_max=35.0
)

end = time.time()
t = end-start
print("time : ", t, "sec")



print(sigce_summary)


mean_sparsity = sigce_results['sparsity'].mean()
mean_proximity = sigce_results['proximity'].mean()

print(f"- Sparsity moyenne   : {mean_sparsity:.3f}")
print(f"- Proximity moyenne  : {mean_proximity:.3f}")








