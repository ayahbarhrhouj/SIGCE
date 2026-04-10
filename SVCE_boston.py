import time

import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


def build_regression_pipeline(
    X_train: pd.DataFrame,
    random_state: int = 42
) -> Pipeline:
    
    continuous_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.columns.difference(continuous_features).tolist()

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, continuous_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    regressor = XGBRegressor(
        colsample_bytree=1,
        learning_rate=0.1,
        max_depth=4,
        min_child_weight=1,
        n_estimators=200,
        reg_alpha=0,
        reg_lambda=0.5,
        subsample=1,
        random_state=random_state
    )

    pipeline_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])

    return pipeline_model


def svce(
    pipeline_model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_train: pd.DataFrame,
    feature_names: list[str],
    gamma_min: float,
    gamma_max: float,
    max_iter: int = 1000,
    eps: float = 0.1,
    tol: float = 1e-6,
    random_state: int = 42
):
    
    _ = y_test
    _ = feature_names

    rng = np.random.default_rng(random_state)

    preprocessor = pipeline_model.named_steps["preprocessor"]
    model = pipeline_model.named_steps["regressor"]

    results = []
    found = 0
    not_found = 0

    y_target_min = gamma_min
    y_target_max = gamma_max

    X_train_transformed = preprocessor.transform(X_train)
    if hasattr(X_train_transformed, "toarray"):
        X_train_transformed = X_train_transformed.toarray()

    feature_std = np.std(X_train_transformed, axis=0)
    feature_mean = np.mean(X_train_transformed, axis=0)
    feature_std[feature_std == 0] = 1e-8

    explainer = shap.TreeExplainer(model)

    for i in range(len(X_test)):
        x = X_test.iloc[i:i + 1]

        x_transformed = preprocessor.transform(x)
        if hasattr(x_transformed, "toarray"):
            x_transformed = x_transformed.toarray()
        x_transformed = x_transformed.flatten()

        shap_values = explainer.shap_values(x_transformed.reshape(1, -1))
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        phi_abs = np.abs(shap_values).flatten()
        ranked_features = [(j, phi_abs[j]) for j in range(len(phi_abs))]
        ranked_features.sort(key=lambda t: -t[1])

        y_pred = model.predict(x_transformed.reshape(1, -1))[0]

        success = False
        x_cf = None
        y_cf = None

        
        for m in range(1, len(ranked_features) + 1):
            selected_indices = [j for j, _ in ranked_features[:m]]
            x_prime = x_transformed.copy()
            iteration = 0

            while iteration < max_iter:
                for idx in selected_indices:
                    noise = rng.uniform(-eps, eps) * feature_std[idx]
                    x_prime[idx] += noise
                    x_prime[idx] = np.clip(
                        x_prime[idx],
                        feature_mean[idx] - 3 * feature_std[idx],
                        feature_mean[idx] + 3 * feature_std[idx]
                    )

                y_pred_prime = model.predict(x_prime.reshape(1, -1))[0]

                if y_target_min <= y_pred_prime <= y_target_max:
                    success = True
                    x_cf = x_prime.copy()
                    y_cf = y_pred_prime
                    break

                iteration += 1

            if success:
                break

        if success and x_cf is not None:
            sparsity = int(np.sum(np.abs(x_transformed - x_cf) > tol))
            proximity = float(np.sum(np.abs(x_transformed - x_cf)))
            reduction = float(
                (y_pred - y_cf) / (np.abs(y_pred) + 1e-8) * 100
            )

            results.append({
                "instance_index": i,
                "original_prediction": float(y_pred),
                "counterfactual_prediction": float(y_cf),
                "reduction_percent": reduction,
                "sparsity": sparsity,
                "proximity": proximity,
                "num_features_used": m
            })
            found += 1
        else:
            not_found += 1

    results_df = pd.DataFrame(results)

    summary = {
        "total_instances": len(X_test),
        "found_counterfactuals": found,
        "not_found_counterfactuals": not_found,
        "found_ratio_percent": round(100 * found / len(X_test), 2)
    }

    return results_df, summary


def main():
    # -----------------------------
    # Load dataset
    # -----------------------------
    data = pd.read_csv("boston.csv")

    outcome_name = "MEDV"
    X = data.drop(columns=[outcome_name])
    y = data[outcome_name]

    # -----------------------------
    # Train / test split
    # -----------------------------
    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=0
    )

    # -----------------------------
    # Train model
    # -----------------------------
    pipeline_model = build_regression_pipeline(x_train)
    pipeline_model.fit(x_train, y_train)

    # -----------------------------
    # Run SVCE
    # -----------------------------
    start_time = time.time()

    svce_results, svce_summary = svce(
        pipeline_model=pipeline_model,
        X_test=x_test,
        y_test=y_test,
        X_train=x_train,
        feature_names=x_train.columns.tolist(),
        gamma_min=25.0,
        gamma_max=35.0,
        max_iter=1000,
        eps=0.1,
        tol=1e-6,
        random_state=42
    )

    elapsed_time = time.time() - start_time

    # -----------------------------
    # Print outputs
    # -----------------------------
    print("\nExecution time:", round(elapsed_time, 3), "sec")
    print("\nSummary:")
    print(svce_summary)

    if not svce_results.empty:
        mean_sparsity = svce_results["sparsity"].mean()
        mean_proximity = svce_results["proximity"].mean()

        print("\nMean SVCE results:")
        print(f"- Mean sparsity  : {mean_sparsity:.3f}")
        print(f"- Mean proximity : {mean_proximity:.3f}")
    else:
        print("\nNo counterfactuals found.")


    svce_results.to_csv("svce_results.csv", index=False)


if __name__ == "__main__":
    main()
