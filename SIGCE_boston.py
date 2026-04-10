import time
from itertools import combinations

import numpy as np
import pandas as pd
import shap
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


def build_groups_from_correlation(
    df: pd.DataFrame,
    threshold: float = 0.2,
    method: str = "single"
) -> tuple[pd.DataFrame, dict[int, list[str]]]:

    corr_matrix = df.corr().abs()
    distance_matrix = 1 - corr_matrix

    Z = linkage(distance_matrix, method=method)
    clusters = fcluster(Z, t=threshold, criterion="distance")

    grouped_vars: dict[int, list[str]] = {}
    for i, var in enumerate(corr_matrix.columns):
        group_id = int(clusters[i])
        grouped_vars.setdefault(group_id, []).append(var)

    groups_df = pd.DataFrame({
        "Variable": corr_matrix.columns,
        "Groupe": clusters
    })

    return groups_df, grouped_vars


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


def sigce_fast_grouped(
    pipeline_model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_train: pd.DataFrame,
    feature_names: list[str],
    gamma_min: float,
    gamma_max: float,
    groups_df: pd.DataFrame,
    max_iter: int = 1000,
    top_k_groups: int | None = None,
    eps: float = 0.1,
    tol: float = 1e-6,
    shap_sample_size: int | None = None,
    random_state: int = 42,
    return_group_scores: bool = False
):
    
    #Generate grouped counterfactuals using SHAP interaction values.
    _ = y_test
    _ = feature_names

    rng = np.random.default_rng(random_state)

    preprocessor = pipeline_model.named_steps["preprocessor"]
    model = pipeline_model.named_steps["regressor"]

    results = []
    found = 0
    not_found = 0

    X_train_transformed = preprocessor.transform(X_train)
    if hasattr(X_train_transformed, "toarray"):
        X_train_transformed = X_train_transformed.toarray()

    feature_mean = np.mean(X_train_transformed, axis=0)
    feature_std = np.std(X_train_transformed, axis=0)
    feature_std[feature_std == 0] = 1e-8

    if shap_sample_size is not None and shap_sample_size < len(X_train_transformed):
        sample_idx = rng.choice(
            len(X_train_transformed),
            size=shap_sample_size,
            replace=False
        )
        X_shap = X_train_transformed[sample_idx]
    else:
        X_shap = X_train_transformed

    explainer = shap.TreeExplainer(model)
    shap_interaction_values = explainer.shap_interaction_values(X_shap)
    if isinstance(shap_interaction_values, list):
        shap_interaction_values = shap_interaction_values[0]

    print("* SHAP interaction values computed *")

    interaction_strength = np.abs(shap_interaction_values).mean(axis=0)

    raw_vars = list(X_train.columns)
    var_to_idx = {var: i for i, var in enumerate(raw_vars)}

    grouped_vars_idx: dict[int, list[int]] = {}
    for _, row in groups_df.iterrows():
        var = row["Variable"]
        group_id = row["Groupe"]
        if var in var_to_idx:
            grouped_vars_idx.setdefault(group_id, []).append(var_to_idx[var])

    grouped_vars_idx = {
        gid: sorted(set(idxs))
        for gid, idxs in grouped_vars_idx.items()
        if len(idxs) > 0
    }

    group_scores = []
    for gid, idxs in grouped_vars_idx.items():
        if len(idxs) > 1:
            pair_scores = [interaction_strength[i, j] for i, j in combinations(idxs, 2)]
            score = float(np.mean(pair_scores)) if pair_scores else 0.0
        else:
            idx_single = idxs[0]
            score = float(interaction_strength[idx_single, idx_single])

        group_scores.append({
            "Groupe": gid,
            "Variables_idx": idxs,
            "Group_size": len(idxs),
            "Interaction_score": score
        })

    group_scores_df = pd.DataFrame(group_scores).sort_values(
        by="Interaction_score",
        ascending=False
    ).reset_index(drop=True)

    if top_k_groups is None:
        top_k_groups = len(group_scores_df)

    selected_groups_df = group_scores_df.head(min(top_k_groups, len(group_scores_df)))
    all_groups = selected_groups_df["Variables_idx"].tolist()

    print("* Top groups selected *")

    for i in range(len(X_test)):
        x = X_test.iloc[i:i + 1]

        x_transformed = preprocessor.transform(x)
        if hasattr(x_transformed, "toarray"):
            x_transformed = x_transformed.toarray()
        x_transformed = x_transformed.flatten()

        y_pred = model.predict(x_transformed.reshape(1, -1))[0]
        success = False

        for m in range(1, len(all_groups) + 1):
            selected_groups = all_groups[:m]
            x_prime = x_transformed.copy()
            iteration = 0

            while iteration < max_iter:
                for group in selected_groups:
                    delta = rng.uniform(-eps, eps)

                    for idx in group:
                        x_prime[idx] += delta * feature_std[idx]
                        x_prime[idx] = np.clip(
                            x_prime[idx],
                            feature_mean[idx] - 3 * feature_std[idx],
                            feature_mean[idx] + 3 * feature_std[idx]
                        )

                y_pred_prime = model.predict(x_prime.reshape(1, -1))[0]

                if gamma_min <= y_pred_prime <= gamma_max:
                    success = True

                    sparsity = int(np.sum(np.abs(x_transformed - x_prime) > tol))
                    proximity = float(np.sum(np.abs(x_transformed - x_prime)))
                    reduction = float(
                        (y_pred - y_pred_prime) / (np.abs(y_pred) + 1e-8) * 100
                    )

                    results.append({
                        "instance_index": i,
                        "original_prediction": float(y_pred),
                        "counterfactual_prediction": float(y_pred_prime),
                        "reduction_percent": reduction,
                        "sparsity": sparsity,
                        "proximity": proximity,
                        "num_groups_used": m
                    })
                    found += 1
                    break

                iteration += 1

            if success:
                break

        if not success:
            not_found += 1

    results_df = pd.DataFrame(results)

    summary = {
        "total_instances": len(X_test),
        "found_counterfactuals": found,
        "not_found_counterfactuals": not_found,
        "found_ratio_percent": round(100 * found / len(X_test), 2)
    }

    if return_group_scores:
        return results_df, summary, group_scores_df

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
    # Build groups from correlations
    # -----------------------------
    groups_df, grouped_vars = build_groups_from_correlation(
        df=X,
        threshold=0.2,
        method="single"
    )

    print("Grouped variables:")
    for group_id, variables in grouped_vars.items():
        print(f"Group {group_id}: {variables}")

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
    # Run SIGCE
    # -----------------------------
    start_time = time.time()

    results_df, summary, group_scores_df = sigce_fast_grouped(
        pipeline_model=pipeline_model,
        X_test=x_test,
        y_test=y_test,
        X_train=x_train,
        feature_names=x_train.columns.tolist(),
        gamma_min=25.0,
        gamma_max=35.0,
        groups_df=groups_df,
        max_iter=1000,
        top_k_groups=groups_df["Groupe"].nunique(),
        eps=0.1,
        shap_sample_size=200,
        return_group_scores=True
    )

    elapsed_time = time.time() - start_time

    # -----------------------------
    # Print outputs
    # -----------------------------
    print("\nExecution time:", round(elapsed_time, 3), "sec")
    print("\nSummary:")
    print(summary)

    if not results_df.empty:
        mean_sparsity = results_df["sparsity"].mean()
        mean_proximity = results_df["proximity"].mean()
        print(f"\nMean sparsity  : {mean_sparsity:.3f}")
        print(f"Mean proximity : {mean_proximity:.3f}")
    else:
        print("\nNo counterfactuals found.")

    print("\nTop grouped interaction scores:")
    print(group_scores_df.head())

    results_df.to_csv("results_df.csv", index=False)
    group_scores_df.to_csv("group_scores_df.csv", index=False)


if __name__ == "__main__":
    main()