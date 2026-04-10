import time
import warnings

import dice_ml
import numpy as np
import pandas as pd
from dice_ml import Dice
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


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


def create_dynamic_permitted_range(
    instance: pd.DataFrame,
    percentage: float = 0.5
) -> dict:

    permitted_range = {}

    for col in instance.columns:
        val = instance[col].values[0]
        lower = val * (1 - percentage)
        upper = val * (1 + percentage)
        permitted_range[col] = [lower, upper]

    return permitted_range


def transform_to_dense(preprocessor, df: pd.DataFrame) -> np.ndarray:

    transformed = preprocessor.transform(df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    return np.asarray(transformed)


def evaluate_dice_on_all_test_instances(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    pipeline_model: Pipeline,
    dice_explainer,
    percentage: float = 0.5,
    desired_range: tuple[float, float] = (25.0, 35.0),
    total_cfs: int = 2,
    tol: float = 1e-6
):

    _ = y_test

    preprocessor = pipeline_model.named_steps["preprocessor"]

    results = []
    found = 0
    not_found = 0

    for i in range(len(X_test)):
        query_instance = X_test.iloc[i:i + 1]

        permitted_range = create_dynamic_permitted_range(
            query_instance,
            percentage=percentage
        )

        try:
            cf = dice_explainer.generate_counterfactuals(
                query_instances=query_instance,
                total_CFs=total_cfs,
                desired_range=list(desired_range),
                permitted_range=permitted_range
            )

            cf_df = cf.cf_examples_list[0].final_cfs_df

            if cf_df is not None and not cf_df.empty:
                
                cf_input = cf_df[query_instance.columns].iloc[[0]].copy()

                original_pred = pipeline_model.predict(query_instance)[0]
                cf_pred = pipeline_model.predict(cf_input)[0]

                # Compute metrics in transformed space
                query_transformed = transform_to_dense(preprocessor, query_instance).flatten()
                cf_transformed = transform_to_dense(preprocessor, cf_input).flatten()

                sparsity = int(np.sum(np.abs(cf_transformed - query_transformed) > tol))
                proximity = float(np.sum(np.abs(cf_transformed - query_transformed)))
                reduction = float(
                    (original_pred - cf_pred) / (np.abs(original_pred) + 1e-8) * 100
                )

                results.append({
                    "instance_index": i,
                    "original_prediction": float(original_pred),
                    "counterfactual_prediction": float(cf_pred),
                    "reduction_percent": reduction,
                    "sparsity": sparsity,
                    "proximity": proximity
                })
                found += 1
            else:
                not_found += 1

        except Exception:
            not_found += 1
            continue

    results_df = pd.DataFrame(results)

    summary = {
        "total_instances": len(X_test),
        "found_counterfactuals": found,
        "not_found_counterfactuals": not_found,
        "success_rate_percent": round(100 * found / len(X_test), 2)
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
    # Build DiCE objects
    # -----------------------------
    continuous_features = X.select_dtypes(include=[np.number]).columns.tolist()

    dice_data = dice_ml.Data(
        dataframe=data,
        continuous_features=continuous_features,
        outcome_name=outcome_name
    )

    dice_model = dice_ml.Model(
        model=pipeline_model,
        backend="sklearn",
        model_type="regressor"
    )

    dice_explainer = Dice(dice_data, dice_model, method="genetic")

    # -----------------------------
    # Run DiCE evaluation
    # -----------------------------
    start_time = time.time()

    dice_results, dice_summary = evaluate_dice_on_all_test_instances(
        X_test=x_test,
        y_test=y_test,
        pipeline_model=pipeline_model,
        dice_explainer=dice_explainer,
        percentage=0.5,
        desired_range=(25.0, 35.0),
        total_cfs=2,
        tol=1e-6
    )

    elapsed_time = time.time() - start_time

    # -----------------------------
    # Print outputs
    # -----------------------------
    print("\nExecution time:", round(elapsed_time, 3), "sec")
    print("\nSummary:")
    print(dice_summary)

    if not dice_results.empty:
        mean_sparsity = dice_results["sparsity"].mean()
        mean_proximity = dice_results["proximity"].mean()

        print("\nMean DiCE results:")
        print(f"- Mean sparsity  : {mean_sparsity:.3f}")
        print(f"- Mean proximity : {mean_proximity:.3f}")
    else:
        print("\nNo counterfactuals found.")

    # Optional: save outputs
    dice_results.to_csv("dice_results.csv", index=False)


if __name__ == "__main__":
    main()
