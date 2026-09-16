import warnings
import joblib

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold
)

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")

DATA_FILE = "telco_churn_clean.csv"

MODEL_FILE = "churn_final_model.pkl"

FEATURE_FILE = "churn_model_features.pkl"

COMPARISON_FILE = "model_comparison.csv"

RANDOM_STATE = 42


try:

    from xgboost import XGBClassifier

    XGB_AVAILABLE = True

    print("XGBoost available.")

except ImportError:

    XGB_AVAILABLE = False

    print(
        "XGBoost is not installed."
    )

    print(
        "Install using: pip install xgboost"
    )


def load_data():

    print("\n" + "=" * 60)
    print("LOADING CLEANED DATA")
    print("=" * 60)

    df = pd.read_csv(
        DATA_FILE
    )

    print(
        "Dataset Shape:",
        df.shape
    )

    return df


def prepare_data(df):

    if "Churn" not in df.columns:

        raise ValueError(
            "Churn column not found in dataset."
        )

    X = df.drop(
        columns=["Churn"]
    )

    y = df["Churn"]

    print(
        "\nFeatures Shape:",
        X.shape
    )

    print(
        "Target Shape:",
        y.shape
    )

    print(
        "\nChurn Distribution:"
    )

    print(
        y.value_counts()
    )

    return X, y



def split_data(X, y):

    print("\n" + "=" * 60)
    print("TRAIN / TEST SPLIT")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=RANDOM_STATE,

        stratify=y
    )

    print(
        "X_train:",
        X_train.shape
    )

    print(
        "X_test :",
        X_test.shape
    )

    print(
        "y_train:",
        y_train.shape
    )

    print(
        "y_test :",
        y_test.shape
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


def evaluate_model(
    name,
    model,
    X_train,
    y_train,
    X_test,
    y_test
):

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    # Train model
    model.fit(
        X_train,
        y_train
    )

    # Predictions
    y_pred = model.predict(
        X_test
    )

    # Probabilities
    y_proba = model.predict_proba(
        X_test
    )[:, 1]

    # Metrics
    metrics = {

        "Model": name,

        "Accuracy": accuracy_score(
            y_test,
            y_pred
        ),

        "Precision": precision_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "F1": f1_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "ROC-AUC": roc_auc_score(
            y_test,
            y_proba
        )
    }

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "No Churn",
                "Churn"
            ],
            zero_division=0
        )
    )

    for key, value in metrics.items():

        if key != "Model":

            print(
                f"{key:<12}: {value:.4f}"
            )

    return (
        metrics,
        model,
        y_proba
    )


def train_baseline_models(
    X_train,
    y_train,
    X_test,
    y_test
):

    models = {}


    lr_model = Pipeline([

        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_STATE
            )
        )
    ])

    result = evaluate_model(

        "Logistic Regression",

        lr_model,

        X_train,
        y_train,

        X_test,
        y_test
    )

    models[
        "Logistic Regression"
    ] = result


    rf_model = RandomForestClassifier(

        n_estimators=100,

        class_weight="balanced",

        random_state=RANDOM_STATE,

        n_jobs=-1
    )

    result = evaluate_model(

        "Random Forest (Default)",

        rf_model,

        X_train,
        y_train,

        X_test,
        y_test
    )

    models[
        "Random Forest (Default)"
    ] = result



    if XGB_AVAILABLE:

        scale_pos_weight = (

            (y_train == 0).sum()

            /

            (y_train == 1).sum()
        )

        xgb_model = XGBClassifier(

            n_estimators=100,

            scale_pos_weight=scale_pos_weight,

            eval_metric="logloss",

            random_state=RANDOM_STATE,

            n_jobs=-1
        )

        result = evaluate_model(

            "XGBoost (Default)",

            xgb_model,

            X_train,
            y_train,

            X_test,
            y_test
        )

        models[
            "XGBoost (Default)"
        ] = result

    return models



def tune_random_forest(
    X_train,
    y_train,
    X_test,
    y_test
):

    print("\n" + "=" * 60)
    print("TUNING RANDOM FOREST")
    print("=" * 60)

    cv = StratifiedKFold(

        n_splits=3,

        shuffle=True,

        random_state=RANDOM_STATE
    )

    rf = RandomForestClassifier(

        class_weight="balanced",

        random_state=RANDOM_STATE,

        n_jobs=-1
    )

    parameters = {

        "n_estimators": [
            100,
            200
        ],

        "max_depth": [
            10,
            20,
            None
        ],

        "max_features": [
            "sqrt",
            "log2"
        ]
    }

    search = RandomizedSearchCV(

        rf,

        parameters,

        n_iter=5,

        scoring="roc_auc",

        cv=cv,

        random_state=RANDOM_STATE,

        n_jobs=-1,

        verbose=1
    )

    search.fit(
        X_train,
        y_train
    )

    print(
        "\nBest Random Forest Parameters:"
    )

    print(
        search.best_params_
    )

    result = evaluate_model(

        "Random Forest (Tuned)",

        search.best_estimator_,

        X_train,
        y_train,

        X_test,
        y_test
    )

    return result



def tune_xgboost(
    X_train,
    y_train,
    X_test,
    y_test
):

    if not XGB_AVAILABLE:

        return None

    print("\n" + "=" * 60)
    print("TUNING XGBOOST")
    print("=" * 60)

    cv = StratifiedKFold(

        n_splits=3,

        shuffle=True,

        random_state=RANDOM_STATE
    )

    scale_pos_weight = (

        (y_train == 0).sum()

        /

        (y_train == 1).sum()
    )

    xgb = XGBClassifier(

        scale_pos_weight=scale_pos_weight,

        eval_metric="logloss",

        random_state=RANDOM_STATE,

        n_jobs=-1
    )

    parameters = {

        "n_estimators": [
            100,
            200
        ],

        "max_depth": [
            3,
            5,
            7
        ],

        "learning_rate": [
            0.05,
            0.1,
            0.2
        ]
    }

    search = RandomizedSearchCV(

        xgb,

        parameters,

        n_iter=5,

        scoring="roc_auc",

        cv=cv,

        random_state=RANDOM_STATE,

        n_jobs=-1,

        verbose=1
    )

    search.fit(
        X_train,
        y_train
    )

    print(
        "\nBest XGBoost Parameters:"
    )

    print(
        search.best_params_
    )

    result = evaluate_model(

        "XGBoost (Tuned)",

        search.best_estimator_,

        X_train,
        y_train,

        X_test,
        y_test
    )

    return result


def compare_models(models):

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    metrics_list = []

    for name, result in models.items():

        metrics_list.append(
            result[0]
        )

    results_df = pd.DataFrame(
        metrics_list
    )

    results_df = (

        results_df

        .set_index("Model")

        .sort_values(
            "ROC-AUC",
            ascending=False
        )
    )

    print(
        results_df.round(4)
    )

    results_df.to_csv(
        COMPARISON_FILE
    )

    print(
        f"\nSaved: {COMPARISON_FILE}"
    )

    return results_df



def create_comparison_chart(
    results_df
):

    results_df[
        [
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC"
        ]
    ].plot(
        kind="bar",
        figsize=(13, 6)
    )

    plt.title(
        "Model Comparison"
    )

    plt.ylabel(
        "Score"
    )

    plt.ylim(
        0.45,
        1.0
    )

    plt.xticks(
        rotation=20,
        ha="right"
    )

    plt.legend(
        loc="lower right"
    )

    plt.tight_layout()

    plt.savefig(
        "model_comparison_chart.png",
        dpi=150
    )

    plt.close()



def select_final_model(
    results_df,
    models
):

    best_name = (
        results_df[
            "ROC-AUC"
        ].idxmax()
    )

    (
        metrics,
        model,
        proba
    ) = models[
        best_name
    ]

    print("\n" + "=" * 60)
    print("FINAL MODEL")
    print("=" * 60)

    print(
        "Selected Model:",
        best_name
    )

    print(
        "ROC-AUC:",
        round(
            metrics["ROC-AUC"],
            4
        )
    )

    print(
        "F1:",
        round(
            metrics["F1"],
            4
        )
    )

    print(
        "Recall:",
        round(
            metrics["Recall"],
            4
        )
    )

    return (
        best_name,
        model
    )



def create_confusion_matrix(
    model,
    model_name,
    X_test,
    y_test
):

    y_pred = model.predict(
        X_test
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    ConfusionMatrixDisplay(

        confusion_matrix=cm,

        display_labels=[
            "No Churn",
            "Churn"
        ]

    ).plot(
        ax=ax,
        colorbar=False
    )

    ax.set_title(
        f"Confusion Matrix\n{model_name}"
    )

    plt.tight_layout()

    plt.savefig(
        "confusion_matrix_final.png",
        dpi=150
    )

    plt.close()



def create_roc_curves(
    models,
    y_test
):

    plt.figure(
        figsize=(8, 6)
    )

    for name, result in models.items():

        proba = result[2]

        fpr, tpr, _ = roc_curve(
            y_test,
            proba
        )

        auc = roc_auc_score(
            y_test,
            proba
        )

        plt.plot(
            fpr,
            tpr,
            lw=2,
            label=f"{name} (AUC={auc:.3f})"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        "k--"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "ROC Curves"
    )

    plt.legend(
        loc="lower right"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        "roc_curves.png",
        dpi=150
    )

    plt.close()



def create_feature_importance(
    model,
    model_name,
    X_train
):

    if not hasattr(
        model,
        "feature_importances_"
    ):

        print(
            "\nFeature importance skipped."
        )

        return

    importance = pd.Series(

        model.feature_importances_,

        index=X_train.columns
    )

    top20 = (
        importance

        .sort_values(
            ascending=False
        )

        .head(20)
    )

    plt.figure(
        figsize=(9, 7)
    )

    sns.barplot(

        x=top20.values,

        y=top20.index
    )

    plt.title(
        f"Top 20 Features - {model_name}"
    )

    plt.xlabel(
        "Importance"
    )

    plt.tight_layout()

    plt.savefig(
        "feature_importance_final.png",
        dpi=150
    )

    plt.close()

    print(
        "\nTop 5 Features:"
    )

    for feature, score in (
        top20.head(5).items()
    ):

        print(
            f"{feature:<40} "
            f"{score:.4f}"
        )


def save_model(
    model,
    X_train
):

    joblib.dump(
        model,
        MODEL_FILE
    )

    joblib.dump(
        list(X_train.columns),
        FEATURE_FILE
    )

    print("\n" + "=" * 60)
    print("MODEL SAVED")
    print("=" * 60)

    print(
        MODEL_FILE
    )

    print(
        FEATURE_FILE
    )


def main():

    print("\n")
    print("=" * 70)
    print("MEMBER 2 - ML MODEL LEAD")
    print("=" * 70)

    # 1. Load Member 1's cleaned dataset
    df = load_data()

    # 2. Prepare X and y
    X, y = prepare_data(
        df
    )

    # 3. Reproducible split
    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_data(
        X,
        y
    )

    # 4. Baseline models
    models = train_baseline_models(
        X_train,
        y_train,
        X_test,
        y_test
    )

    # 5. Tune Random Forest
    rf_result = tune_random_forest(
        X_train,
        y_train,
        X_test,
        y_test
    )

    models[
        "Random Forest (Tuned)"
    ] = rf_result

    # 6. Tune XGBoost
    if XGB_AVAILABLE:

        xgb_result = tune_xgboost(
            X_train,
            y_train,
            X_test,
            y_test
        )

        if xgb_result is not None:

            models[
                "XGBoost (Tuned)"
            ] = xgb_result

    # 7. Compare models
    results_df = compare_models(
        models
    )

    # 8. Comparison chart
    create_comparison_chart(
        results_df
    )

    # 9. Select final model
    (
        best_name,
        final_model
    ) = select_final_model(
        results_df,
        models
    )

    # 10. Diagnostics
    create_confusion_matrix(
        final_model,
        best_name,
        X_test,
        y_test
    )

    create_roc_curves(
        models,
        y_test
    )

    create_feature_importance(
        final_model,
        best_name,
        X_train
    )

    # 11. Save model
    save_model(
        final_model,
        X_train
    )

    print("\n" + "=" * 70)
    print("MEMBER 2 ML PIPELINE COMPLETE")
    print("=" * 70)



if __name__ == "__main__":
    main()
