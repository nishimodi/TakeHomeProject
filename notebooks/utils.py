import csv
import numpy as np
import pandas as pd
import warnings

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# Metrics
from sklearn.metrics import (
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)

# HP Tunings
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split


def summary_statistics(df):
    """
    Get basic stats for all columns in a dataframe.
    Handles continuous, categorical, and datetime columns differently.
    """
    var_name, var_type, missing_cnt, freq_val, mean, std = [], [], [], [], [], []
    min, max, unique, outlier = [], [], [], []
    per_5, per_10, per_25, median, per_75, per_90, per_95 = [], [], [], [], [], [], []
    per_zero, per_neg = [], []

    column_name = [
        "Variable_Name",
        "Variable_Type",
        "Missing_Count",
        "Most_Frequent_Value",
        "Mean",
        "Standard_Deviation",
        "Min",
        "Max",
        "Unique_Values",
        "IQR_Outliers",
        "5th percentile",
        "10th percentile",
        "25th percentile",
        "50th percentile (Median)",
        "75th percentile",
        "90th percentile",
        "95th percentile",
        "Percentage_of_Zeros",
        "Percentage_of_Negatives",
    ]

    for column in df.columns:
        var_name.append(column)
        missing_cnt.append(df[column].isnull().sum())

        if pd.isnull(df[column]).all():
            freq_val.append(np.nan)
        else:
            freq_val.append(df[column].mode().iloc[0])

        # Continuous variable
        if df[column].dtype == "float64" or df[column].dtype == "int64":
            var_type.append("Continuous")

            # --- Existing Continuous Calculations ---
            mean.append(df[column].mean())
            std.append(df[column].std())
            min.append(df[column].min())
            max.append(df[column].max())
            unique.append(df[column].nunique())

            # Outlier calculation
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Calculate percentage of outliers
            outlier.append(
                (((df[column] < lower_bound) | (df[column] > upper_bound)).sum())
                * 100
                / df.shape[0]
            )

            # Percentile calculations
            per_5.append(df[column].quantile(0.05))
            per_10.append(df[column].quantile(0.1))
            per_25.append(df[column].quantile(0.25))
            median.append(df[column].quantile(0.5))
            per_75.append(df[column].quantile(0.75))
            per_90.append(df[column].quantile(0.9))
            per_95.append(df[column].quantile(0.95))

            # Calculate percentage of zero values
            total_count = df[column].dropna().shape[0]  # Count non-missing values
            zero_count = (df[column].dropna() == 0).sum()
            if total_count > 0:
                per_zero.append(zero_count * 100 / total_count)
            else:
                per_zero.append(0.0)  # Handle case of all missing values

            # Calculate percentage of negative values
            neg_count = (df[column].dropna() < 0).sum()
            if total_count > 0:
                per_neg.append(neg_count * 100 / total_count)
            else:
                per_neg.append(0.0)  # Handle case of all missing values

        # Date variables
        elif df[column].dtype == "datetime64[ns]":
            var_type.append("DateTime")
            mean.append("NA")
            std.append("NA")
            min.append(df[column].dt.date.min())
            max.append(df[column].dt.date.max())
            unique.append(df[column].dt.date.nunique())
            outlier.append("NA")
            per_5.append("NA")
            per_10.append("NA")
            per_25.append("NA")
            median.append("NA")
            per_75.append("NA")
            per_90.append("NA")
            per_95.append("NA")
            # Add 'NA' for new columns
            per_zero.append("NA")
            per_neg.append("NA")

        # Categorical variables
        elif (
            df[column].dtype == "object" or df[column].dtype == "bool"
        ):  # Added 'bool' as it often acts as categorical
            var_type.append("Categorical")
            mean.append("NA")
            std.append("NA")
            min.append("NA")
            max.append("NA")
            unique.append(df[column].nunique())
            outlier.append("NA")
            per_5.append("NA")
            per_10.append("NA")
            per_25.append("NA")
            median.append("NA")
            per_75.append("NA")
            per_90.append("NA")
            per_95.append("NA")
            per_zero.append("NA")
            per_neg.append("NA")

        # Default for other types (e.g., category, timedelta)
        else:
            var_type.append(str(df[column].dtype))
            mean.append("NA")
            std.append("NA")
            min.append("NA")
            max.append("NA")
            unique.append(df[column].nunique())
            outlier.append("NA")
            per_5.append("NA")
            per_10.append("NA")
            per_25.append("NA")
            median.append("NA")
            per_75.append("NA")
            per_90.append("NA")
            per_95.append("NA")
            per_zero.append("NA")
            per_neg.append("NA")

    summary = pd.DataFrame(
        list(
            zip(
                var_name,
                var_type,
                missing_cnt,
                freq_val,
                mean,
                std,
                min,
                max,
                unique,
                outlier,
                per_5,
                per_10,
                per_25,
                median,
                per_75,
                per_90,
                per_95,
                per_zero,
                per_neg,
            )
        ),
        columns=column_name,
    )

    return summary


def bivariate_report(df, target, weight_col="weight"):
    """
    Creates a summary table showing how categorical variables relate to the target.
    Includes both regular counts and weighted counts if you have sampling weights.
    """

    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    reports = []
    total_rows = len(df)
    total_weighted = df[weight_col].sum()
    y = df[target].astype(int)

    for col in cat_cols:
        if col == target:
            continue

        grp = df[[col, weight_col]].join(y)

        temp = (
            grp.groupby(col)
            .apply(
                lambda x: pd.Series(
                    {
                        "Count_1": (x[target] == 1).sum(),
                        "Count_0": (x[target] == 0).sum(),
                        "Total": len(x),
                        "Weighted_Count_1": (x[target] * x[weight_col]).sum(),
                        "Weighted_Count_0": ((1 - x[target]) * x[weight_col]).sum(),
                        "Weighted_Total": x[weight_col].sum(),
                    }
                )
            )
            .reset_index()
        )

        temp["Event_Rate_%"] = (temp["Count_1"] / temp["Total"] * 100).round(2)
        temp["Weighted_Event_Rate_%"] = (
            temp["Weighted_Count_1"] / temp["Weighted_Total"] * 100
        ).round(2)
        temp["Population_%"] = (temp["Total"] / total_rows * 100).round(2)
        temp["Weighted_Population_%"] = (
            temp["Weighted_Total"] / total_weighted * 100
        ).round(2)

        temp.insert(0, "Variable", col)
        temp = temp.rename(columns={col: "Category"})
        temp = temp[
            [
                "Variable",
                "Category",
                "Count_1",
                "Count_0",
                "Total",
                "Event_Rate_%",
                "Weighted_Event_Rate_%",
                "Population_%",
                "Weighted_Population_%",
            ]
        ]

        reports.append(temp)

    return pd.concat(reports, ignore_index=True)


def clean_string_values(df):
    """Removes extra spaces and converts to lowercase"""

    df_clean = df.copy()
    cat_cols = df_clean.select_dtypes(include=["object"]).columns

    for col in cat_cols:
        df_clean[col] = df_clean[col].str.strip()
        df_clean[col] = df_clean[col].str.lower()

    return df_clean


# function to get F1 score
def get_metrics_(X_train, y_train, X_test, y_test, pipe_model):
    """Calculate and print accuracy and F1 scores for train and test sets"""

    train_accuracy, test_accuracy = (
        pipe_model.score(X_train, y_train),
        pipe_model.score(X_test, y_test),
    )

    print(f"Train accuracy is - {train_accuracy:2f}")
    print(f"Test accuracy is - {test_accuracy:2f}")

    y_train_prediction, y_test_prediction = (
        pipe_model.predict(X_train),
        pipe_model.predict(X_test),
    )

    train_f1, test_f1 = (
        f1_score(y_train, y_train_prediction),
        f1_score(y_test, y_test_prediction),
    )

    print(f"Train F1 score is - {train_f1:2f}")
    print(f"Test F1 score is - {test_f1:2f}")

    return train_accuracy, test_accuracy, train_f1, test_f1


# function to plot AUC ROC curve
def get_auc_roc_curve(X_train, y_train, X_test, y_test, pipe_model):
    """Plot ROC curves and return AUC scores"""
    y_train_predicted_probabality, y_test_predicted_probabality = (
        pipe_model.predict_proba(X_train)[:, 1],
        pipe_model.predict_proba(X_test)[:, 1],
    )

    train_auc, test_auc = (
        roc_auc_score(y_train, y_train_predicted_probabality),
        roc_auc_score(y_test, y_test_predicted_probabality),
    )

    print(f"Train AUC ROC is - {train_auc:2f}")
    print(f"Test AUC ROC is - {test_auc:2f}")

    train_fpr, train_tpr, train_cuts = roc_curve(y_train, y_train_predicted_probabality)
    test_fpr, test_tpr, test_cuts = roc_curve(y_test, y_test_predicted_probabality)

    plt.plot(
        train_fpr,
        train_tpr,
        lw=2,
        label="Train AUC-ROC Graph (%0.2f)" % train_auc,
        color="green",
    )
    plt.plot(
        test_fpr,
        test_tpr,
        lw=2,
        label="Test AUC-ROC Graph (%0.2f)" % test_auc,
        color="blue",
    )
    plt.plot([0, 1], [0, 1], color="red", lw=2, linestyle="--")

    plt.title("ROC Curve")
    plt.ylabel("True positive rate")
    plt.xlabel("False positive rate")
    plt.ylim([0.0, 1.1])
    plt.xlim([0.0, 1.0])
    plt.legend(loc="upper right")

    return train_auc, test_auc


# function to get the precision recall curve
def get_precision_recall_curve(X_train, y_train, X_test, y_test, pipe_model):
    """Plot precision-recall curves and return precision/recall scores"""

    y_train_prediction, y_test_prediction = (
        pipe_model.predict(X_train),
        pipe_model.predict(X_test),
    )
    y_train_predicted_probabality, y_test_predicted_probabality = (
        pipe_model.predict_proba(X_train)[:, 1],
        pipe_model.predict_proba(X_test)[:, 1],
    )

    train_precision, train_recall, train_cuts = precision_recall_curve(
        y_train, y_train_predicted_probabality
    )
    test_precision, test_recall, test_cuts = precision_recall_curve(
        y_test, y_test_predicted_probabality
    )

    y_train_predicted_binary = pipe_model.predict(X_train)
    y_test_predicted_binary = pipe_model.predict(X_test)

    precsion_score_train, precsion_score_test = (
        precision_score(y_train, y_train_predicted_binary),
        precision_score(y_test, y_test_predicted_binary),
    )
    recall_score_train, recall_score_test = (
        recall_score(y_train, y_train_predicted_binary),
        recall_score(y_test, y_test_predicted_binary),
    )

    print(f"Train Precision is - {precsion_score_train:2f}")
    print(f"Train Recall is - {recall_score_train:2f}")

    print(f"Test Precision is - {precsion_score_test:2f}")
    print(f"Test Recall is - {recall_score_test:2f}")

    plt.step(
        train_recall,
        train_precision,
        lw=2,
        label=f"Train PR Graph (F1 score ={f1_score(y_train, y_train_prediction):2f})",
        color="green",
    )
    plt.step(
        test_recall,
        test_precision,
        lw=2,
        label=f"Test PR Graph (F1 score ={f1_score(y_test, y_test_prediction):2f})",
        color="blue",
    )
    plt.title("PR Curve")
    plt.ylabel("Precision")
    plt.xlabel("Recall")
    plt.ylim([0.0, 1.1])
    plt.xlim([0.0, 1.0])
    plt.legend(loc="upper right")

    return (
        precsion_score_train,
        precsion_score_test,
        recall_score_train,
        recall_score_test,
    )


def top_features(
    processing, optimal_model_pipe, cols_for_numeric, cols_for_target_encoding
):
    """
    Gets top 10 important features from  model.
    Shows a bar chart of the top 10.
    """

    search_estimator = optimal_model_pipe[1]

    if hasattr(search_estimator, "best_estimator_"):
        optimal_model_object = search_estimator.best_estimator_
    else:
        optimal_model_object = search_estimator

    if not hasattr(optimal_model_object, "feature_importances_"):
        print(
            "Error: The final model does not have a 'feature_importances_' attribute."
        )
        return []

    combined_features = list(processing.get_feature_names_out())
    ohe_features = [
        col.split("__", 1)[1]
        for col in combined_features
        if col.startswith("pipeline-2__")
    ]
    combined_features = cols_for_numeric + ohe_features + cols_for_target_encoding
    important_features = pd.DataFrame(
        {
            "feature": combined_features,
            "importance": optimal_model_object.feature_importances_,
        }
    )

    important_features = important_features[
        important_features["importance"] != 0
    ].sort_values(by="importance", ascending=False)

    top_feature = important_features["feature"].tolist()[:11]
    importance = important_features["importance"].tolist()[:11]

    df_plot = pd.DataFrame(
        list(zip(top_feature, importance)), columns=["feature", "importance"]
    )
    df_plot.sort_values(by="importance", ascending=True).plot.barh(
        figsize=(10, 5), x="feature", legend=False
    )
    plt.title("Top 10 features")
    plt.xlabel("Feature Importance")
    plt.ylabel("Variables")
    plt.tight_layout()  # Improves layout for long labels
    plt.show()  #

    return combined_features


def weighted_mean(group):
    """Helper function to calculate weighted average of the target variable"""
    if "target" not in group.columns:
        return np.nan
    return np.average(group["target"], weights=group["weight"])

def get_state_mapping(value):
    value = str(value).strip().lower()

    if value == "?" or value == "nan" or value == "not in universe":
        return "not in universe"

    elif value == "abroad":
        return "abroad"

    else:
        return "us_state"
