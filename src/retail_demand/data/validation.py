from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


TRAIN_REQUIRED_COLUMNS = {
    "Store",
    "DayOfWeek",
    "Date",
    "Sales",
    "Customers",
    "Open",
    "Promo",
    "StateHoliday",
    "SchoolHoliday",
}

STORE_REQUIRED_COLUMNS = {
    "Store",
    "StoreType",
    "Assortment",
    "CompetitionDistance",
    "CompetitionOpenSinceMonth",
    "CompetitionOpenSinceYear",
    "Promo2",
    "Promo2SinceWeek",
    "Promo2SinceYear",
    "PromoInterval",
}


@dataclass(frozen=True)
class ValidationIssue:
    """One raw-data validation problem."""

    severity: str
    check: str
    message: str


def _missing_columns(df: pd.DataFrame, required_columns: set[str]) -> list[str]:
    return sorted(required_columns.difference(df.columns))


def _unexpected_values(
    df: pd.DataFrame, column: str, allowed_values: set[object]
) -> set[object]:
    if column not in df.columns:
        return set()
    return set(df[column].dropna().unique()).difference(allowed_values)


def validate_train_data(train_df: pd.DataFrame) -> list[ValidationIssue]:
    """Validate raw train.csv assumptions needed before modeling."""

    issues: list[ValidationIssue] = []

    missing = _missing_columns(train_df, TRAIN_REQUIRED_COLUMNS)
    if missing:
        issues.append(
            ValidationIssue(
                severity="error",
                check="train_required_columns",
                message=f"train.csv is missing columns: {missing}",
            )
        )
        return issues

    if train_df.empty:
        issues.append(
            ValidationIssue("error", "train_not_empty", "train.csv has no rows.")
        )

    if train_df.duplicated().any():
        issues.append(
            ValidationIssue(
                "error",
                "train_duplicate_rows",
                "train.csv contains duplicate rows.",
            )
        )

    if train_df[["Store", "Date"]].duplicated().any():
        issues.append(
            ValidationIssue(
                "error",
                "train_store_date_unique",
                "train.csv contains duplicate Store-Date observations.",
            )
        )

    if train_df["Sales"].lt(0).any():
        issues.append(
            ValidationIssue(
                "error",
                "sales_non_negative",
                "Sales contains negative values.",
            )
        )

    if train_df["Customers"].lt(0).any():
        issues.append(
            ValidationIssue(
                "error",
                "customers_non_negative",
                "Customers contains negative values.",
            )
        )

    for column in ["Open", "Promo", "SchoolHoliday"]:
        unexpected = _unexpected_values(train_df, column, {0, 1})
        if unexpected:
            issues.append(
                ValidationIssue(
                    "error",
                    f"{column.lower()}_binary",
                    f"{column} contains unexpected values: {sorted(unexpected)}",
                )
            )

    unexpected_day_values = _unexpected_values(train_df, "DayOfWeek", set(range(1, 8)))
    if unexpected_day_values:
        issues.append(
            ValidationIssue(
                "error",
                "day_of_week_range",
                f"DayOfWeek contains unexpected values: {sorted(unexpected_day_values)}",
            )
        )

    unexpected_holidays = _unexpected_values(train_df, "StateHoliday", {"0", "a", "b", "c"})
    if unexpected_holidays:
        issues.append(
            ValidationIssue(
                "error",
                "state_holiday_values",
                f"StateHoliday contains unexpected values: {sorted(unexpected_holidays)}",
            )
        )

    return issues


def validate_store_data(store_df: pd.DataFrame) -> list[ValidationIssue]:
    """Validate raw store.csv assumptions needed before joining store features."""

    issues: list[ValidationIssue] = []

    missing = _missing_columns(store_df, STORE_REQUIRED_COLUMNS)
    if missing:
        issues.append(
            ValidationIssue(
                severity="error",
                check="store_required_columns",
                message=f"store.csv is missing columns: {missing}",
            )
        )
        return issues

    if store_df.empty:
        issues.append(
            ValidationIssue("error", "store_not_empty", "store.csv has no rows.")
        )

    if store_df["Store"].duplicated().any():
        issues.append(
            ValidationIssue(
                "error",
                "store_id_unique",
                "store.csv contains duplicate Store rows.",
            )
        )

    unexpected_store_types = _unexpected_values(store_df, "StoreType", {"a", "b", "c", "d"})
    if unexpected_store_types:
        issues.append(
            ValidationIssue(
                "error",
                "store_type_values",
                f"StoreType contains unexpected values: {sorted(unexpected_store_types)}",
            )
        )

    unexpected_assortments = _unexpected_values(store_df, "Assortment", {"a", "b", "c"})
    if unexpected_assortments:
        issues.append(
            ValidationIssue(
                "error",
                "assortment_values",
                f"Assortment contains unexpected values: {sorted(unexpected_assortments)}",
            )
        )

    unexpected_promo2 = _unexpected_values(store_df, "Promo2", {0, 1})
    if unexpected_promo2:
        issues.append(
            ValidationIssue(
                "error",
                "promo2_binary",
                f"Promo2 contains unexpected values: {sorted(unexpected_promo2)}",
            )
        )

    return issues


def validate_train_store_relationship(
    train_df: pd.DataFrame, store_df: pd.DataFrame
) -> list[ValidationIssue]:
    """Validate assumptions across train.csv and store.csv."""

    issues: list[ValidationIssue] = []

    if "Store" not in train_df.columns or "Store" not in store_df.columns:
        return issues

    train_stores = set(train_df["Store"].unique())
    store_stores = set(store_df["Store"].unique())

    missing_store_rows = sorted(train_stores.difference(store_stores))
    if missing_store_rows:
        issues.append(
            ValidationIssue(
                "error",
                "train_stores_exist_in_store",
                f"Stores exist in train.csv but not store.csv: {missing_store_rows[:10]}",
            )
        )

    return issues


def issues_to_dicts(issues: list[ValidationIssue]) -> list[dict[str, str]]:
    return [
        {"severity": issue.severity, "check": issue.check, "message": issue.message}
        for issue in issues
    ]

