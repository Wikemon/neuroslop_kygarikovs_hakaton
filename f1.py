import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def evaluate_features(
    df: pd.DataFrame,
    columns=None,
    *,
    target="is_deal",
    exclude=("listing_id", "seller_id"),
    n_splits=5,
    random_state=42,
):
    """Оценивает признаки с помощью логистической регрессии и CV.

    df: таблица с признаками и проверенной бинарной разметкой (0/1).
    columns: список признаков или None (все, кроме target и exclude).
    exclude: идентификаторы/служебные столбцы, не используемые моделью.
    Возвращает словарь средних метрик и стандартное отклонение F1.
    """
    if not df.columns.is_unique:
        raise ValueError("В DataFrame есть повторяющиеся названия столбцов.")
    if target not in df.columns:
        raise ValueError(f"Нет столбца {target!r}. Сначала добавьте verified-разметку.")

    excluded = {target, *exclude}
    if columns is None:
        columns = [col for col in df.columns if col not in excluded]
    else:
        if isinstance(columns, str):
            columns = [columns]
        columns = list(dict.fromkeys(columns))
        if excluded.intersection(columns):
            raise ValueError("Нельзя использовать целевую переменную или исключённые ID как признаки.")

    if not columns:
        raise ValueError("Не выбрано ни одного признака.")

    X = df[columns].copy()
    y = df[target]
    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError("Целевая переменная должна содержать оба класса: 0 и 1, без пропусков.")
    if y.value_counts().min() < n_splits:
        raise ValueError("Недостаточно объектов меньшего класса для выбранного числа фолдов.")
    if "listing_id" in df.columns and not df["listing_id"].is_unique:
        raise ValueError("listing_id должен быть уникальным: одна строка на объявление.")

    # Определяем типы признаков автоматически.
    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [col for col in columns if col not in numeric]

    # Бесконечности обрабатываем как пропуски. Категории приводим к строкам,
    # сохраняя пропуски, чтобы поддерживать также StringDtype и даты.
    for col in numeric:
        X[col] = X[col].astype(float).replace([np.inf, -np.inf], np.nan)
    for col in categorical:
        values = X[col].astype("string")
        X[col] = values.astype(object).where(values.notna(), np.nan)

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000)),
    ])
    cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    scores = cross_validate(
        model, X, y, cv=cv,
        scoring=["f1", "precision", "recall", "roc_auc"],
        error_score="raise",
    )
    return {
        "F1": scores["test_f1"].mean(),
        "F1_std": scores["test_f1"].std(),
        "Precision": scores["test_precision"].mean(),
        "Recall": scores["test_recall"].mean(),
        "ROC_AUC": scores["test_roc_auc"].mean(),
    }
