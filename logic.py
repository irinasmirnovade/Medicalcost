"""
logic.py
Gesamte Logik fuer die Insurance-Charges-Analyse:
Daten laden, Preprocessing, Modelle, Evaluation, Analyse, Plots, Annahmen-Check.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

from sklearn.model_selection import train_test_split, cross_validate, RandomizedSearchCV, KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    PolynomialFeatures,
    FunctionTransformer,
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor


# ============================================
# Konfiguration
# ============================================

@dataclass
class Config:
    target: str = "charges"
    numeric_features: List[str] = field(
        default_factory=lambda: ["age", "bmi", "children"]
    )
    categorical_features: List[str] = field(
        default_factory=lambda: ["sex", "smoker", "region"]
    )
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5


# ============================================
# 1. Daten
# ============================================

class DataManager:
    """Laedt die Daten und erzeugt den Train/Test-Split."""

    def __init__(self, config: Config):
        self.config = config
        self.df: pd.DataFrame = None
        self.X: pd.DataFrame = None
        self.y: pd.Series = None
        self.X_train = self.X_test = self.y_train = self.y_test = None

    def load(self, file_path: str) -> "DataManager":
        self.df = pd.read_csv(file_path)
        self.X = self.df.drop(self.config.target, axis=1)
        self.y = self.df[self.config.target]
        return self

    def split(self) -> "DataManager":
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        ) = train_test_split(
            self.X,
            self.y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
        )
        return self


# ============================================
# 2. Preprocessing
# ============================================

def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Zusatzmerkmal: Adipositas (BMI >= 30), wichtig in Kombination mit smoker."""
    X = X.copy()
    X["bmi_obese"] = (X["bmi"] >= 30).astype(int)
    return X


class PreprocessorFactory:
    """Erzeugt die verschiedenen ColumnTransformer."""

    def __init__(self, config: Config):
        self.num = config.numeric_features
        self.cat = config.categorical_features

    def standard(self) -> ColumnTransformer:
        return ColumnTransformer(
            transformers=[
                ("numeric", StandardScaler(), self.num),
                ("categorical", OneHotEncoder(handle_unknown="ignore"), self.cat),
            ]
        )

    def polynomial(self) -> ColumnTransformer:
        return ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    Pipeline(
                        steps=[
                            ("scaler", StandardScaler()),
                            ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
                        ]
                    ),
                    self.num,
                ),
                (
                    "categorical",
                    OneHotEncoder(drop="first", handle_unknown="ignore"),
                    self.cat,
                ),
            ]
        )

    def interaction(self) -> ColumnTransformer:
        return ColumnTransformer(
            transformers=[
                ("numeric", StandardScaler(), self.num + ["bmi_obese"]),
                (
                    "categorical",
                    OneHotEncoder(drop="first", handle_unknown="ignore"),
                    self.cat,
                ),
            ]
        )

    def tree(self) -> ColumnTransformer:
        """Fuer baumbasierte Modelle (RF, XGBoost): keine Skalierung noetig."""
        return ColumnTransformer(
            transformers=[
                ("numeric", "passthrough", self.num),
                ("categorical", OneHotEncoder(handle_unknown="ignore"), self.cat),
            ]
        )


# ============================================
# 3. Modelle
# ============================================

class ModelFactory:
    """Baut alle Modell-Pipelines."""

    DEFAULT_XGB_PARAMS = dict(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        n_jobs=-1,
    )

    def __init__(self, config: Config, xgb_params: Optional[dict] = None):
        self.config = config
        self.prep = PreprocessorFactory(config)
        self.xgb_params = {
            **self.DEFAULT_XGB_PARAMS,
            "random_state": config.random_state,
            **(xgb_params or {}),
        }

    def build_all(self) -> Dict[str, Pipeline]:
        rs = self.config.random_state

        return {
            "Linear Regression": Pipeline([
                ("preprocessor", self.prep.standard()),
                ("regressor", LinearRegression()),
            ]),
            "Ridge Regression": Pipeline([
                ("preprocessor", self.prep.standard()),
                ("regressor", Ridge(alpha=10.0)),
            ]),
            "Lasso Regression": Pipeline([
                ("preprocessor", self.prep.standard()),
                ("regressor", Lasso(alpha=10.0, max_iter=10000)),
            ]),
            "Polynomial Regression": Pipeline([
                ("preprocessor", self.prep.polynomial()),
                ("regressor", LinearRegression()),
            ]),
            # Erst kodieren, DANACH Polynome/Interaktionen ueber alle Spalten
            "Poly + Interactions (Ridge)": Pipeline([
                ("features", FunctionTransformer(add_features)),
                ("preprocessor", self.prep.interaction()),
                ("poly", PolynomialFeatures(degree=2, include_bias=False)),
                ("regressor", Ridge(alpha=0.1)),
            ]),
            "Random Forest": Pipeline([
                ("preprocessor", self.prep.tree()),
                ("regressor", RandomForestRegressor(
                    n_estimators=200,
                    max_depth=7,
                    random_state=rs,
                    n_jobs=-1,
                )),
            ]),
            "XGBoost": Pipeline([
                ("preprocessor", self.prep.tree()),
                ("regressor", XGBRegressor(**self.xgb_params),
                    #n_estimators=300,
                    #max_depth=3,
                    #learning_rate=0.05,
                    #subsample=0.8,
                    #colsample_bytree=0.8,
                    #reg_lambda=1.0,
                    #objective="reg:squarederror",
                    #random_state=rs,
                    #n_jobs=-1,
                #)
            ),
            ]),
        }

class XGBoostTuner:
    """Kleine Randomized Search fuer XGBoost (nur auf den Trainingsdaten)."""

    def __init__(self, config: Config, n_iter: int = 30):
        self.config = config
        self.n_iter = n_iter
        self.search: Optional[RandomizedSearchCV] = None

    def run(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict:
        rs = self.config.random_state

        pipe = Pipeline([
            ("preprocessor", PreprocessorFactory(self.config).tree()),
            # n_jobs=1: die Parallelisierung uebernimmt die Suche
            ("regressor", XGBRegressor(
                objective="reg:squarederror",
                random_state=rs,
                n_jobs=1,
            )),
        ])

        param_dist = {
            "regressor__n_estimators": stats.randint(100, 600),
            "regressor__max_depth": stats.randint(2, 6),            # 2..5
            "regressor__learning_rate": stats.uniform(0.01, 0.14),  # 0.01..0.15
            "regressor__subsample": stats.uniform(0.6, 0.4),        # 0.6..1.0
            "regressor__colsample_bytree": stats.uniform(0.6, 0.4), # 0.6..1.0
            "regressor__min_child_weight": stats.randint(1, 10),
            "regressor__reg_lambda": stats.uniform(0.5, 9.5),       # 0.5..10
        }

        self.search = RandomizedSearchCV(
            pipe,
            param_distributions=param_dist,
            n_iter=self.n_iter,
            scoring="r2",
            cv=KFold(self.config.cv_folds, shuffle=True, random_state=rs),
            random_state=rs,
            n_jobs=-1,
            verbose=1,
        )
        self.search.fit(X_train, y_train)

        best = {
            k.replace("regressor__", ""): v
            for k, v in self.search.best_params_.items()
        }
        print(f"Bester CV-R²: {self.search.best_score_:.4f}")
        print("Beste Parameter:")
        for k, v in best.items():
            print(f"  {k}: {v:.4g}" if isinstance(v, float) else f"  {k}: {v}")
        return best
# ============================================
# 4. Training + Evaluation
# ============================================

class ModelEvaluator:
    """Cross-Validation, Training, Vorhersage und Metriken."""

    def __init__(self, models: Dict[str, Pipeline], data: DataManager, config: Config):
        self.models = models
        self.data = data
        self.config = config
        self.predictions: Dict[str, np.ndarray] = {}
        self.metrics: Dict[str, Dict[str, float]] = {}

    def cross_validate_all(self) -> pd.DataFrame:
        rows = []
        for name, model in self.models.items():
            scores = cross_validate(
                model,
                self.data.X_train,
                self.data.y_train,
                cv=self.config.cv_folds,
                scoring="r2",
            )
            mean, std = scores["test_score"].mean(), scores["test_score"].std()
            print(f"{name}: R² = {mean:.4f} +/- {std:.4f}")
            rows.append({"model": name, "cv_r2_mean": mean, "cv_r2_std": std})
        return pd.DataFrame(rows)

    def fit_all(self) -> None:
        for model in self.models.values():
            model.fit(self.data.X_train, self.data.y_train)

    def predict_all(self) -> None:
        for name, model in self.models.items():
            self.predictions[name] = model.predict(self.data.X_test)

    def evaluate_all(self) -> pd.DataFrame:
        y_test = self.data.y_test
        for name, pred in self.predictions.items():
            mse = mean_squared_error(y_test, pred)
            self.metrics[name] = {
                "MAE": mean_absolute_error(y_test, pred),
                "MSE": mse,
                "RMSE": mse ** 0.5,
                "R2": r2_score(y_test, pred),
            }
            m = self.metrics[name]
            print(f"\n{name}")
            print("-" * 30)
            print(f"MAE:  {m['MAE']:.2f}")
            print(f"MSE:  {m['MSE']:.2f}")
            print(f"RMSE: {m['RMSE']:.2f}")
            print(f"R²:   {m['R2']:.4f}")
        return pd.DataFrame(self.metrics).T.sort_values("R2", ascending=False)


# ============================================
# 5. Modell-Analyse (Koeffizienten / Importances)
# ============================================

class ModelInspector:
    """Liest Koeffizienten und Feature Importances aus Pipelines."""

    @staticmethod
    def coefficients(pipeline: Pipeline) -> pd.DataFrame:
        names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        coefs = pipeline.named_steps["regressor"].coef_
        return (
            pd.DataFrame({"feature": names, "coefficient": coefs})
            .sort_values("coefficient", ascending=False)
            .reset_index(drop=True)
        )

    @staticmethod
    def feature_importance(pipeline: Pipeline) -> pd.DataFrame:
        names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        imp = pipeline.named_steps["regressor"].feature_importances_
        return (
            pd.DataFrame({"feature": names, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )


# ============================================
# 6. Plots
# ============================================

class Plotter:
    """Vergleichsplots der Modelle."""

    def __init__(self, models: Dict[str, Pipeline], data: DataManager, config: Config):
        self.models = models
        self.data = data
        self.config = config

    def plot_r2_comparison(self, metrics: Dict[str, Dict[str, float]]) -> None:
        names = list(metrics.keys())
        r2 = [metrics[n]["R2"] for n in names]

        plt.figure(figsize=(11, 5))
        plt.bar(names, r2)
        plt.ylabel("R²")
        plt.title("Model Comparison (Test R²)")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.show()

    def plot_models_by_feature(
        self,
        feature: str,
        sex: str = "male",
        smoker: str = "no",
        region: str = "southeast",
    ) -> None:
        X = self.data.X
        lo, hi = X[feature].min(), X[feature].max()

        grid = pd.DataFrame({feature: np.linspace(lo, hi, 101)})

        # Uebrige numerische Merkmale auf Mittelwert fixieren
        for f in self.config.numeric_features:
            if f != feature:
                grid[f] = X[f].mean()

        # Kategoriale Merkmale fixieren
        grid["sex"] = sex
        grid["smoker"] = smoker
        grid["region"] = region

        plt.figure(figsize=(12, 7))
        plt.scatter(
            self.data.X_test[feature],
            self.data.y_test,
            alpha=0.4,
            label="Actual data",
        )
        for name, model in self.models.items():
            plt.plot(grid[feature], model.predict(grid), linewidth=2, label=name)

        plt.xlabel(feature)
        plt.ylabel("Charges")
        plt.title(
            f"Regression Models: {feature} vs Charges "
            f"(sex={sex}, smoker={smoker}, region={region})"
        )
        plt.legend()
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_importance(df: pd.DataFrame, title: str, top_n: int = 10) -> None:
        top = df.head(top_n).iloc[::-1]
        plt.figure(figsize=(8, 5))
        plt.barh(top["feature"], top["importance"])
        plt.title(title)
        plt.tight_layout()
        plt.show()


# ============================================
# 7. Annahmen der linearen Regression
# ============================================

class AssumptionChecker:
    """Prueft die Annahmen der linearen Regression (Plots + Tests)."""

    def __init__(self, model: Pipeline, data: DataManager, config: Config):
        self.model = model
        self.data = data
        self.config = config
        self.fitted = model.predict(data.X_test)
        self.residuals = data.y_test - self.fitted

    def plot_linearity(self) -> None:
        df = self.data.df
        feats = self.config.numeric_features
        fig, axes = plt.subplots(1, len(feats), figsize=(6 * len(feats), 5))

        for ax, feature in zip(np.atleast_1d(axes), feats):
            ax.scatter(df[feature], df[self.config.target], alpha=0.4)
            sns.regplot(
                x=df[feature],
                y=df[self.config.target],
                scatter=False,
                lowess=True,
                color="red",
                ax=ax,
            )
            ax.set_title(f"{feature} vs charges")
            ax.grid(True)

        plt.suptitle("Linearität: Scatter Plots (rote Linie = LOWESS)")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df, x="age", y=self.config.target, hue="smoker", alpha=0.6)
        plt.title("age vs charges, nach smoker")
        plt.grid(True)
        plt.show()

    def plot_residuals_vs_fitted(self) -> None:
        plt.figure(figsize=(8, 5))
        plt.scatter(self.fitted, self.residuals, alpha=0.5)
        plt.axhline(0, color="red", linestyle="--")
        low = sm.nonparametric.lowess(self.residuals, self.fitted, frac=0.3)
        plt.plot(low[:, 0], low[:, 1], color="orange", linewidth=2)
        plt.xlabel("Fitted values (Vorhersage)")
        plt.ylabel("Residuals")
        plt.title("Residuals vs Fitted")
        plt.grid(True)
        plt.show()

    def plot_qq(self) -> None:
        fig, ax = plt.subplots(figsize=(6, 6))
        sm.qqplot(self.residuals, line="45", fit=True, ax=ax)
        ax.set_title("Q-Q Plot der Residuen")
        ax.grid(True)
        plt.show()

    def plot_residual_hist(self) -> None:
        plt.figure(figsize=(8, 5))
        sns.histplot(self.residuals, kde=True, bins=30)
        plt.xlabel("Residuals")
        plt.title("Verteilung der Residuen")
        plt.grid(True)
        plt.show()

    def plot_scale_location(self) -> None:
        std_res = (self.residuals - self.residuals.mean()) / self.residuals.std()
        sqrt_abs = np.sqrt(np.abs(std_res))

        plt.figure(figsize=(8, 5))
        plt.scatter(self.fitted, sqrt_abs, alpha=0.5)
        low = sm.nonparametric.lowess(sqrt_abs, self.fitted, frac=0.3)
        plt.plot(low[:, 0], low[:, 1], color="red", linewidth=2)
        plt.xlabel("Fitted values")
        plt.ylabel("√|standardisierte Residuen|")
        plt.title("Scale-Location Plot")
        plt.grid(True)
        plt.show()

    def run_tests(self) -> None:
        # Shapiro-Wilk: p < 0.05 -> Residuen NICHT normalverteilt
        _, shapiro_p = stats.shapiro(self.residuals)
        print(f"Shapiro-Wilk: p = {shapiro_p:.4g}")

        # Breusch-Pagan: p < 0.05 -> Heteroskedastizitaet
        X_t = self.model.named_steps["preprocessor"].transform(self.data.X_test)
        if hasattr(X_t, "toarray"):
            X_t = X_t.toarray()
        bp = het_breuschpagan(self.residuals, sm.add_constant(X_t))
        print(f"Breusch-Pagan: p = {bp[1]:.4g}")

        # Durbin-Watson: ~2 = keine Autokorrelation
        print(f"Durbin-Watson: {durbin_watson(self.residuals):.3f}")

    def run_all(self) -> None:
        self.plot_linearity()
        self.plot_residuals_vs_fitted()
        self.plot_qq()
        self.plot_residual_hist()
        self.plot_scale_location()
        self.run_tests()