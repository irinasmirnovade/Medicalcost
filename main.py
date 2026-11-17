"""
main.py
Einstiegspunkt: steuert den Ablauf, die Logik liegt in logic.py.

Aufruf:
    python main.py

"""

import argparse

import pandas as pd

from logic import (
    Config,
    DataManager,
    ModelFactory,
    ModelEvaluator,
    ModelInspector,
    Plotter,
    AssumptionChecker,
    XGBoostTuner
)

# DEFAULT_PATH = r"C:\Users\irina\Documents\Innowise\Code\Dataset\insurance.csv"
DEFAULT_PATH = "raw_data/insurance.csv"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Insurance charges: Modellvergleich")
    parser.add_argument("--data", default=DEFAULT_PATH, help="Pfad zur insurance.csv")
    parser.add_argument("--no-tune", action="store_true", help="XGBoost-Suche ueberspringen")
    parser.add_argument("--no-plots", action="store_true", help="Plots ueberspringen")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pd.set_option("display.width", 120)

    config = Config()

    # 1. Daten
    data = DataManager(config).load(args.data).split()

    # 2. Modelle
    #models = ModelFactory(config).build_all()
    # 2. XGBoost-Hyperparameter-Suche (nur Trainingsdaten)
    xgb_params = None
    if not args.no_tune:
        print("=== XGBoost Hyperparameter-Suche ===")
        xgb_params = XGBoostTuner(config).run(data.X_train, data.y_train)

    # 3. Modelle
    models = ModelFactory(config, xgb_params).build_all()
    evaluator = ModelEvaluator(models, data, config)

    # 3. Cross Validation
    print("=== Cross Validation (Train) ===")
    evaluator.cross_validate_all()

    # 4. Training
    evaluator.fit_all()

    # 5. Koeffizienten / Importances
    print("\nRidge coefficients:")
    print(ModelInspector.coefficients(models["Ridge Regression"]))

    rf_imp = ModelInspector.feature_importance(models["Random Forest"])
    xgb_imp = ModelInspector.feature_importance(models["XGBoost"])
    print("\nRandom Forest feature importance:")
    print(rf_imp)
    print("\nXGBoost feature importance:")
    print(xgb_imp)

    # 6. Vorhersage + Evaluation (Test)
    print("\n=== Test-Set Evaluation ===")
    evaluator.predict_all()
    summary = evaluator.evaluate_all()
    print("\n=== Zusammenfassung (sortiert nach R²) ===")
    print(summary.round(4))

    if args.no_plots:
        return

    # 7. Plots
    plotter = Plotter(models, data, config)
    plotter.plot_r2_comparison(evaluator.metrics)
    plotter.plot_importance(rf_imp, "Random Forest: Feature Importance")
    plotter.plot_importance(xgb_imp, "XGBoost: Feature Importance")

    for feature in ["age", "bmi", "children"]:
        plotter.plot_models_by_feature(feature)

    # 8. Annahmen der linearen Regression
    checker = AssumptionChecker(models["Linear Regression"], data, config)
    checker.run_all()


if __name__ == "__main__":
    main()