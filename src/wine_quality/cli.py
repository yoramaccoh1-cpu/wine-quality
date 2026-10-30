import argparse
import json

import pandas as pd

from wine_quality.data import DatasetLoader
from wine_quality.evaluate import evaluate_test
from wine_quality.models import Trainer, load_model


def train(config_path: str) -> None:
    """Load the data, compare the models, tune the best ones and save the winner."""
    loader = DatasetLoader(config_path)
    loader.load()
    X_train, _, y_train, _ = loader.split()

    trainer = Trainer(config_path)
    print(trainer.compare(X_train, y_train))
    trainer.tune(X_train, y_train)
    path = trainer.save()
    print(f"Model saved to {path}")


def evaluate(config_path: str) -> None:
    """Reload the saved model and print its F1 and ROC-AUC on the test set."""
    loader = DatasetLoader(config_path)
    loader.load()
    _, X_test, _, y_test = loader.split()

    model = load_model(loader.cfg["output"]["model_path"])
    print(evaluate_test(model, X_test, y_test))


def predict(config_path: str, input_path: str) -> None:
    """Read one wine from a JSON file and print whether the model says it is good."""
    loader = DatasetLoader(config_path)
    model = load_model(loader.cfg["output"]["model_path"])

    with open(input_path) as f:
        wine = json.load(f)
    X_new = pd.DataFrame([wine])

    proba = model.predict_proba(X_new)[0, 1]
    label = "good" if model.predict(X_new)[0] == 1 else "not good"
    print(f"Prediction: {label} (P(good) = {proba:.2f})")


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser with train, evaluate and predict."""
    parser = argparse.ArgumentParser(
        prog="wine-quality", description="Wine quality classifier"
    )
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("train", help="compare, tune and save the best model")
    sub.add_parser("evaluate", help="evaluate the saved model on the test set")
    predict_parser = sub.add_parser("predict", help="predict one wine from a JSON file")
    predict_parser.add_argument("--input", required=True, help="JSON file with one wine")
    return parser


def main() -> None:
    """Run the command chosen on the command line."""
    args = build_parser().parse_args()
    if args.command == "train":
        train(args.config)
    elif args.command == "evaluate":
        evaluate(args.config)
    else:
        predict(args.config, args.input)


if __name__ == "__main__":
    main()