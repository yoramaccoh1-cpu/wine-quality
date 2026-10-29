import pandas as pd
import yaml
import joblib
import pathlib
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from wine_quality.features import build_preprocessor


def _load_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


class Trainer:
    def __init__(self, config_path="config.yaml"):
        self.cfg = _load_config(config_path)
        self.best_pipeline = None
        self.cv_results = None
        self._cv = StratifiedKFold(
            n_splits=self.cfg["model"]["cv_folds"],
            shuffle=True,
            random_state=self.cfg["model"]["random_state"]
        )

    def make_pipeline(self, classifier) -> Pipeline:
        """création du pipeline avec le preprocessor et un classifier """
        return Pipeline([
            ("prep", build_preprocessor()),
            ("clf", classifier)
        ])

    def compare(self, X_train, y_train)-> pd.DataFrame:
        """Comparaison de 4 modèles sous Cross validation et donne un tableau F1/ROC-AUC """
        candidates = {
            "Dummy": DummyClassifier(strategy="most_frequent"),
            "logistic regression": LogisticRegression(max_iter=1000),
            "k-NN": KNeighborsClassifier(n_neighbors=15),
            "linear SVM": LinearSVC(C=1),
        }
        rows = {}
        for name, clf in candidates.items():
            out = cross_validate(
                self.make_pipeline(clf),
                X_train, y_train,
                cv=self._cv,
                scoring=["f1", "roc_auc"]
            )
            rows[name] = {
                "f1": round(out["test_f1"].mean(), 3),
                "roc_auc": round(out["test_roc_auc"].mean(), 3),
            }
        self.cv_results = pd.DataFrame(rows).T.sort_values("f1", ascending=False)
        return self.cv_results

    def tune(self, X_train, y_train) -> Pipeline:
        """GridSearchCV appliqué sur LogReg et kNN il retourne le meilleur pipeline des 4."""
        rs = self.cfg["model"]["random_state"]

        searches = {
            "logistic regression": GridSearchCV(
                self.make_pipeline(LogisticRegression(max_iter=1000, random_state=rs)),
                param_grid={"clf__C": self.cfg["grids"]["logistic_regression"]["C"]},
                cv=self._cv,
                scoring="f1"
            ),
            "k-NN": GridSearchCV(
                self.make_pipeline(KNeighborsClassifier()),
                param_grid={"clf__n_neighbors": self.cfg["grids"]["knn"]["n_neighbors"]},
                cv=self._cv,
                scoring="f1"
            ),
        }
        best_score = -1
        best_pipeline = None

        for name, search in searches.items():
            search.fit(X_train, y_train)
            print(f"{name}: meilleurs params = {search.best_params_}, F1 = {search.best_score_:.3f}")
            if search.best_score_ > best_score:
                best_score = search.best_score_
                best_pipeline = search.best_estimator_

        self.best_pipeline = best_pipeline
        return self.best_pipeline

    def save(self) -> pathlib.Path:
        """sauvegarde le meilleur pipeline avec joblib """
        path = pathlib.Path(self.cfg["output"]["model_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.best_pipeline, path)
        return path

def load_model(path: str) -> Pipeline:
    """Recharge le modèle sauvegardé par save() """
    return joblib.load(path)