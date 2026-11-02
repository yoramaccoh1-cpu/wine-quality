import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def _load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class DatasetLoader:
    """Load, clean and split the Wine Quality dataset."""
    def __init__(self, config_path: str = "config.yaml") -> None:
        self.cfg = _load_config(config_path)
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load(self) -> pd.DataFrame:
        """Download red and white from UCI, merge, clean and create the binary label."""
        url_red = self.cfg["data"]["red_url"]
        url_white = self.cfg["data"]["white_url"]
        red = pd.read_csv(url_red, sep=";")
        white = pd.read_csv(url_white, sep=";")
        red["wine_type"] = "red"
        white["wine_type"] = "white"
        df = pd.concat([red, white], ignore_index=True)
        df = self._clean(df)
        df = self._make_label(df)
        self.df = df
        return df

    def _clean(self, df):
        df = df.drop_duplicates()
        return df

    def _make_label(self, df):
        cutoff = self.cfg["model"]["good_quality_cutoff"]
        df = df.copy()
        df["good"] = (df["quality"] >= cutoff).astype(int)
        return df


    def split(self) -> tuple:
        """Stratified 80/20 train/test split. Call after load()."""
        X = self.df.drop(columns=["quality", "good"])
        y = self.df["good"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.cfg["model"]["test_size"],
            stratify=y,
            random_state=self.cfg["model"]["random_state"]
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

