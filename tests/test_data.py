"""Tests for data loading, label creation and the train/test split."""

from wine_quality.data import DatasetLoader

EXPECTED_ROWS = 5320  # 6497 wines minus 1177 exact duplicates
MAX_SHARE_GAP = 0.005  # max gap in share of good wines, train vs test


def test_merge_row_count() -> None:
    """Merging red and white then dropping duplicates gives 5320 rows."""
    loader = DatasetLoader()
    loader.load()
    assert len(loader.df) == EXPECTED_ROWS


def test_no_train_test_overlap() -> None:
    """No row index appears in both the train set and the test set."""
    loader = DatasetLoader()
    loader.load()
    loader.split()
    common = set(loader.X_train.index) & set(loader.X_test.index)
    assert len(common) == 0


def test_label_is_binary() -> None:
    """The good label only contains the values 0 and 1."""
    loader = DatasetLoader()
    loader.load()
    assert set(loader.df["good"].unique()) == {0, 1}


def test_split_is_stratified() -> None:
    """Train and test have almost the same share of good wines."""
    loader = DatasetLoader()
    loader.load()
    loader.split()
    gap = abs(loader.y_train.mean() - loader.y_test.mean())
    assert gap < MAX_SHARE_GAP