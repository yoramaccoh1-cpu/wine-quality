from wine_quality.data import DatasetLoader


def test_merge_row_count():
    loader = DatasetLoader()
    loader.load()
    assert len(loader.df) == 5320


def test_no_train_test_overlap():
    loader = DatasetLoader()
    loader.load()
    loader.split()
    common = set(loader.X_train.index) & set(loader.X_test.index)
    assert len(common) == 0


def test_label_is_binary():
    loader = DatasetLoader()
    loader.load()
    assert set(loader.df["good"].unique()) == {0, 1}


def test_split_is_stratified():
    loader = DatasetLoader()
    loader.load()
    loader.split()
    train_share = loader.y_train.mean()
    test_share = loader.y_test.mean()
    assert abs(train_share - test_share) < 0.005


