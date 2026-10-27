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