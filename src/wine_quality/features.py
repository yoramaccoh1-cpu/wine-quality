from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM_FEATURES = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol"]
CAT_FEATURES = ["wine_type"]

def prep_features() -> ColumnTransformer:
    """Retourne un ColumnTransformer qui fait StandardScaler sur les numériques, OneHotEncoder sur wine_type."""
    return ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ])

