"""Pipelines de los tres modelos (regresión lineal, Random Forest, CatBoost) y utilidades comunes."""
from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from read_csv import CAT_FEATURES, NUM_FEATURES

RANDOM_STATE = 42


class CatBoostSK(BaseEstimator, RegressorMixin):
    """Envoltorio mínimo para que CatBoost funcione con clone/cross_validate de scikit-learn."""

    def __init__(self, cat_cols=None, iterations=600, learning_rate=0.08, depth=6, random_state=RANDOM_STATE):
        self.cat_cols = cat_cols
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.random_state = random_state

    def fit(self, X, y):
        self.model_ = CatBoostRegressor(
            iterations=self.iterations, learning_rate=self.learning_rate, depth=self.depth,
            loss_function="RMSE", random_seed=self.random_state, cat_features=list(self.cat_cols or []),
            verbose=0, allow_writing_files=False,
        )
        self.model_.fit(X, y)
        self.feature_names_ = list(X.columns)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def feature_importance(self):
        return dict(zip(self.feature_names_, self.model_.get_feature_importance()))


def _preprocessor(num_cols, cat_cols, scale):
    num_steps = [SimpleImputer(strategy="median", add_indicator=True)]
    cat_steps = [OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30, sparse_output=False)]
    if scale:  # variables estandarizadas -> coeficientes comparables entre sí
        num_steps.append(StandardScaler())
        cat_steps.append(StandardScaler())
    transformers = [("num", make_pipeline(*num_steps), list(num_cols))]
    if cat_cols:
        transformers.append(("cat", make_pipeline(*cat_steps), list(cat_cols)))
    return ColumnTransformer(transformers)


def build_models(num_cols=NUM_FEATURES, cat_cols=CAT_FEATURES):
    """Devuelve {nombre: estimador}. Cada llamada crea modelos nuevos sin entrenar."""
    return {
        "Regresión lineal": Pipeline([
            ("prep", _preprocessor(num_cols, cat_cols, scale=True)),
            ("model", LinearRegression()),
        ]),
        "Random Forest": Pipeline([
            ("prep", _preprocessor(num_cols, cat_cols, scale=False)),
            ("model", RandomForestRegressor(n_estimators=200, min_samples_leaf=3, max_features=0.33,
                                            random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "CatBoost": CatBoostSK(cat_cols=list(cat_cols)),
    }


def original_variable(feature_name, cat_cols=CAT_FEATURES):
    """Variable original a la que pertenece una columna transformada (dummy, indicador de nulo, etc.)."""
    name = feature_name.split("__", 1)[1]
    if name.startswith("missingindicator_"):
        return name.removeprefix("missingindicator_")
    for col in cat_cols:
        if name.startswith(col + "_"):
            return col
    return name