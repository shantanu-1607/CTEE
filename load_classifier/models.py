"""A small, declared set of classifiers for recording summaries."""

from sklearn.dummy import DummyClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .features import STATIC_FEATURES


def pipeline(classifier, scale=False):
    steps = [("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline([*steps, ("classifier", classifier)])


def candidates(seed=42):
    models = {
        "majority": pipeline(DummyClassifier(strategy="most_frequent")),
        "logistic": pipeline(LogisticRegression(C=1, class_weight="balanced", max_iter=3000), scale=True),
        "svm_c1": pipeline(SVC(C=1, kernel="rbf", class_weight="balanced", probability=True, random_state=seed), scale=True),
        "svm_c10": pipeline(SVC(C=10, kernel="rbf", class_weight="balanced", probability=True, random_state=seed), scale=True),
        "random_forest": pipeline(RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                  max_features="sqrt", class_weight="balanced", random_state=seed, n_jobs=4)),
        "extra_trees": pipeline(ExtraTreesClassifier(n_estimators=400, min_samples_leaf=1,
                                max_features="sqrt", class_weight="balanced", random_state=seed, n_jobs=4)),
        "extra_trees_leaf3": pipeline(ExtraTreesClassifier(n_estimators=400, min_samples_leaf=3,
                                      max_features="sqrt", class_weight="balanced", random_state=seed, n_jobs=4)),
        "extra_trees_half": pipeline(ExtraTreesClassifier(n_estimators=400, min_samples_leaf=1,
                                    max_features=.5, class_weight="balanced", random_state=seed, n_jobs=4)),
        "extra_trees_all": pipeline(ExtraTreesClassifier(n_estimators=400, min_samples_leaf=1,
                                   max_features=1., class_weight="balanced", random_state=seed, n_jobs=4)),
        "gradient_boosting": pipeline(HistGradientBoostingClassifier(max_iter=200, max_leaf_nodes=15,
                                      l2_regularization=1, learning_rate=.06, early_stopping=False,
                                      class_weight="balanced", random_state=seed)),
        "mlp": pipeline(MLPClassifier(hidden_layer_sizes=(64, 32), alpha=.01, batch_size=64,
                        max_iter=600, early_stopping=False, random_state=seed), scale=True),
    }
    temporal = ("i_rms", "P_real_W", "pf", "sin_phi", "i_crest_factor", "i_thd")
    selected = [*STATIC_FEATURES, *[f"{name}__{stat}" for name in temporal
                                  for stat in ("early", "late", "slope", "step")]]
    focused = pipeline(ExtraTreesClassifier(n_estimators=400, max_features=.5,
                       class_weight="balanced", random_state=seed, n_jobs=4))
    models["extra_trees_compact"] = Pipeline([
        ("select", ColumnTransformer([("features", "passthrough", selected)], remainder="drop")),
        *focused.steps,
    ])
    return models
