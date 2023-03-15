# Author: CJM <455932903@qq.com>
# Demo of this app https://share.streamlit.io/dataprofessor/ml-auto-app/main/app.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.utils import all_estimators
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    f1_score,
    r2_score,
    mean_squared_error,
)
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
import numpy as np
from tqdm import tqdm
import time
import warnings
import xgboost
import lightgbm

warnings.filterwarnings("ignore")
pd.set_option("display.precision", 2)
pd.set_option("display.float_format", lambda x: "%.2f" % x)

removed_classifiers = [
    "ClassifierChain",
    "ComplementNB",
    "GradientBoostingClassifier",
    "GaussianProcessClassifier",
    "HistGradientBoostingClassifier",
    "MLPClassifier",
    "LogisticRegressionCV",
    "MultiOutputClassifier",
    "MultinomialNB",
    "OneVsOneClassifier",
    "OneVsRestClassifier",
    "OutputCodeClassifier",
    "RadiusNeighborsClassifier",
    "VotingClassifier",
]

removed_regressors = [
    "TheilSenRegressor",
    "ARDRegression",
    "CCA",
    "IsotonicRegression",
    "StackingRegressor",
    "MultiOutputRegressor",
    "MultiTaskElasticNet",
    "MultiTaskElasticNetCV",
    "MultiTaskLasso",
    "MultiTaskLassoCV",
    "PLSCanonical",
    "PLSRegression",
    "RadiusNeighborsRegressor",
    "RegressorChain",
    "VotingRegressor",
]

CLASSIFIERS = [
    est
    for est in all_estimators()
    if (issubclass(est[1], ClassifierMixin) and (est[0] not in removed_classifiers))
]

REGRESSORS = [
    est
    for est in all_estimators()
    if (issubclass(est[1], RegressorMixin) and (est[0] not in removed_regressors))
]

REGRESSORS.append(("XGBRegressor", xgboost.XGBRegressor))
REGRESSORS.append(("LGBMRegressor", lightgbm.LGBMRegressor))


CLASSIFIERS.append(("XGBClassifier", xgboost.XGBClassifier))
CLASSIFIERS.append(("LGBMClassifier", lightgbm.LGBMClassifier))

#Pipeline数据预处理
# 连续型变量，SimpleImputer 填充缺失值,策略为平均值填充；StandardScaler对数据进行标准化
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
)

#类别变量独热编码
categorical_transformer_low = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoding", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ]
)

categorical_transformer_high = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        # 'OrdianlEncoder' Raise a ValueError when encounters an unknown value. Check https://github.com/scikit-learn/scikit-learn/pull/13423
        ("encoding", OrdinalEncoder()),
    ]
)


# Helper function


def get_card_split(df, cols, n=11):
    """
    Splits categorical columns into 2 lists based on cardinality (i.e # of unique values)
    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame from which the cardinality of the columns is calculated.
    cols : list-like
        Categorical columns to list
    n : int, optional (default=11)
        The value of 'n' will be used to split columns.
    Returns
    -------
    card_low : list-like
        Columns with cardinality < n
    card_high : list-like
        Columns with cardinality >= n
    """
    cond = df[cols].nunique() > n
    card_high = cols[cond]
    card_low = cols[~cond]
    return card_low, card_high


# Helper class for performing classification


class FastClassifier:
    """
    本模块使用Scikit-learn中所有分类算法
    ----------
    verbose : int, optional (default=0)
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.
    ignore_warnings : bool, optional (default=True)
        When set to True, the warning related to algorigms that are not able to run are ignored.
    custom_metric : function, optional (default=None)
        When function is provided, models are evaluated based on the custom evaluation metric provided.
    prediction : bool, optional (default=False)
        When set to True, the predictions of all the models models are returned as dataframe.
    classifiers : list, optional (default="all")
        When function is provided, trains the chosen classifier(s).

    Examples
    --------
    >>> from sklearn.datasets import load_breast_cancer
    >>> from sklearn.model_selection import train_test_split
    >>> data = load_breast_cancer()
    >>> X = data.data
    >>> y= data.target
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.5,random_state =123)
    >>> clf = FastClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
    >>> models,predictions = clf.fit(X_train, X_test, y_train, y_test)
    >>> model_dictionary = clf.provide_models(X_train,X_test,y_train,y_test)
    >>> models
    | Model                          |   Accuracy |   Balanced Accuracy |   ROC AUC |   F1 Score |   Time Taken |
    |:-------------------------------|-----------:|--------------------:|----------:|-----------:|-------------:|
    | LinearSVC                      |   0.989474 |            0.987544 |  0.987544 |   0.989462 |    0.0150008 |
    | SGDClassifier                  |   0.989474 |            0.987544 |  0.987544 |   0.989462 |    0.0109992 |
    | MLPClassifier                  |   0.985965 |            0.986904 |  0.986904 |   0.985994 |    0.426     |
    | Perceptron                     |   0.985965 |            0.984797 |  0.984797 |   0.985965 |    0.0120046 |
    | LogisticRegression             |   0.985965 |            0.98269  |  0.98269  |   0.985934 |    0.0200036 |
    | LogisticRegressionCV           |   0.985965 |            0.98269  |  0.98269  |   0.985934 |    0.262997  |
    | SVC                            |   0.982456 |            0.979942 |  0.979942 |   0.982437 |    0.0140011 |
    | CalibratedClassifierCV         |   0.982456 |            0.975728 |  0.975728 |   0.982357 |    0.0350015 |
    | PassiveAggressiveClassifier    |   0.975439 |            0.974448 |  0.974448 |   0.975464 |    0.0130005 |
    | LabelPropagation               |   0.975439 |            0.974448 |  0.974448 |   0.975464 |    0.0429988 |
    | LabelSpreading                 |   0.975439 |            0.974448 |  0.974448 |   0.975464 |    0.0310006 |
    | RandomForestClassifier         |   0.97193  |            0.969594 |  0.969594 |   0.97193  |    0.033     |
    | GradientBoostingClassifier     |   0.97193  |            0.967486 |  0.967486 |   0.971869 |    0.166998  |
    | QuadraticDiscriminantAnalysis  |   0.964912 |            0.966206 |  0.966206 |   0.965052 |    0.0119994 |
    | HistGradientBoostingClassifier |   0.968421 |            0.964739 |  0.964739 |   0.968387 |    0.682003  |
    | RidgeClassifierCV              |   0.97193  |            0.963272 |  0.963272 |   0.971736 |    0.0130029 |
    | RidgeClassifier                |   0.968421 |            0.960525 |  0.960525 |   0.968242 |    0.0119977 |
    | AdaBoostClassifier             |   0.961404 |            0.959245 |  0.959245 |   0.961444 |    0.204998  |
    | ExtraTreesClassifier           |   0.961404 |            0.957138 |  0.957138 |   0.961362 |    0.0270066 |
    | KNeighborsClassifier           |   0.961404 |            0.95503  |  0.95503  |   0.961276 |    0.0560005 |
    | BaggingClassifier              |   0.947368 |            0.954577 |  0.954577 |   0.947882 |    0.0559971 |
    | BernoulliNB                    |   0.950877 |            0.951003 |  0.951003 |   0.951072 |    0.0169988 |
    | LinearDiscriminantAnalysis     |   0.961404 |            0.950816 |  0.950816 |   0.961089 |    0.0199995 |
    | GaussianNB                     |   0.954386 |            0.949536 |  0.949536 |   0.954337 |    0.0139935 |
    | NuSVC                          |   0.954386 |            0.943215 |  0.943215 |   0.954014 |    0.019989  |
    | DecisionTreeClassifier         |   0.936842 |            0.933693 |  0.933693 |   0.936971 |    0.0170023 |
    | NearestCentroid                |   0.947368 |            0.933506 |  0.933506 |   0.946801 |    0.0160074 |
    | ExtraTreeClassifier            |   0.922807 |            0.912168 |  0.912168 |   0.922462 |    0.0109999 |
    | CheckingClassifier             |   0.361404 |            0.5      |  0.5      |   0.191879 |    0.0170043 |
    | DummyClassifier                |   0.512281 |            0.489598 |  0.489598 |   0.518924 |    0.0119965 |
    """

    def __init__(
        self,
        verbose=0,
        ignore_warnings=True,
        custom_metric=None,
        predictions=False,
        random_state=42,
        classifiers="all",
    ):
        self.verbose = verbose
        self.ignore_warnings = ignore_warnings
        self.custom_metric = custom_metric
        self.predictions = predictions
        self.models = {}
        self.random_state = random_state
        self.classifiers = classifiers

    def fit(self, X_train, X_test, y_train, y_test):
        """Fit Classification algorithms to X_train and y_train, predict and score on X_test, y_test.
        Parameters
        ----------
        X_train : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        X_test : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        y_train : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        y_test : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        Returns
        -------
        scores : Pandas DataFrame
            Returns metrics of all the models in a Pandas DataFrame.
        predictions : Pandas DataFrame
            Returns predictions of all the models in a Pandas DataFrame.
        """
        Accuracy = []
        B_Accuracy = []
        ROC_AUC = []
        F1 = []
        names = []
        TIME = []
        predictions = {}

        if self.custom_metric is not None:
            CUSTOM_METRIC = []

        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)

        numeric_features = X_train.select_dtypes(include=[np.number]).columns
        categorical_features = X_train.select_dtypes(include=["object"]).columns

        categorical_low, categorical_high = get_card_split(
            X_train, categorical_features
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, numeric_features),
                ("categorical_low", categorical_transformer_low, categorical_low),
                ("categorical_high", categorical_transformer_high, categorical_high),
            ]
        )

        if self.classifiers == "all":
            self.classifiers = CLASSIFIERS
        else:
            try:
                temp_list = []
                for classifier in self.classifiers:
                    full_name = (classifier.__name__, classifier)
                    temp_list.append(full_name)
                self.classifiers = temp_list
            except Exception as exception:
                print(exception)
                print("Invalid Classifier(s)")

        for name, model in tqdm(self.classifiers):
            start = time.time()
            try:
                if "random_state" in model().get_params().keys():
                    pipe = Pipeline(
                        steps=[
                            ("preprocessor", preprocessor),
                            ("classifier", model(random_state=self.random_state)),
                        ]
                    )
                else:
                    pipe = Pipeline(
                        steps=[("preprocessor", preprocessor), ("classifier", model())]
                    )

                pipe.fit(X_train, y_train)
                self.models[name] = pipe
                y_pred = pipe.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred, normalize=True)
                b_accuracy = balanced_accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="weighted")
                try:
                    roc_auc = roc_auc_score(y_test, y_pred)
                except Exception as exception:
                    roc_auc = None
                    if self.ignore_warnings is False:
                        print("ROC AUC couldn't be calculated for " + name)
                        print(exception)
                names.append(name)
                Accuracy.append(accuracy)
                B_Accuracy.append(b_accuracy)
                ROC_AUC.append(roc_auc)
                F1.append(f1)
                TIME.append(time.time() - start)
                if self.custom_metric is not None:
                    custom_metric = self.custom_metric(y_test, y_pred)
                    CUSTOM_METRIC.append(custom_metric)
                if self.verbose > 0:
                    if self.custom_metric is not None:
                        print(
                            {
                                "Model": name,
                                "Accuracy": accuracy,
                                "Balanced Accuracy": b_accuracy,
                                "ROC AUC": roc_auc,
                                "F1 Score": f1,
                                self.custom_metric.__name__: custom_metric,
                                "Time taken": time.time() - start,
                            }
                        )
                    else:
                        print(
                            {
                                "Model": name,
                                "Accuracy": accuracy,
                                "Balanced Accuracy": b_accuracy,
                                "ROC AUC": roc_auc,
                                "F1 Score": f1,
                                "Time taken": time.time() - start,
                            }
                        )
                if self.predictions:
                    predictions[name] = y_pred
            except Exception as exception:
                if self.ignore_warnings is False:
                    print(name + " model failed to execute")
                    print(exception)
        if self.custom_metric is None:
            scores = pd.DataFrame(
                {
                    "Model": names,
                    "Accuracy": Accuracy,
                    "Balanced Accuracy": B_Accuracy,
                    "ROC AUC": ROC_AUC,
                    "F1 Score": F1,
                    "Time Taken": TIME,
                }
            )
        else:
            scores = pd.DataFrame(
                {
                    "Model": names,
                    "Accuracy": Accuracy,
                    "Balanced Accuracy": B_Accuracy,
                    "ROC AUC": ROC_AUC,
                    "F1 Score": F1,
                    self.custom_metric.__name__: CUSTOM_METRIC,
                    "Time Taken": TIME,
                }
            )
        scores = scores.sort_values(by="Balanced Accuracy", ascending=False).set_index(
            "Model"
        )

        if self.predictions:
            predictions_df = pd.DataFrame.from_dict(predictions)
        return scores, predictions_df if self.predictions is True else scores

    def provide_models(self, X_train, X_test, y_train, y_test):
        """
        This function returns all the model objects trained in fit function.
        If fit is not called already, then we call fit and then return the models.
        Parameters
        ----------
        X_train : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        X_test : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        y_train : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        y_test : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        Returns
        -------
        models: dict-object,
            Returns a dictionary with each model pipeline as value
            with key as name of models.
        """
        if len(self.models.keys()) == 0:
            self.fit(X_train, X_test, y_train, y_test)

        return self.models


def adjusted_rsquared(r2, n, p):
    return 1 - (1 - r2) * ((n - 1) / (n - p - 1))

class FastRegressor:
    """
    本模块使用Scikit-learn中所有回归算法
    ----------
    verbose : int, optional (default=0)
        For the liblinear and lbfgs solvers set verbose to any positive
        number for verbosity.
    ignore_warnings : bool, optional (default=True)
        When set to True, the warning related to algorigms that are not able to run are ignored.
    custom_metric : function, optional (default=None)
        When function is provided, models are evaluated based on the custom evaluation metric provided.
    prediction : bool, optional (default=False)
        When set to True, the predictions of all the models models are returned as dataframe.
    regressors : list, optional (default="all")
        When function is provided, trains the chosen regressor(s).

    Examples
    --------
    >>> from sklearn import datasets
    >>> from sklearn.utils import shuffle
    >>> import numpy as np

    >>> boston = datasets.load_boston()
    >>> X, y = shuffle(boston.data, boston.target, random_state=13)
    >>> X = X.astype(np.float32)

    >>> offset = int(X.shape[0] * 0.9)
    >>> X_train, y_train = X[:offset], y[:offset]
    >>> X_test, y_test = X[offset:], y[offset:]

    >>> reg = FastRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
    >>> models, predictions = reg.fit(X_train, X_test, y_train, y_test)
    >>> model_dictionary = reg.provide_models(X_train, X_test, y_train, y_test)
    >>> models
    | Model                         | Adjusted R-Squared | R-Squared |  RMSE | Time Taken |
    |:------------------------------|-------------------:|----------:|------:|-----------:|
    | SVR                           |               0.83 |      0.88 |  2.62 |       0.01 |
    | BaggingRegressor              |               0.83 |      0.88 |  2.63 |       0.03 |
    | NuSVR                         |               0.82 |      0.86 |  2.76 |       0.03 |
    | RandomForestRegressor         |               0.81 |      0.86 |  2.78 |       0.21 |
    | XGBRegressor                  |               0.81 |      0.86 |  2.79 |       0.06 |
    | GradientBoostingRegressor     |               0.81 |      0.86 |  2.84 |       0.11 |
    | ExtraTreesRegressor           |               0.79 |      0.84 |  2.98 |       0.12 |
    | AdaBoostRegressor             |               0.78 |      0.83 |  3.04 |       0.07 |
    | HistGradientBoostingRegressor |               0.77 |      0.83 |  3.06 |       0.17 |
    | PoissonRegressor              |               0.77 |      0.83 |  3.11 |       0.01 |
    | LGBMRegressor                 |               0.77 |      0.83 |  3.11 |       0.07 |
    | KNeighborsRegressor           |               0.77 |      0.83 |  3.12 |       0.01 |
    | DecisionTreeRegressor         |               0.65 |      0.74 |  3.79 |       0.01 |
    | MLPRegressor                  |               0.65 |      0.74 |  3.80 |       1.63 |
    | HuberRegressor                |               0.64 |      0.74 |  3.84 |       0.01 |
    | GammaRegressor                |               0.64 |      0.73 |  3.88 |       0.01 |
    | LinearSVR                     |               0.62 |      0.72 |  3.96 |       0.01 |
    | RidgeCV                       |               0.62 |      0.72 |  3.97 |       0.01 |
    | BayesianRidge                 |               0.62 |      0.72 |  3.97 |       0.01 |
    | Ridge                         |               0.62 |      0.72 |  3.97 |       0.01 |
    | TransformedTargetRegressor    |               0.62 |      0.72 |  3.97 |       0.01 |
    | LinearRegression              |               0.62 |      0.72 |  3.97 |       0.01 |
    | ElasticNetCV                  |               0.62 |      0.72 |  3.98 |       0.04 |
    | LassoCV                       |               0.62 |      0.72 |  3.98 |       0.06 |
    | LassoLarsIC                   |               0.62 |      0.72 |  3.98 |       0.01 |
    | LassoLarsCV                   |               0.62 |      0.72 |  3.98 |       0.02 |
    | Lars                          |               0.61 |      0.72 |  3.99 |       0.01 |
    | LarsCV                        |               0.61 |      0.71 |  4.02 |       0.04 |
    | SGDRegressor                  |               0.60 |      0.70 |  4.07 |       0.01 |
    | TweedieRegressor              |               0.59 |      0.70 |  4.12 |       0.01 |
    | GeneralizedLinearRegressor    |               0.59 |      0.70 |  4.12 |       0.01 |
    | ElasticNet                    |               0.58 |      0.69 |  4.16 |       0.01 |
    | Lasso                         |               0.54 |      0.66 |  4.35 |       0.02 |
    | RANSACRegressor               |               0.53 |      0.65 |  4.41 |       0.04 |
    | OrthogonalMatchingPursuitCV   |               0.45 |      0.59 |  4.78 |       0.02 |
    | PassiveAggressiveRegressor    |               0.37 |      0.54 |  5.09 |       0.01 |
    | GaussianProcessRegressor      |               0.23 |      0.43 |  5.65 |       0.03 |
    | OrthogonalMatchingPursuit     |               0.16 |      0.38 |  5.89 |       0.01 |
    | ExtraTreeRegressor            |               0.08 |      0.32 |  6.17 |       0.01 |
    | DummyRegressor                |              -0.38 |     -0.02 |  7.56 |       0.01 |
    | LassoLars                     |              -0.38 |     -0.02 |  7.56 |       0.01 |
    | KernelRidge                   |             -11.50 |     -8.25 | 22.74 |       0.01 |
    """

    def __init__(
        self,
        verbose=0,
        ignore_warnings=True,
        custom_metric=None,
        predictions=False,
        random_state=42,
        regressors="all",
    ):
        self.verbose = verbose
        self.ignore_warnings = ignore_warnings
        self.custom_metric = custom_metric
        self.predictions = predictions
        self.models = {}
        self.random_state = random_state
        self.regressors = regressors

    def fit(self, X_train, X_test, y_train, y_test):
        """Fit Regression algorithms to X_train and y_train, predict and score on X_test, y_test.
        Parameters
        ----------
        X_train : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        X_test : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        y_train : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        y_test : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        Returns
        -------
        scores : Pandas DataFrame
            Returns metrics of all the models in a Pandas DataFrame.
        predictions : Pandas DataFrame
            Returns predictions of all the models in a Pandas DataFrame.
        """
        R2 = []
        ADJR2 = []
        RMSE = []
        # WIN = []
        names = []
        TIME = []
        predictions = {}

        if self.custom_metric:
            CUSTOM_METRIC = []

        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)

        numeric_features = X_train.select_dtypes(include=[np.number]).columns
        categorical_features = X_train.select_dtypes(include=["object"]).columns

        categorical_low, categorical_high = get_card_split(
            X_train, categorical_features
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_transformer, numeric_features),
                ("categorical_low", categorical_transformer_low, categorical_low),
                ("categorical_high", categorical_transformer_high, categorical_high),
            ]
        )

        if self.regressors == "all":
            self.regressors = REGRESSORS
        else:
            try:
                temp_list = []
                for regressor in self.regressors:
                    full_name = (regressor.__name__, regressor)
                    temp_list.append(full_name)
                self.regressors = temp_list
            except Exception as exception:
                print(exception)
                print("Invalid Regressor(s)")

        for name, model in tqdm(self.regressors):
            start = time.time()
            try:
                if "random_state" in model().get_params().keys():
                    pipe = Pipeline(
                        steps=[
                            ("preprocessor", preprocessor),
                            ("regressor", model(random_state=self.random_state)),
                        ]
                    )
                else:
                    pipe = Pipeline(
                        steps=[("preprocessor", preprocessor), ("regressor", model())]
                    )

                pipe.fit(X_train, y_train)
                self.models[name] = pipe
                y_pred = pipe.predict(X_test)
                r_squared = r2_score(y_test, y_pred)
                adj_rsquared = adjusted_rsquared(
                    r_squared, X_test.shape[0], X_test.shape[1]
                )
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                names.append(name)
                R2.append(r_squared)
                ADJR2.append(adj_rsquared)
                RMSE.append(rmse)
                TIME.append(time.time() - start)

                if self.custom_metric:
                    custom_metric = self.custom_metric(y_test, y_pred)
                    CUSTOM_METRIC.append(custom_metric)

                if self.verbose > 0:
                    scores_verbose = {
                        "Model": name,
                        "R-Squared": r_squared,
                        "Adjusted R-Squared": adj_rsquared,
                        "RMSE": rmse,
                        "Time taken": time.time() - start,
                    }

                    if self.custom_metric:
                        scores_verbose[self.custom_metric.__name__] = custom_metric

                    print(scores_verbose)
                if self.predictions:
                    predictions[name] = y_pred
            except Exception as exception:
                if self.ignore_warnings is False:
                    print(name + " model failed to execute")
                    print(exception)

        scores = {
            "Model": names,
            "Adjusted R-Squared": ADJR2,
            "R-Squared": R2,
            "RMSE": RMSE,
            "Time Taken": TIME,
        }

        if self.custom_metric:
            scores[self.custom_metric.__name__] = CUSTOM_METRIC

        scores = pd.DataFrame(scores)
        scores = scores.sort_values(by="Adjusted R-Squared", ascending=False).set_index(
            "Model"
        )

        if self.predictions:
            predictions_df = pd.DataFrame.from_dict(predictions)
        return scores, predictions_df if self.predictions is True else scores

    def provide_models(self, X_train, X_test, y_train, y_test):
        """
        This function returns all the model objects trained in fit function.
        If fit is not called already, then we call fit and then return the models.
        Parameters
        ----------
        X_train : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        X_test : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        y_train : array-like,
            Training vectors, where rows is the number of samples
            and columns is the number of features.
        y_test : array-like,
            Testing vectors, where rows is the number of samples
            and columns is the number of features.
        Returns
        -------
        models: dict-object,
            Returns a dictionary with each model pipeline as value
            with key as name of models.
        """
        if len(self.models.keys()) == 0:
            self.fit(X_train, X_test, y_train, y_test)

        return self.models

Regression = FastRegressor
Classification = FastClassifier


# 建立模型
def build_model(df):

    X = df.iloc[:,:-1] # Using all column except for the last column as X
    Y = df.iloc[:,-1] # Selecting the last column as Y

    st.markdown('**1.2. 数据集维度**')
    st.write('X:')
    st.info(X.shape)
    st.write('Y:')
    st.info(Y.shape)

    st.markdown('**1.3. 变量名称**')
    st.write('X 变量 (展示前20项):')
    st.info(list(X.columns[:20]))
    st.write('Y 变量:')
    st.info(Y.name)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size = split_size,random_state = seed_number)
   # 判断是回归问题还是分类问题
    if  problem_select == "回归问题" :
        reg = FastRegressor(verbose=0,ignore_warnings=False, custom_metric=None)
        models_train,predictions_train = reg.fit(X_train, X_train, Y_train, Y_train)
        models_test,predictions_test = reg.fit(X_train, X_test, Y_train, Y_test)

        st.subheader('2. 模型性能表')
        st.write('训练集:')
        st.write(predictions_train)
        st.markdown(filedownload(predictions_train, 'training.csv'), unsafe_allow_html=True)
        st.write('测试集:')
        st.write(predictions_test)
        st.markdown(filedownload(predictions_test, 'test.csv'), unsafe_allow_html=True)
        st.subheader('3. 模型性能图 (测试集)')

        with st.markdown('**$R^{2}$:**'):
            # Tall
            predictions_test["R-Squared"] = [0 if i < 0 else i for i in predictions_test["R-Squared"]]
            plt.figure(figsize=(3, 9))
            sns.set_theme(style="whitegrid")
            ax1 = sns.barplot(y=predictions_test.index, x="R-Squared", data=predictions_test)
            ax1.set(xlim=(0, 1))
        st.markdown(imagedownload(plt, 'plot-$R^{2}$-tall.pdf'), unsafe_allow_html=True)
        # Wide
        plt.figure(figsize=(9, 3))
        sns.set_theme(style="whitegrid")
        ax1 = sns.barplot(x=predictions_test.index, y="R-Squared", data=predictions_test)
        ax1.set(ylim=(0, 1))
        plt.xticks(rotation=90)
        st.pyplot(plt)
        st.markdown(imagedownload(plt, 'plot-$R^{2}$-wide.pdf'), unsafe_allow_html=True)

        with st.markdown('**RMSE (最高50):**'):
            # Tall
            predictions_test["RMSE"] = [50 if i > 50 else i for i in predictions_test["RMSE"]]
            plt.figure(figsize=(3, 9))
            sns.set_theme(style="whitegrid")
            ax2 = sns.barplot(y=predictions_test.index, x="RMSE", data=predictions_test)
        st.markdown(imagedownload(plt, 'plot-rmse-tall.pdf'), unsafe_allow_html=True)
        # Wide
        plt.figure(figsize=(9, 3))
        sns.set_theme(style="whitegrid")
        ax2 = sns.barplot(x=predictions_test.index, y="RMSE", data=predictions_test)
        plt.xticks(rotation=90)
        st.pyplot(plt)
        st.markdown(imagedownload(plt, 'plot-rmse-wide.pdf'), unsafe_allow_html=True)

        with st.markdown('**计算时间:**'):
            # Tall
            predictions_test["Time Taken"] = [0 if i < 0 else i for i in predictions_test["Time Taken"]]
            plt.figure(figsize=(3, 9))
            sns.set_theme(style="whitegrid")
            ax3 = sns.barplot(y=predictions_test.index, x="Time Taken", data=predictions_test)
        st.markdown(imagedownload(plt, 'plot-calculation-time-tall.pdf'), unsafe_allow_html=True)
        # Wide
        plt.figure(figsize=(9, 3))
        sns.set_theme(style="whitegrid")
        ax3 = sns.barplot(x=predictions_test.index, y="Time Taken", data=predictions_test)
        plt.xticks(rotation=90)
        st.pyplot(plt)
        st.markdown(imagedownload(plt, 'plot-calculation-time-wide.pdf'), unsafe_allow_html=True)

    else:
        clf = FastClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
        models_train, predictions_train = clf.fit(X_train, X_train, Y_train, Y_train)
        models_test, predictions_test = clf.fit(X_train, X_test, Y_train, Y_test)

        st.subheader('2. 模型性能表')
        st.write('**2.1. 训练集结果:**')
        st.write(predictions_train)
        st.markdown(filedownload(predictions_train, 'training.csv'), unsafe_allow_html=True)
        st.write('**2.2. 测试集结果:**')
        st.write(predictions_test)
        st.markdown(filedownload(predictions_test, 'test.csv'), unsafe_allow_html=True)
        st.subheader('3. 模型性能图 (测试集)')

        with st.markdown('**3.1. Accuracy:**'):
            # Tall
            predictions_test["Accuracy"] = [0 if i < 0 else i for i in predictions_test["Accuracy"]]
            plt.figure(figsize=(3, 9))
            sns.set_theme(style="whitegrid")
            ax1 = sns.barplot(y=predictions_test.index, x="Accuracy", data=predictions_test)
            ax1.set(xlim=(0, 1))
        st.markdown(imagedownload(plt, 'plot-accuracy-tall.pdf'), unsafe_allow_html=True)
        # Wide
        plt.figure(figsize=(9, 3))
        sns.set_theme(style="whitegrid")
        ax1 = sns.barplot(x=predictions_test.index, y="Accuracy", data=predictions_test)
        ax1.set(ylim=(0, 1))
        plt.xticks(rotation=90)
        st.pyplot(plt)
        st.markdown(imagedownload(plt, 'plot-accuracy-wide.pdf'), unsafe_allow_html=True)

        with st.markdown('**3.2. ROC AUC:**'):
            # Tall
            predictions_test["ROC AUC"] = [50 if i > 50 else i for i in predictions_test["ROC AUC"]]
            plt.figure(figsize=(3, 9))
            sns.set_theme(style="whitegrid")
            ax2 = sns.barplot(y=predictions_test.index, x="ROC AUC", data=predictions_test)
        st.markdown(imagedownload(plt, 'plot-roc auc-tall.pdf'), unsafe_allow_html=True)
        # Wide
        plt.figure(figsize=(9, 3))
        sns.set_theme(style="whitegrid")
        ax2 = sns.barplot(x=predictions_test.index, y="ROC AUC", data=predictions_test)
        plt.xticks(rotation=90)
        st.pyplot(plt)
        st.markdown(imagedownload(plt, 'plot-roc auc-wide.pdf'), unsafe_allow_html=True)

        with st.markdown('**3.3. F1 Score:**'):
            # Tall
            predictions_test["F1 Score"] = [50 if i > 50 else i for i in predictions_test["F1 Score"]]
            plt.figure(figsize=(3, 9))
            sns.set_theme(style="whitegrid")
            ax2 = sns.barplot(y=predictions_test.index, x="F1 Score", data=predictions_test)
        st.markdown(imagedownload(plt, 'plot-F1 Score-tall.pdf'), unsafe_allow_html=True)
        # Wide
        plt.figure(figsize=(9, 3))
        sns.set_theme(style="whitegrid")
        ax2 = sns.barplot(x=predictions_test.index, y="F1 Score", data=predictions_test)
        plt.xticks(rotation=90)
        st.pyplot(plt)
        st.markdown(imagedownload(plt, 'plot-F1 Score-wide.pdf'), unsafe_allow_html=True)

        with st.markdown('**3.4. 计算时间:**'):
            # Tall
            predictions_test["Time Taken"] = [0 if i < 0 else i for i in predictions_test["Time Taken"]]
            plt.figure(figsize=(3, 9))
            sns.set_theme(style="whitegrid")
            ax3 = sns.barplot(y=predictions_test.index, x="Time Taken", data=predictions_test)
        st.markdown(imagedownload(plt, 'plot-calculation-time-tall.pdf'), unsafe_allow_html=True)
        # Wide
        plt.figure(figsize=(9, 3))
        sns.set_theme(style="whitegrid")
        ax3 = sns.barplot(x=predictions_test.index, y="Time Taken", data=predictions_test)
        plt.xticks(rotation=90)
        st.pyplot(plt)
        st.markdown(imagedownload(plt, 'plot-calculation-time-wide.pdf'), unsafe_allow_html=True)


def filedownload(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href

def imagedownload(plt, filename):
    s = io.BytesIO()
    plt.savefig(s, format='pdf', bbox_inches='tight')
    plt.close()
    b64 = base64.b64encode(s.getvalue()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:image/png;base64,{b64}" download={filename}>Download {filename} File</a>'
    return href

#---------------------------------#
# 主页的布局
# 主页可放大扩展至全屏
st.set_page_config(page_title='机器学习算法性能对比的web应用',
    layout='wide')

st.write("""
# 机器学习算法性能对比的web应用  

在本web应用中, 实现一次性建立多种机器学习算法，并进行性能比较。


""")

# 侧边栏 - 搜集用户输入数据1.问题分类2.上传训练数据3.参数设定

with st.sidebar.header('1. 选择问题类型'):
   problem_select = st.sidebar.radio(
        "你的问题是回归问题还是分类问题？",
        ("回归问题", "分类问题")
    )

with st.sidebar.header('2. 上传 CSV 数据集'):
    uploaded_file = st.sidebar.file_uploader("上传你的训练数据集（CSV格式）", type=["csv"])
    st.sidebar.markdown("""
    CSV 数据案例：  
[ 回归问题数据集](https://github.com/goddesskiller/fastpredict/blob/main/diabetes.csv)  
[ 分类问题数据集](https://github.com/goddesskiller/fastpredict/blob/main/breast_cancer.csv)
""")
#案例数据集已上传github

with st.sidebar.header('3. 参数设定'):
    split_size = st.sidebar.slider('数据划分比例 (训练集占% )', 10, 90, 80, 5)
    seed_number = st.sidebar.slider('设定随机数种子（控制数据划分时的随机性）', 1, 100, 42, 1)


#---------------------------------#
# 展示数据集
st.subheader('1. 数据集')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**1.1. 数据集概览**')
    st.write(df.head(10))
    build_model(df)
else:
    st.info('等待上传 CSV 文件。')
    if st.button('使用案例数据集'):
        # 使用diabetes数据集
        if problem_select == "回归问题" :
            example_dataset = load_diabetes()
            #X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
            #Y = pd.Series(diabetes.target, name='target')
            X = pd.DataFrame(example_dataset.data, columns=example_dataset.feature_names).loc[:100] # 为提高测试效率，只取前100行数据
            Y = pd.Series(example_dataset.target, name='target').loc[:100] # 为提高测试效率，只取前100行数据
            df = pd.concat( [X,Y], axis=1 )
            st.markdown('**1.1. 数据集概览**')
            st.markdown('使用diabetes数据集作为案例。')
        #使用breast_cancer数据集
        else:
            example_dataset = load_breast_cancer()
            # X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
            # Y = pd.Series(diabetes.target, name='target')
            X = pd.DataFrame(example_dataset.data, columns=example_dataset.feature_names).loc[:100]  # FOR TESTING PURPOSE, COMMENT THIS OUT FOR PRODUCTION
            Y = pd.Series(example_dataset.target, name='target').loc[:100]  # FOR TESTING PURPOSE, COMMENT THIS OUT FOR PRODUCTION
            df = pd.concat([X, Y], axis=1)
            st.markdown('**1.1. 数据集概览**')
            st.markdown('使用breast_cancer数据集作为案例。')
        st.write(df.head(10))

        build_model(df)

