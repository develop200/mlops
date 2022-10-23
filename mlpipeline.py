"""Operate with pretrained ML models.

Module provides classes which enable to fit and tune pretrained
ML models, get their predictions of user' data and operate with
them (read, delete).

Classes:

    MLPipeline
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


class MLPipeline:
    """Provide interface for operating with pretrained models."""

    def __init__(self):
        """Load pretrained models (or train new if models do not exist) and
        construct an object of the class containing them."""

        self._pickles = {
            "linear": ".//data//pickles//linear_model.pkl",
            "forest": ".//data//pickles//forest_model.pkl"
        }
        self._initial_dataset = ".//data//datasets//train.csv"

        self._fitted_flg = False

        self._hyperparams = {
            "linear": {"alpha": 0.},
            "forest": {
                "n_estimators": 100,
                "max_depth": 5,
                "min_samples_split": 10,
                "max_features": "log2"
            }
        }
        self._hyperparams_types = {
            "linear": {"alpha": float},
            "forest": {
                "n_estimators": int,
                "max_depth": int,
                "min_samples_split": int,
                "max_features": float
            }
        }
        self._models_classes = {
            "linear": Ridge,
            "forest": RandomForestRegressor
        }
        self._models = {
            "linear": None,
            "forest": None
        }

        train = pd.read_csv(self._initial_dataset)
        X, y = train[['Length1', 'Length2', 'Length3', 'Height', 'Width']], train['Weight']
        for model_type, pkl_name in self._pickles.items():
            if os.path.exists(pkl_name):
                self._models[model_type] = joblib.load(pkl_name)
            else:
                self._models[model_type] = self._models_classes[model_type](**self._hyperparams[model_type]).fit(X, y)
                joblib.dump(self._models[model_type], pkl_name)

        self._fitted_flg = True

    def refit(self):
        """Call fit method of pretrained models on predefined train dataset
        again and save last checkpoints of models."""
        self._fitted_flg = False

        train = pd.read_csv(self._initial_dataset)
        X, y = train[['Length1', 'Length2', 'Length3', 'Height', 'Width']], train['Weight']
        for model_type, pkl_name in self._pickles.items():
            self._models[model_type] = self._models_classes[model_type](**self._hyperparams[model_type]).fit(X, y)
            joblib.dump(self._models[model_type], pkl_name)

        self._fitted_flg = True

    def predict(self, test, model_type):
        """Make prediction by chosen type of model.

        Keyword arguments:
        test -- dataframe with observations to predict
        model_type -- model type for prediction

        Return dictionary with success flag, predictions and additional info
        about possible errors in input.
        """
        if model_type not in self._models:
            return {
                "success_flg": False,
                "info": f"Model type {model_type} is not available."
            }

        if type(test) != pd.DataFrame or test.shape == 0:
            return {"success_flg": False, "info": "Bad input for prediction."}

        train = pd.read_csv(self._initial_dataset)
        X = train[['Length1', 'Length2', 'Length3', 'Height', 'Width']]

        if self._fitted_flg:
            if len(set(list(test.columns)).intersection(set(list(X.columns)))) == X.shape[1]:
                res = self._models[model_type].predict(test[X.columns].fillna(0))
                return {
                    "success_flg": True,
                    "data": list(res),
                    "info": f"Predictions of {model_type} model."
                }
            else:
                return {"success_flg": False, "info": "No needed columns."}
        else:
            return {"success_flg": False, "info": "Models are not fitted."}

    def set_hyperparams(self, model_type, params):
        """Set hyperparameters of chosen type of model.

        Keyword arguments:
        model_type -- model type for prediction
        params -- dictionary with new hyperparameters

        Return dictionary with success flag, all available hyperparameters
        of all available models' types and additional info about possible
        errors in input.
        """
        if model_type not in self._models:
            return {
                "success_flg": False,
                "hyperparams": self._hyperparams,
                "info": f"Model type {model_type} is not available."
            }

        if type(params) != dict or len(params) == 0:
            return {
                "success_flg": False,
                "hyperparams": self._hyperparams,
                "info": "Bad input of hyperparams."
            }

        if len(set(list(self._hyperparams[model_type].keys())).intersection(set(list(params.keys())))) != len(set(list(params.keys()))):
            return {
                "success_flg": False,
                "hyperparams": self._hyperparams,
                "info": "Some of this hyperparams are not allowed."
            }

        for key, value in params.items():
            try:
                params[key] = self._hyperparams_types[model_type][key](value)
            except Exception:
                return {
                    "success_flg": False,
                    "hyperparams": self._hyperparams,
                    "info": "Bad input of hyperparams (bad types)."
                }

        for key, value in params.items():
            self._hyperparams[model_type][key] = value
        self.refit()
        return {
            "success_flg": True,
            "hyperparams": self._hyperparams,
            "info": "Hyperparams were changed. Models are refitted."
        }

    def fit_hyperparams(self):
        """Tune all available models' hyperparameters with optuna framework,
        refit and save last checkpoints of all models with best values.

        Return dictionary with success flag, all available hyperparameters
        af all available models' types and additional info about possible
        errors in input.
        """
        train = pd.read_csv(self._initial_dataset)
        X, y = train[['Length1', 'Length2', 'Length3', 'Height', 'Width']], train['Weight']

        model_type = "forest"
        if model_type in self._models:
            def objective(trial):
                param_grid = {
                    "n_estimators": trial.suggest_int("n_estimators", 10, 100),
                    "max_depth": trial.suggest_int("max_depth", 2, 10),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
                    "max_features": trial.suggest_discrete_uniform("max_features", 0.5, 1., 0.1)
                }

                cv = KFold(n_splits=3, shuffle=True, random_state=10)
                cv_scores = np.zeros(3)

                for i, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                    model = self._models_classes[model_type](**param_grid, random_state=10, n_jobs=-1)
                    model.fit(X.loc[train_idx], y.loc[train_idx])
                    preds = model.predict(X.loc[val_idx])
                    cv_scores[i] = r2_score(y.loc[val_idx], preds)

                return np.mean(cv_scores)

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=20)
            self._hyperparams[model_type] = study.best_params

        model_type = "linear"
        if model_type in self._models:
            def objective(trial):
                param_grid = {
                    "alpha": trial.suggest_loguniform("alpha", 1e-5, 1e2),
                }

                cv = KFold(n_splits=3, shuffle=True, random_state=10)
                cv_scores = np.zeros(3)

                for i, (train_idx, val_idx) in enumerate(cv.split(X, y)):
                    model = self._models_classes[model_type](**param_grid, random_state=10)
                    model.fit(X.loc[train_idx], y.loc[train_idx])
                    preds = model.predict(X.loc[val_idx])
                    cv_scores[i] = r2_score(y.loc[val_idx], preds)

                return np.mean(cv_scores)

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=20)
            self._hyperparams[model_type] = study.best_params

        self.refit()

        return {
            "success_flg": True,
            "hyperparams": self._hyperparams,
            "info": "Hyperparams are optimal. Models are refitted."
        }

    def get_avaliable_model_classes(self):
        """Return dictionary with success flag and list of all
        available models' classes"""
        return {"success_flg": True, "data": list(self._models.keys())}

    def fit_new_data(self, new_X, new_y):
        """Fit all available models on new data.

        Keyword arguments:
        new_X -- dataframe with features
        new_y -- pd.Series/np.array with target

        Return dictionary with success flag and additional info about
        possible errors in input.
        """
        if type(new_X) != pd.DataFrame or new_X.shape[0] < 3:
            return {"success_flg": False, "info": "Very small data for train."}
        if (type(new_y) != pd.Series and type(new_y) != np.array) or new_y.shape[0] != new_X.shape[0]:
            return {"success_flg": False, "info": "Bad target."}

        train = pd.read_csv(self._initial_dataset)
        X = train[['Length1', 'Length2', 'Length3', 'Height', 'Width']]

        if len(set(list(new_X.columns)).intersection(set(list(X.columns)))) == X.shape[1]:
            train = new_X
            train["Weight"] = new_y
            train.to_csv(self._initial_dataset, index=False)
            self.refit()
            return {
                "success_flg": True,
                "info": "Models are refitted on new data. Train dataset was changed."
            }
        else:
            return {
                "success_flg": False,
                "info": "No needed columns in new train dataset."
            }

    def delete_model(self, model_type):
        """Delete model of chosen type.

        Keyword arguments:
        model_type -- type of models to delete

        Return dictionary with success flag, new array of available models'
        types and additional info about possible errors in input.
        """
        if model_type in self._models:
            os.remove(self._pickles[model_type])
            del self._pickles[model_type]
            del self._hyperparams[model_type]
            del self._models_classes[model_type]
            del self._models[model_type]
            return {
                "success_flg": True,
                "models_types": list(self._models.keys()),
                "info": f"Model type {model_type} was deleted."
            }
        else:
            return {
                "success_flg": False,
                "models_types": list(self._models.keys()),
                "info": f"Model type {model_type} is not available."
            }
