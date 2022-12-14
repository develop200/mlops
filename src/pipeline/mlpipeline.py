import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import optuna
from src.model import MLLinearModel, MLForestModel
optuna.logging.set_verbosity(optuna.logging.WARNING)


class MLPipeline:
    """Provide interface for operating with pretrained models."""

    def __init__(self, models_base):
        """Load database and construct an object
        of the class containing it."""

        self.models_base = models_base

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

        self._models_tables = {
            "linear": MLLinearModel,
            "forest": MLForestModel
        }

    def validate_model_type(self, model_type):
        """Validate the existance of selected type of model.

        Keyword arguments:
        model_type -- model type

        Return dictionary with success flag and additional info
        about possible errors in input.
        """
        if model_type not in self._models_tables:
            return self.generate_response(
                success_flg=False, info=f"Model type {model_type} is not available.")
        else:
            return self.generate_response(
                success_flg=True, info=f"Model type {model_type} is available.")

    def validate_model_id(self, model_type, model_id):
        """Validate the existance of selected model id.

        Keyword arguments:
        model_type -- model type
        model_id -- model id

        Return dictionary with success flag and additional info
        about possible errors in input.
        """
        if model_type not in self._models_tables:
            return self.generate_response(
                success_flg=False, info=f"Model type {model_type} is not available.")
        if model_id not in [
                model.id for model in self._models_tables[model_type].query.all()]:
            return self.generate_response(
                success_flg=False, info=f"Model id {model_id} is not available.")
        return self.generate_response(
            success_flg=True, info=f"Model id {model_id} is available.")

    def validate_model_params(self, model_type, params):
        """Validate user's hyperparams for selected model type.

        Keyword arguments:
        model_type -- model type
        params -- dictionary with hyperparams

        Return dictionary with success flag and additional info
        about possible errors in input.
        """
        if not isinstance(params, dict) or len(params) == 0:
            return self.generate_response(
                success_flg=False, info="Bad input of hyperparams.")
        if len(set(list(self._hyperparams_types[model_type].keys())).intersection(
                set(list(params.keys())))) != len(set(list(params.keys()))):
            return self.generate_response(
                success_flg=False, info="Some of this hyperparams are not allowed for chosen model_type.")
        for key, value in params.items():
            try:
                params[key] = self._hyperparams_types[model_type][key](value)
            except Exception:
                return self.generate_response(
                    success_flg=False, info="Bad input of hyperparams (bad types).")
        return self.generate_response(
            success_flg=True, info="Hyperparams are good.")

    def validate_dataset(self, train_dataset, train_flg=True):
        """Validate user's selected dataset for training/prediction.

        Keyword arguments:
        train_dataset -- pandas DataFrame with data
        train_flg -- validate dataset for training if True else validate for prediction

        Return dictionary with success flag and additional info
        about possible errors in input.
        """
        needed_columns = {}
        if train_flg:
            needed_columns = {
                'Length1',
                'Length2',
                'Length3',
                'Height',
                'Width',
                'Weight'}
        else:
            needed_columns = {
                'Length1',
                'Length2',
                'Length3',
                'Height',
                'Width'}
        needed_min_rows = -1
        if train_flg:
            needed_min_rows = 20
        else:
            needed_min_rows = 1

        if not isinstance(
                train_dataset, pd.DataFrame) or train_dataset.shape[0] < needed_min_rows:
            return self.generate_response(
                success_flg=False, info="Poor data for train.")

        if len(set(list(train_dataset.columns)).intersection(
                set(needed_columns))) != len(needed_columns):
            return self.generate_response(
                success_flg=False, info=f"No needed columns in dataset. You need {needed_columns}.")
        return self.generate_response(
            success_flg=True, info="Dataset is good.")

    def generate_response(self, **kwargs):
        return kwargs

    def predict(self, test, model_type, model_id):
        """Make prediction by chosen model.

        Keyword arguments:
        test -- dataframe with observations to predict
        model_type -- model type for prediction
        model_id -- model id for prediction

        Return dictionary with success flag, predictions and additional info
        about possible errors in input.
        """
        valid_res = self.validate_model_type(model_type)
        if valid_res["success_flg"] is False:
            return valid_res
        valid_res = self.validate_model_id(model_type, model_id)
        if valid_res["success_flg"] is False:
            return valid_res
        valid_res = self.validate_dataset(test, train_flg=False)
        if valid_res["success_flg"] is False:
            return valid_res

        model = self._models_tables[model_type].query.get(model_id).data
        res = model.predict(
            test[['Length1', 'Length2', 'Length3', 'Height', 'Width']].fillna(0))
        res = pd.Series(np.array(res), index=test.index).to_dict()
        return self.generate_response(
            success_flg=True, data=res, info=f"Predictions of {model_type} model with id = {model_id}.")

    def forest_optuna_fit(self, model_type, train_features, train_target):
        """Find optimal hyperparams for forest.

        Keyword arguments:
        model_type -- model type for prediction
        train_features -- features for objects to train model
        train_target -- target for objects to train model

        Return dictionary with best hyperparams.
        """
        def objective(trial):
            param_grid = {
                "n_estimators": trial.suggest_int("n_estimators", 10, 100),
                "max_depth": trial.suggest_int("max_depth", 2, 10),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 50),
                "max_features": trial.suggest_discrete_uniform("max_features", 0.5, 1., 0.1)
            }

            cv = KFold(n_splits=3, shuffle=True, random_state=10)
            cv_scores = np.zeros(3)

            for i, (train_idx, val_idx) in enumerate(
                    cv.split(train_features, train_target)):
                model = self._models_classes[model_type](
                    **param_grid, random_state=10, n_jobs=-1)
                model.fit(
                    train_features.loc[train_idx],
                    train_target.loc[train_idx])
                preds = model.predict(train_features.loc[val_idx])
                cv_scores[i] = r2_score(train_target.loc[val_idx], preds)

            return np.mean(cv_scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20)
        best_hyperparams = study.best_params
        return best_hyperparams

    def linear_optuna_fit(self, model_type, train_features, train_target):
        """Find optimal hyperparams for linear model.

        Keyword arguments:
        model_type -- model type for prediction
        train_features -- features for objects to train model
        train_target -- target for objects to train model

        Return dictionary with best hyperparams.
        """
        def objective(trial):
            param_grid = {
                "alpha": trial.suggest_loguniform("alpha", 1e-5, 1e2),
            }

            cv = KFold(n_splits=3, shuffle=True, random_state=10)
            cv_scores = np.zeros(3)

            for i, (train_idx, val_idx) in enumerate(
                    cv.split(train_features, train_target)):
                model = self._models_classes[model_type](
                    **param_grid, random_state=10)
                model.fit(
                    train_features.loc[train_idx],
                    train_target.loc[train_idx])
                preds = model.predict(train_features.loc[val_idx])
                cv_scores[i] = r2_score(train_target.loc[val_idx], preds)

            return np.mean(cv_scores)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20)
        best_hyperparams = study.best_params
        return best_hyperparams

    def find_optimal_hyperparams(self, model_type, train_dataset):
        """Find best hyperparams of selected model type with optuna.

        Keyword arguments:
        model_type -- model type
        train_dataset -- dataframe with observations to train

        Return dictionary with best hyperparams.
        """
        train_features = train_dataset[[
            'Length1', 'Length2', 'Length3', 'Height', 'Width']]
        train_target = train_dataset['Weight']

        if model_type == "forest":
            best_hyperparams = self.forest_optuna_fit(
                model_type, train_features, train_target)
        if model_type == "linear":
            best_hyperparams = self.linear_optuna_fit(
                model_type, train_features, train_target)

        return best_hyperparams

    def create_model(self, model_type, train_dataset, params):
        """Create new model and write it to database.

        Keyword arguments:
        model_type -- model type
        train_dataset -- dataframe with observations to train
        params -- hyperparams for new model

        Return id of new model.
        """
        train_features = train_dataset[[
            'Length1', 'Length2', 'Length3', 'Height', 'Width']]
        train_target = train_dataset['Weight']
        model = self._models_classes[model_type](
            **params).fit(train_features, train_target)
        new_element = self._models_tables[model_type](data=model)
        self.models_base.session.add(new_element)
        self.models_base.session.flush()
        self.models_base.session.commit()
        self.models_base.session.refresh(new_element)
        new_model_id = new_element.id
        return new_model_id

    def train_model(self, model_type, train_dataset, params=None):
        """Train model of chosen type with chosen hyperparameters on the dataset.
        If no hyperparameters provided find the best with optuna. Write new model
        to database.

        Keyword arguments:
        model_type -- model type
        train_dataset -- data for training model
        params -- dictionary with new hyperparameters
                  (if None - find best with optuna)

        Return dictionary with success flag and id of new model (or additional
        info about possible errors in input.
        """
        valid_res = self.validate_model_type(model_type)
        if valid_res["success_flg"] is False:
            return valid_res
        if params is not None:
            valid_res = self.validate_model_params(model_type, params)
            if valid_res["success_flg"] is False:
                return valid_res
        valid_res = self.validate_dataset(train_dataset)
        if valid_res["success_flg"] is False:
            return valid_res

        if params is None:
            params = self.find_optimal_hyperparams(model_type, train_dataset)

        new_model_id = self.create_model(model_type, train_dataset, params)

        return self.generate_response(
            success_flg=True, info=f"Model of type {model_type} with your/optuna hyperparams was trained. It's id - {new_model_id}.")

    def get_avaliable_model_classes(self):
        """Return dictionary with success flag and list of all
        available models' classes and ids"""
        return self.generate_response(success_flg=True, data={model_type: [
                                      model.id for model in self._models_tables[model_type].query.all()] for model_type in self._models_tables})

    def delete_model(self, model_type, model_id):
        """Delete model.

        Keyword arguments:
        model_type -- type of model to delete
        model_id -- id of model to delete

        Return dictionary with success flag, new array of available models
        and additional info about possible errors in input.
        """
        valid_res = self.validate_model_type(model_type)
        if valid_res["success_flg"] is False:
            return valid_res
        valid_res = self.validate_model_id(model_type, model_id)
        if valid_res["success_flg"] is False:
            return valid_res

        self._models_tables[model_type].query.filter_by(id=model_id).delete()
        self.models_base.session.flush()
        self.models_base.session.commit()

        return self.generate_response(success_flg=True, available_models=self.get_avaliable_model_classes()[
                                      "data"], info=f"Model {model_id} of type {model_type} was deleted.")
