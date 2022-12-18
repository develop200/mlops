import pytest
from flask import Flask
import pandas as pd
from sklearn.linear_model import Ridge
import sys
sys.path.insert(0, '../..')
sys.path.insert(0, '../')
sys.path.insert(0, '.')
from src.model import MLLinearModel, db
import src.api
from src.api import api


EXAMPLE_DATASET_PATH = "example.json"


@pytest.fixture
def mock_db(mocker):
    mock = mocker.patch("src.model.db")
    return mock


@pytest.fixture
def mock_sqlalchemy_getter(mocker):
    mock = mocker.patch(
        "flask_sqlalchemy._QueryProperty.__get__").return_value = mocker.Mock()
    return mock


@pytest.fixture()
def mock_app():
    app_mock = Flask(__name__)
    app_mock.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    app_mock.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(app_mock)
    return app_mock


@pytest.fixture
def mock_linear_model():
    model = Ridge()
    train_dataset = pd.read_json(EXAMPLE_DATASET_PATH, orient="index")
    train_features = train_dataset[[
        'Length1', 'Length2', 'Length3', 'Height', 'Width']]
    train_target = train_dataset['Weight']
    model.fit(train_features, train_target)
    model = MLLinearModel(data=model)
    return model


@pytest.fixture
def mock_linear_id_property(mocker):
    mock = mocker.patch(
        "src.model.MLLinearModel.id",
        new_callable=mocker.PropertyMock,
        return_value=0
    )
    return mock

@pytest.fixture()
def mock_api():
    app_mock = Flask(__name__)
    api.init_app(app_mock)
    return app_mock
