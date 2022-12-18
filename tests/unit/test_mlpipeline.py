import sys
import pandas as pd
import json
from src.pipeline import MLPipeline
import pytest
from conftest import EXAMPLE_DATASET_PATH


@pytest.mark.parametrize("model_type,corrupt_dataset_mode", [("linear", None), ("forest", None), ("unk", None), ("linear", 0), ("forest", 1)])
def test_predict(
    mock_app,
    mock_sqlalchemy_getter,
    mock_db,
    mock_linear_model,
    mock_linear_id_property,
    model_type,
    corrupt_dataset_mode,
    model_id=0
):
    pipeline = MLPipeline(mock_db)
    with mock_app.app_context():
        mock_sqlalchemy_getter.get.return_value = mock_linear_model
        mock_sqlalchemy_getter.all.return_value = [mock_linear_model]
        test_dataset = pd.read_json(EXAMPLE_DATASET_PATH, orient="index")
        if corrupt_dataset_mode is not None:
            if corrupt_dataset_mode == 0:
                test_dataset = test_dataset.drop(["Length1"], axis=1)
            if corrupt_dataset_mode == 1:
                test_dataset = test_dataset.iloc[:0,:]
        res = pipeline.predict(test_dataset, model_type, model_id)
        
        if model_type in ["linear", "forest"] and corrupt_dataset_mode is None:
            assert res["success_flg"]
            assert type(res["data"]) == dict and all([int(return_id) == int(query_id) for return_id, query_id in zip(
                res["data"].keys(), json.load(open(EXAMPLE_DATASET_PATH)).keys())])
        else:
            assert not res["success_flg"]


def test_models_list(
    mock_app,
    mock_sqlalchemy_getter,
    mock_db,
    mock_linear_model,
    mock_linear_id_property,
    model_type="linear",
    model_id=0
):
    pipeline = MLPipeline(mock_db)
    with mock_app.app_context():
        mock_sqlalchemy_getter.get.return_value = mock_linear_model
        mock_sqlalchemy_getter.all.return_value = [mock_linear_model]
        res = pipeline.get_avaliable_model_classes()
        assert res["success_flg"]
        assert set(res["data"].keys()) == set(["linear", "forest"])
        assert set(res["data"]["linear"]) == set(
            [0,]) and set(res["data"]["forest"]) == set([0,])


@pytest.mark.parametrize("model_type,corrupt_dataset_mode,params", [("linear", None, None), ("unk", None, None), ("linear", 0, None), ("linear", 1, None), ("linear", None, {"alpha": 10}), ("linear", None, {"n_estimators": 10})])
def test_train(
    mock_app,
    mock_sqlalchemy_getter,
    mock_db,
    mock_linear_model,
    mock_linear_id_property,
    model_type,
    corrupt_dataset_mode,
    params
):
    pipeline = MLPipeline(mock_db)
    with mock_app.app_context():
        mock_sqlalchemy_getter.get.return_value = mock_linear_model
        mock_sqlalchemy_getter.all.return_value = [mock_linear_model]
        train_dataset = pd.read_json(EXAMPLE_DATASET_PATH, orient="index")
        if corrupt_dataset_mode is not None:
            if corrupt_dataset_mode == 0:
                train_dataset = train_dataset.drop(["Length1"], axis=1)
            if corrupt_dataset_mode == 1:
                train_dataset = train_dataset.iloc[:0,:]
        res = pipeline.train_model(model_type, train_dataset, params)
        if model_type not in ["linear", "forest"]:
            assert not res["success_flg"]
        elif corrupt_dataset_mode is not None:
            assert not res["success_flg"]
        elif params is not None and len(set(list(pipeline._hyperparams_types[model_type].keys())).intersection(set(list(params.keys())))) != len(set(list(params.keys()))):
            assert not res["success_flg"]
        else:
            assert res["success_flg"]


@pytest.mark.parametrize("model_type", ["linear", "unk"])
def test_delete(
    mock_app,
    mock_sqlalchemy_getter,
    mock_db,
    mock_linear_model,
    mock_linear_id_property,
    model_type,
    model_id=0
):
    pipeline = MLPipeline(mock_db)
    with mock_app.app_context():
        mock_sqlalchemy_getter.get.return_value = mock_linear_model
        mock_sqlalchemy_getter.all.return_value = [mock_linear_model]
        res = pipeline.delete_model(model_type, model_id)
        if model_type not in ["linear", "forest"]:
            assert not res["success_flg"]
        else:
            assert res["success_flg"]
            assert set(res["available_models"].keys()) == set(["linear", "forest"])
            assert set(res["available_models"]["linear"]) == set(
                [0,]) and set(res["available_models"]["forest"]) == set([0,])
