import pytest
import sys
import pandas as pd
import json
from src.pipeline import MLPipeline
from conftest import EXAMPLE_DATASET_PATH
import io


@pytest.mark.parametrize("file_specified", [True, False])
def test_api_predict(
    mock_api,
    mocker,
    file_specified,
    tmp_data="example"
):
    mocker.patch("src.api.pipeline.predict", return_value={"success_flg": True, "tmp_data": tmp_data})
    if file_specified:
        with open(EXAMPLE_DATASET_PATH, 'rb') as f:
            response = mock_api.test_client().post('/api/predict?model_type=linear&model_id=0', data={"file": (f, f.name)})
    else:
        response = mock_api.test_client().post('/api/predict?model_type=linear&model_id=0')
    res = json.loads(response.text)
    if file_specified:
        assert response.status_code == 200
        assert res["tmp_data"] == tmp_data
    else:
        assert response.status_code == 400


@pytest.mark.parametrize("file_specified", [True, False])
def test_api_train(
    mock_api,
    mocker,
    file_specified,
    tmp_data="example"
):
    mocker.patch("src.api.pipeline.train_model", return_value={"success_flg": True, "tmp_data": tmp_data})
    if file_specified:
        with open(EXAMPLE_DATASET_PATH, 'rb') as f:
            response = mock_api.test_client().post('/api/train?model_type=linear', data={"file": (f, f.name)})
    else:
        response = mock_api.test_client().post('/api/train?model_type=linear')
    res = json.loads(response.text)
    if file_specified:
        assert response.status_code == 200
        assert res["tmp_data"] == tmp_data
    else:
        assert response.status_code == 400


def test_api_models_types(
    mock_api,
    mocker,
    tmp_data="example"
):
    mocker.patch("src.api.pipeline.get_avaliable_model_classes", return_value={"success_flg": True, "tmp_data": tmp_data})
    response = mock_api.test_client().get('/api/models_types')
    res = json.loads(response.text)
    assert response.status_code == 200
    assert res["tmp_data"] == tmp_data


@pytest.mark.parametrize("success_flg", [True, False])
def test_api_models_types(
    mock_api,
    mocker,
    success_flg,
    tmp_data="example"
):
    mocker.patch("src.api.pipeline.delete_model", return_value={"success_flg": success_flg, "tmp_data": tmp_data})
    response = mock_api.test_client().delete('/api/delete?model_type=linear&model_id=0')
    res = json.loads(response.text)
    if success_flg:
        assert response.status_code == 200
    else:
        assert response.status_code == 400
    assert res["tmp_data"] == tmp_data
