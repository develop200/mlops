from flask import Flask, make_response, jsonify, request
from flask_restx import Api, Resource, fields, reqparse
from werkzeug.datastructures import FileStorage
from mlpipeline import MLPipeline
import pandas as pd

app = Flask(__name__)
api = Api(
    version="1.0",
    title="ML Pipeline API",
    description="API, предоставляющий интерфейс к предобученным моделям, решающим задачу предсказани веса рыбы по её признакам. Описание задачи можно найти по ссылке https://www.kaggle.com/datasets/aungpyaeap/fish-market",
    contact="adromanov_2@edu.hse.ru",
    default="APImethods",
    default_label="Функционал"
)

pipeline = MLPipeline()


model = api.model('models_typs_model', {
    "success_flg": fields.Boolean(description="Флаг успешного выполнения", example=True),
    "data": fields.List(fields.String, description="Список доступных типов моделей", example=["linear", "forest"])
})


@api.doc(description="Возвращает список доступных типов моделей.")
@api.response(200, 'Успешное выполнение', model)
class ModelsTypesGetter(Resource):
    """Provide access to resource 'models types'."""

    def get(self):
        """
        Возвращает список доступных типов моделей

        Call method of MLPipeline object providing available models' types
        (see mlpipeline.MLPipeline.get_avaliable_model_classes() documentation).

        Return output of called method converted to Response class
        with code 200.
        """
        response = make_response(jsonify(pipeline.get_avaliable_model_classes()), 200)
        response.headers["Content-Type"] = "application/json"
        return response


model_true = api.model('deleter_true_model', {
    "success_flg": fields.Boolean(description="Флаг успешного выполнения", example=True),
    "models_types": fields.List(fields.String, description="Список доступных типов моделей", example=["linear", "forest"]),
    "info": fields.String(description="Доп. информация о выполнении.", example="Model type boosting was deleted.")
})
model_false = api.model('deleter_false_model', {
    "success_flg": fields.Boolean(description="Флаг успешного выполнения", example=False),
    "models_types": fields.List(fields.String, description="Список доступных типов моделей", example=["linear", "forest"]),
    "info": fields.String(description="Доп. информация о выполнении.", example="Model type boosting is not available.")
})


@api.doc(
    params={'model_type': 'Тип модели'},
    description="Производит удаление модели по её типу."
)
@api.response(200, 'Успешное выполнение', model_true)
@api.response(400, 'Неуспешное выполнение (доп. информация предоставляется в поле "info")', model_false)
class ModelsDeleter(Resource):
    """Provide access to resource 'models list'."""

    def delete(self, model_type):
        """
        Производит удаление модели по её типу

        Call method of MLPipeline object deleting chosen model type
        (see mlpipeline.MLPipeline.delete_model() documentation).

        Return output of called method converted to Response class
        with code 200 if output["success_flg"] is True else 400.
        """
        result = pipeline.delete_model(model_type)
        response = make_response(jsonify(result), int(200 * result["success_flg"] + 400 * (1 - result["success_flg"])))
        response.headers["Content-Type"] = "application/json"
        return response


model_true = api.model('hyperparams_true_model', {
    "success_flg": fields.Boolean(description="Флаг успешного выполнения", example=True),
    "hyperparams": fields.Raw(description="Гиперпараметры всех доступных моделей", example={"linear": {"alpha": 0.}, "forest": {"n_estimators": 100, "max_depth": 5, "min_samples_split": 10, "max_features": "log2"}}),
    "info": fields.String(description="Доп. информация о выполнении.", example="Hyperparams are optimal. Models are refitted.")
})
model_false = api.model('hyperparams_false_model', {
    "success_flg": fields.Boolean(description="Флаг успешного выполнения", example=False),
    "hyperparams": fields.Raw(description="Гиперпараметры всех доступных моделей", example={"linear": {"alpha": 0.}, "forest": {"n_estimators": 100, "max_depth": 5, "min_samples_split": 10, "max_features": "log2"}}),
    "info": fields.String(description="Доп. информация о выполнении.", example="Bad input of hyperparams.")
})


@api.doc(
    params={
        'model_type': 'Тип модели (возможные знаения: "all" или существующий тип модели)',
        'n_estimators': {'description': 'Количество деревьев в ансамбле (необязательный параметр, актуален только для типа модели "forest").', 'in': 'query', 'type': 'int', 'required': False},
        'max_depth': {'description': 'Максимальная глубина деревьев в ансамбле (необязательный параметр, актуален только для типа модели "forest").', 'in': 'query', 'type': 'int', 'required': False},
        'min_samples_split': {'description': 'Минимальное количество наблюдений для разделения узла на листы в деревьях ансамбля (необязательный параметр, актуален только для типа модели "forest").', 'in': 'query', 'type': 'int', 'required': False},
        'max_features': {'description': 'Доля признаков, использующихся при построении моделей бэггинга (необязательный параметр, актуален только для типа модели "forest").', 'in': 'query', 'type': 'float', 'required': False},
        'alpha': {'description': 'Коэффициент регуляризации (необязательный параметр, актуален только для типа модели "linear").', 'in': 'query', 'type': 'float', 'required': False},
    },
    description="Производит изменение гиперпараметров моделей. Если в качестве параметра 'model_type' передаётся 'all', то гиперпараметры подбираются автоматически для всех моделей. Если в качестве параметра 'model_type' передаётся тип модели, то также предполагается передача списка гиперпараметров в параметрах запроса, после чего будет произведена замена гиперпараметров и переобучение модели данного типа."
)
@api.response(200, 'Успешное выполнение', model_true)
@api.response(400, 'Неуспешное выполнение (доп. информация предоставляется в поле "info")', model_false)
class HyperparamsRuler(Resource):
    """Provide access to resource 'models hyperparams'."""

    def post(self, model_type):
        """
        Производит изменение гиперпараметров моделей

        Call methods of MLPipeline object changing hyperparams of chosen model type
        (see mlpipeline.MLPipeline.fit_hyperparams() and mlpipeline.MLPipeline.set_hyperparams()
        documentation).

        If 'model_type'=='all' all hyperparams of all models will be
        tuned by optuna. Else hyperparams of model with type 'model_type'
        will be set to values passed in query.

        Keyword arguments:
        model_type -- type of models.

        Return output of called method converted to Response class with
        code 200 if output["success_flg"] is True else 400.
        """
        if model_type == "all":
            result = pipeline.fit_hyperparams()
        else:
            result = pipeline.set_hyperparams(model_type, request.args.to_dict())
        response = make_response(jsonify(result), int(200 * result["success_flg"] + 400 * (1 - result["success_flg"])))
        response.headers["Content-Type"] = "application/json"
        return response


model_true = api.model('predict_true_model', {
    "success_flg": fields.Boolean(description="Флаг успешного выполнения", example=True),
    "data": fields.List(fields.Float, description="Список с предсказаниями", example=[100, 134, 18.9, 130, 45.1, 14.4]),
    "info": fields.String(description="Доп. информация о выполнении.", example="Predictions of linear model.")
})
model_false = api.model('predict_false_model', {
    "success_flg": fields.Boolean(description="Флаг успешного выполнения", example=False),
    "info": fields.String(description="Доп. информация о выполнении.", example="No needed columns.")
})
upload_parser = reqparse.RequestParser()
upload_parser.add_argument('file', location='files', type=FileStorage, required=True)


@api.doc(
    params={'model_type': 'Тип модели'},
    description="Производит предсказание моделью выбранного типа на данных в переданном в запросе файле."
)
@api.expect(upload_parser)
@api.response(200, 'Успешное выполнение', model_true)
@api.response(400, 'Неуспешное выполнение (доп. информация предоставляется в поле "info")', model_false)
class PredictRuler(Resource):
    """Provide access to resource 'models predictions'."""

    def post(self, model_type):
        """
        Производит предсказание моделью выбранного типа

        Call method of MLPipeline object predicting target by chosen model
        type on provided dataset (see mlpipeline.MLPipeline.predict() documentation).

        Keyword arguments:
        model_type -- type of models for predictions.

        Return output of called method converted to Response class
        with code 200 if output["success_flg"] is True and provided file
        for predictions is correct else 400.
        """
        if 'file' not in request.files or request.files['file'].filename == '':
            result = {"success_flg": False, "info": "No file provided."}
        else:
            try:
                df = pd.read_csv(request.files['file'])
                result = pipeline.predict(df, model_type)
            except Exception:
                result = {
                    "success_flg": False,
                    "info": "Problems with your file (needed opened standard '.csv' file with dataset)."
                }
        response = make_response(jsonify(result), int(200 * result["success_flg"] + 400 * (1 - result["success_flg"])))
        response.headers["Content-Type"] = "application/json"
        return response


model_true = api.model('refit_true_model', {
    "success_flg": fields.Boolean(description="Флаг успешного выполнения", example=True),
    "info": fields.String(description="Доп. информация о выполнении.", example="Models are refitted on new data. Train dataset was changed.")
})
model_false = api.model('refit_false_model', {
    "success_flg": fields.Boolean(description="Флаг успешного выполнения", example=False),
    "info": fields.String(description="Доп. информация о выполнении.", example="No needed columns in new train dataset.")
})
upload_parser = reqparse.RequestParser()
upload_parser.add_argument('file', location='files', type=FileStorage, required=True)


@api.doc(description="Производит переобучение моделей на данных в переданном в запросе файле.")
@api.expect(upload_parser)
@api.response(200, 'Успешное выполнение', model_true)
@api.response(400, 'Неуспешное выполнение (доп. информация предоставляется в поле "info")', model_false)
class RefitRuler(Resource):
    """Provide access to resource 'models state'."""

    def post(self):
        """
        Производит переобучение моделей

        Call method of MLPipeline object refiting on provided dataset
        (see mlpipeline.MLPipeline.fit_new_data() documentation).

        Return output of called method converted to Response class with
        code 200 if output["success_flg"] is True and provided file
        for training is correct else 400.
        """
        if 'file' not in request.files or request.files['file'].filename == '':
            result = {"success_flg": False, "info": "No file provided."}
        else:
            try:
                df = pd.read_csv(request.files['file'])
                result = pipeline.fit_new_data(df.drop("target", axis=1), df["target"])
            except Exception:
                result = {
                    "success_flg": False,
                    "info": "Problems with your file (needed opened standard '.csv' file with dataset which has 'target' column)."
                }
        response = make_response(jsonify(result), int(200 * result["success_flg"] + 400 * (1 - result["success_flg"])))
        response.headers["Content-Type"] = "application/json"
        return response


api.add_resource(ModelsTypesGetter, "/api/models_types")
api.add_resource(ModelsDeleter, "/api/delete/<string:model_type>")
api.add_resource(HyperparamsRuler, "/api/hyperparams/<string:model_type>")
api.add_resource(PredictRuler, "/api/predict/<string:model_type>")
api.add_resource(RefitRuler, "/api/refit")
api.init_app(app)

if __name__ == "__main__":
    app.run(debug=True, port=5000, host="127.0.0.1")
