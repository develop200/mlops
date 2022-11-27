from flask import make_response, jsonify, request
from flask_restx import Api, Resource, fields, reqparse
from werkzeug.datastructures import FileStorage
from mlpipeline import MLPipeline
import pandas as pd

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
    "data": fields.Raw(description="Список доступных моделей", example={"linear": [0, 1, 2], "forest": [0, 1]})
})


@api.doc(description="Возвращает список доступных моделей.")
@api.response(200, 'Успешное выполнение', model)
class ModelsTypesGetter(Resource):
    def get(self):
        response = make_response(
            jsonify(
                pipeline.get_avaliable_model_classes()),
            200)
        response.headers["Content-Type"] = "application/json"
        return response


model_true = api.model('deleter_true_model', {
    "success_flg": fields.Boolean(description="Флаг успешного выполнения", example=True),
    "available_models": fields.Raw(description="Список доступных моделей", example={"linear": [0, 1, 2], "forest": [0, 1]}),
    "info": fields.String(description="Доп. информация о выполнении.", example="Model 3 of type linear was deleted.")
})
model_false = api.model('deleter_false_model', {
    "success_flg": fields.Boolean(description="Флаг успешного выполнения", example=False),
    "info": fields.String(description="Доп. информация о выполнении.", example="Model type boosting is not available.")
})
deleter_parser = reqparse.RequestParser()
deleter_parser.add_argument(
    'model_type',
    location='args',
    type=str,
    required=True,
    help='Тип модели')
deleter_parser.add_argument(
    'model_id',
    location='args',
    type=int,
    required=True,
    help='Номер модели')


@api.doc(description="Производит удаление модели по её типу и номеру.")
@api.expect(deleter_parser)
@api.response(200, 'Успешное выполнение', model_true)
@api.response(400, 'Неуспешное выполнение (доп. информация предоставляется в поле "info")', model_false)
class ModelsDeleter(Resource):
    def delete(self):
        args = deleter_parser.parse_args()
        model_type = args["model_type"]
        model_id = args["model_id"]
        result = pipeline.delete_model(model_type, model_id)
        response = make_response(jsonify(result), int(
            200 * result["success_flg"] + 400 * (1 - result["success_flg"])))
        response.headers["Content-Type"] = "application/json"
        return response


model_true = api.model('predict_true_model', {
    "success_flg": fields.Boolean(description="Флаг успешного выполнения", example=True),
    "data": fields.List(fields.Float, description="Предсказания", example={"0": 100, "1": 134, "2": 18.9, "3": 130, "4": 45.1, "5": 14.4}),
    "info": fields.String(description="Доп. информация о выполнении.", example="Predictions of linear model with id = 3.")
})
model_false = api.model('predict_false_model', {
    "success_flg": fields.Boolean(description="Флаг успешного выполнения", example=False),
    "info": fields.String(description="Доп. информация о выполнении.", example="No needed columns.")
})
predicter_parser = reqparse.RequestParser()
predicter_parser.add_argument(
    'file',
    location='files',
    type=FileStorage,
    required=True,
    help="JSON файл с объектами для предсказания (пример example.json)")
predicter_parser.add_argument(
    'model_type',
    location='args',
    type=str,
    required=True,
    help='Тип модели')
predicter_parser.add_argument(
    'model_id',
    location='args',
    type=int,
    required=True,
    help='Номер модели')


@api.doc(description="Производит предсказание выбранной моделью на данных в переданном в запросе файле.")
@api.expect(predicter_parser)
@api.response(200, 'Успешное выполнение', model_true)
@api.response(400, 'Неуспешное выполнение (доп. информация предоставляется в поле "info")', model_false)
class Predicter(Resource):
    def post(self):
        if 'file' not in request.files or request.files['file'].filename == '':
            result = {"success_flg": False, "info": "No file provided."}
        else:
            try:
                args = predicter_parser.parse_args()
                model_type = args["model_type"]
                model_id = args["model_id"]
                file = args["file"]
                df = pd.read_json(file, orient="index")
                result = pipeline.predict(df, model_type, model_id)
            except Exception:
                result = {
                    "success_flg": False,
                    "info": "Problems with your file (needed opened '.json' file with batch)."
                }
        response = make_response(jsonify(result), int(
            200 * result["success_flg"] + 400 * (1 - result["success_flg"])))
        response.headers["Content-Type"] = "application/json"
        return response


model_true = api.model('train_true_model', {
    "success_flg": fields.Boolean(description="Флаг успешного выполнения", example=True),
    "info": fields.String(description="Доп. информация о выполнении.", example="Model of type linear with your/optuna hyperparams was trained. It's id - 3.")
})
model_false = api.model('train_false_model', {
    "success_flg": fields.Boolean(description="Флаг успешного выполнения", example=False),
    "info": fields.String(description="Доп. информация о выполнении.", example="Bad input of hyperparams.")
})
upload_parser = reqparse.RequestParser()
upload_parser.add_argument(
    'file',
    location='files',
    type=FileStorage,
    required=True,
    help='JSON файл с обучающей выборкой (пример example.json)')
upload_parser.add_argument(
    'model_type',
    location='args',
    type=str,
    required=True,
    help='Тип создаваемой модели')
upload_parser.add_argument(
    'n_estimators',
    location='args',
    type=int,
    required=False,
    help='Количество деревьев в ансамбле (необязательный параметр, актуален только для типа модели "forest").')
upload_parser.add_argument(
    'max_depth',
    location='args',
    type=int,
    required=False,
    help='Максимальная глубина деревьев в ансамбле (необязательный параметр, актуален только для типа модели "forest").')
upload_parser.add_argument(
    'min_samples_split',
    location='args',
    type=int,
    required=False,
    help='Минимальное количество наблюдений для разделения узла на листы в деревьях ансамбля (необязательный параметр, актуален только для типа модели "forest").')
upload_parser.add_argument(
    'max_features',
    location='args',
    type=float,
    required=False,
    help='Доля признаков, использующихся при построении моделей бэггинга (необязательный параметр, актуален только для типа модели "forest").')
upload_parser.add_argument(
    'alpha',
    location='args',
    type=float,
    required=False,
    help='Коэффициент регуляризации (необязательный параметр, актуален только для типа модели "linear").')


@api.doc(
    description="Производит обучение новой модели выбранного типа на переданных данных в формате JSON с переданными гиперпараметрами. Если гиперпараметры не были переданы, то их подбор осуществляется автоматически с помощью optuna."
)
@api.expect(upload_parser)
@api.response(200, 'Успешное выполнение', model_true)
@api.response(400, 'Неуспешное выполнение (доп. информация предоставляется в поле "info")', model_false)
class Trainer(Resource):
    def post(self):
        if 'file' not in request.files or request.files['file'].filename == '':
            result = {"success_flg": False, "info": "No file provided."}
        else:
            try:
                params = dict(upload_parser.parse_args())
                file = params["file"]
                model_type = params["model_type"]
                items = list(params.items())
                for key, value in items:
                    if key in ["file", "model_type"] or value is None:
                        del params[key]
                df = pd.read_json(file, orient="index")
                if len(params) == 0:
                    params = None
                result = pipeline.train_model(model_type, df, params)
            except Exception:
                result = {
                    "success_flg": False,
                    "info": "Problems with your file (needed opened '.json' file with train dataset)."
                }
        response = make_response(jsonify(result), int(
            200 * result["success_flg"] + 400 * (1 - result["success_flg"])))
        response.headers["Content-Type"] = "application/json"
        return response


api.add_resource(ModelsTypesGetter, "/api/models_types")
api.add_resource(ModelsDeleter, "/api/delete")
api.add_resource(Predicter, "/api/predict")
api.add_resource(Trainer, "/api/train")
