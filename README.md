# mlops
Проект по созданию ML-микросервиса, предоставляющего доступ к базе данных предобученных моделей для задачи определения веса рыбы через API. API предоставляет средства доступа к существующим моделям для совершения предсказаний на пользовательских данных и созданию новых моделей. Пример данных для предсказания или обучения новых моделей расположен в *example.json*.

### Запуск с помощью docker-compose:
```bash
    docker-compose up
```
Далее сервис будет доступен на host-машине (http://127.0.0.1:8888/). Для сохранения данных пользователя при перезапуске через volume настроено хранение базы данных также на host-машине (папка postgres-data).

### Поднятие сервиса (для примера в minikube):

1. Поднимаем кластер:
```bash
    minikube start
```

2. Скачиваем на нод(е/ах) кластера необходимые образы (лучше сейчас, так как при скачивании образов при запуске из файла с конфигурацей возможна ошибка из-за timeout, так как образы большие):
```bash
    minikube ssh
    docker pull romal200/mlops
    docker pull postgres:10
    exit
```

3. Деплой:
```bash
    kubectl apply -f deployment.yaml
```
После данной команды нужно убедиться, что pod-ы подняты (займёт какое-то время, небольшое, если образы были скачаны на предыдущем шаге).

4. Если вы поднимаете сервис на minikube, то для того чтобы иметь возможность использовать LoadBalancer и увидеть сервис на host-машине, необходимо выполнить команду (в соседнем окне):
```bash
    minikube tunnel
```

Теперь сервис доступен на host-машине (http://127.0.0.1).

### Тестирование:
```bash
    pytest . -v
```

