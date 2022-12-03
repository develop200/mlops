"""Operate with Flask application.

Module provides classes and functions which enable to
operate with Flask application (create, connect with
database and API definition).

Classes:

    ModelsTypesGetter
    ModelsDeleter
    Predicter
    Trainer

Functions:

    create_app
"""

from flask import Flask
import os
from sqlalchemy_utils import database_exists, create_database
from .model import db


def create_app():
    """Create Flask application, connect it
    with database and api definition.

    Return ready application object.
    """
    app = Flask(__name__)
    app.app_context().push()

    config = {
        "SECRET_KEY": os.environ.get("SECRET_KEY"),
        "SQLALCHEMY_DATABASE_URI": os.environ.get("SQLALCHEMY_DATABASE_URI"),
        "DEBUG": bool(os.environ.get("DEBUG"))
    }
    app.config.from_mapping(**config)

    db_url = app.config["SQLALCHEMY_DATABASE_URI"]
    if not database_exists(db_url):
        create_database(db_url)

    db.init_app(app)
    db.create_all()

    from .api import api
    api.init_app(app)

    return app
