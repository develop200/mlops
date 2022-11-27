from flask import Flask
import os
import sys
sys.path.insert(1, './src/')
from api import api

app = Flask(__name__)
api.init_app(app)


if __name__ == "__main__":
    app.run(port=int(os.environ.get("PORT", 5000)), host="0.0.0.0")
