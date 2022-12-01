from .base import db

class MLForestModel(db.Model):
    __tablename__ = "mlforestmodel"

    id = db.Column(db.Integer, unique=True, primary_key=True)
    data = db.Column(db.PickleType, nullable=False)

    def __init__(self, data):
        self.data = data
