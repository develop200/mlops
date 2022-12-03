from .base import db


class MLLinearModel(db.Model):
    """Provide interface for synchronization Python
    objects and postgre table containing linear models."""

    __tablename__ = "mllinearmodel"

    id = db.Column(db.Integer, unique=True, primary_key=True)
    data = db.Column(db.PickleType, nullable=False)

    def __init__(self, data):
        self.data = data
