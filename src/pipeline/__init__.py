"""Operate with pretrained ML models.

Module provides classes which enable to fit and tune pretrained
ML models, get their predictions for user's data and operate with
them (create, read, delete).

Classes:

    MLPipeline
"""

from .mlpipeline import MLPipeline

__all__ = ["MLPipeline"]
