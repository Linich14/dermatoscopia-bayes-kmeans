"""
Módulo de reducción de dimensionalidad especializado para análisis dermatoscópico.

Este módulo implementa técnicas de reducción de dimensionalidad adaptadas
específicamente para el análisis de imágenes dermatoscópicas, con énfasis
en la preservación de información discriminativa para clasificación.
"""

from .pca_especializado import (
    PCAAjustado,
    SelectorComponentesPCA,
    AnalizadorVarianza,
    JustificadorComponentes
)

__all__ = [
    'PCAAjustado',
    'SelectorComponentesPCA', 
    'AnalizadorVarianza',
    'JustificadorComponentes'
]