"""
Módulo de comparadores para clasificadores dermatoscópicos.

Este módulo contiene herramientas para comparar el rendimiento de diferentes
algoritmos de clasificación aplicados a imágenes dermatoscópicas.

Componentes principales:
- ComparadorTriple: Compara RGB, PCA y K-Means simultáneamente
- Utilidades para reportes y análisis comparativo
"""

from .comparador_triple import (
    ComparadorTriple,
    ResultadoComparacion,
    ReporteTriple,
    ejecutar_comparacion_rapida
)

__all__ = [
    'ComparadorTriple',
    'ResultadoComparacion', 
    'ReporteTriple',
    'ejecutar_comparacion_rapida'
]