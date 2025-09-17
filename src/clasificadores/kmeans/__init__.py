"""
Módulo de clasificación no supervisada K-Means para análisis dermatoscópico.

Este módulo implementa el algoritmo K-Means aplicado a imágenes dermatoscópicas
con selección automática de características según los requisitos del proyecto.

FUNCIONES PRINCIPALES:
- Aplicar K-Means sobre conjunto de test con selección de características
- Evaluar diferentes combinaciones de características (RGB, HSV, textura, etc.)
- Reportar la mejor combinación de características encontrada
- Integración modular con la interfaz gráfica existente

ESTRUCTURA DEL MÓDULO:
- clasificador.py: Implementación principal del clasificador K-Means
- seleccion_caracteristicas.py: Sistema modular de selección de características

REQUISITOS CUMPLIDOS:
✅ Aplicar K-Means sobre cada imagen del conjunto de test
✅ Considerar selección de características
✅ Reportar resultado con mejor combinación de características
✅ Integración modular en diseño global de interfaz
"""

from .clasificador import KMeansClasificador, ResultadoKMeans, EvaluacionCombinacion
from .seleccion_caracteristicas import (
    SelectorCaracteristicas, 
    TipoCaracteristica,
    ConfiguracionCaracteristicas,
    crear_configuraciones_combinaciones
)

__all__ = [
    'KMeansClasificador',
    'SelectorCaracteristicas', 
    'TipoCaracteristica',
    'ResultadoKMeans',
    'EvaluacionCombinacion',
    'ConfiguracionCaracteristicas',
    'crear_configuraciones_combinaciones'
]