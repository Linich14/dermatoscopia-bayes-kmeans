"""
Estrategias de selección de umbral para clasificación Bayesiana.

Este módulo implementa diferentes criterios para la selección óptima
del umbral de decisión en clasificación binaria, cada uno optimizando
diferentes aspectos del rendimiento del clasificador.
"""

import numpy as np
from typing import Dict, Any
from abc import ABC
from .base import ISelectorUmbral


class SelectorUmbralYouden(ISelectorUmbral):
    """
    Selector de umbral basado en el índice de Youden.
    
    El índice de Youden (J = Sensibilidad + Especificidad - 1) maximiza
    la capacidad discriminativa del clasificador balanceando la detección
    de verdaderos positivos y la correcta identificación de negativos.
    """
    
    def seleccionar(self, razones_verosimilitud: np.ndarray, 
                   etiquetas_reales: np.ndarray) -> float:
        """
        Selecciona el umbral que maximiza el índice de Youden.
        
        Args:
            razones_verosimilitud: Valores de razón de verosimilitud
            etiquetas_reales: Etiquetas verdaderas (0=sana, 1=lesión)
            
        Returns:
            Umbral óptimo que maximiza J = Sensibilidad + Especificidad - 1
        """
        # Generar umbrales candidatos basados en percentiles
        umbrales_candidatos = np.percentile(razones_verosimilitud, np.linspace(1, 99, 100))
        
        mejor_j = -1
        mejor_umbral = 1.0
        
        for umbral in umbrales_candidatos:
            predicciones = (razones_verosimilitud >= umbral).astype(int)
            
            # Calcular matriz de confusión
            tp = np.sum((predicciones == 1) & (etiquetas_reales == 1))
            fp = np.sum((predicciones == 1) & (etiquetas_reales == 0))
            tn = np.sum((predicciones == 0) & (etiquetas_reales == 0))
            fn = np.sum((predicciones == 0) & (etiquetas_reales == 1))
            
            # Evitar división por cero
            if (tp + fn) == 0 or (tn + fp) == 0:
                continue
            
            # Calcular métricas
            sensibilidad = tp / (tp + fn)
            especificidad = tn / (tn + fp)
            j = sensibilidad + especificidad - 1
            
            if j > mejor_j:
                mejor_j = j
                mejor_umbral = umbral
        
        return mejor_umbral
    
    def justificar(self) -> str:
        """Justifica el criterio de Youden."""
        return """
        El índice de Youden maximiza la suma de sensibilidad y especificidad (TPR + TNR - 1).
        Es ideal para aplicaciones médicas donde se busca un balance óptimo entre la detección
        de casos positivos y la correcta identificación de casos negativos, minimizando tanto
        falsos positivos como falsos negativos de manera equilibrada.
        """.strip()


class SelectorUmbralEqualError(ISelectorUmbral):
    """
    Selector de umbral basado en Equal Error Rate (EER).
    
    Busca el punto donde la tasa de falsos positivos es igual
    a la tasa de falsos negativos, proporcionando un balance
    simétrico en los tipos de error.
    """
    
    def seleccionar(self, razones_verosimilitud: np.ndarray, 
                   etiquetas_reales: np.ndarray) -> float:
        """
        Selecciona el umbral que minimiza |FPR - FNR|.
        
        Args:
            razones_verosimilitud: Valores de razón de verosimilitud
            etiquetas_reales: Etiquetas verdaderas
            
        Returns:
            Umbral que minimiza la diferencia entre FPR y FNR
        """
        umbrales_candidatos = np.percentile(razones_verosimilitud, np.linspace(1, 99, 100))
        
        menor_diferencia = float('inf')
        mejor_umbral = 1.0
        
        for umbral in umbrales_candidatos:
            predicciones = (razones_verosimilitud >= umbral).astype(int)
            
            # Calcular matriz de confusión
            tp = np.sum((predicciones == 1) & (etiquetas_reales == 1))
            fp = np.sum((predicciones == 1) & (etiquetas_reales == 0))
            tn = np.sum((predicciones == 0) & (etiquetas_reales == 0))
            fn = np.sum((predicciones == 0) & (etiquetas_reales == 1))
            
            # Evitar división por cero
            if (tp + fn) == 0 or (tn + fp) == 0:
                continue
            
            # Calcular tasas de error
            fpr = fp / (fp + tn)  # Tasa de falsos positivos
            fnr = fn / (fn + tp)  # Tasa de falsos negativos
            diferencia = abs(fpr - fnr)
            
            if diferencia < menor_diferencia:
                menor_diferencia = diferencia
                mejor_umbral = umbral
        
        return mejor_umbral
    
    def justificar(self) -> str:
        """Justifica el criterio Equal Error Rate."""
        return """
        El criterio Equal Error Rate busca el punto donde la tasa de falsos positivos
        es igual a la tasa de falsos negativos (FPR = FNR). Es útil cuando los costos
        de ambos tipos de error son similares y se desea un clasificador equilibrado
        en términos de errores simétricos.
        """.strip()


class SelectorUmbralPriorBalanced(ISelectorUmbral):
    """
    Selector de umbral basado en probabilidades a priori balanceadas.
    
    Utiliza el umbral teórico derivado de la teoría de decisión Bayesiana
    cuando se asumen probabilidades a priori balanceadas y costos uniformes.
    """
    
    def __init__(self, prior_lesion: float, prior_sana: float):
        """
        Inicializa el selector con probabilidades a priori.
        
        Args:
            prior_lesion: Probabilidad a priori de lesión
            prior_sana: Probabilidad a priori de piel sana
        """
        self.prior_lesion = prior_lesion
        self.prior_sana = prior_sana
    
    def seleccionar(self, razones_verosimilitud: np.ndarray, 
                   etiquetas_reales: np.ndarray) -> float:
        """
        Selecciona el umbral basado en la relación de probabilidades a priori.
        
        Para clasificación Bayesiana óptima con costos uniformes,
        el umbral teórico es P(sana) / P(lesión).
        
        Args:
            razones_verosimilitud: Valores de razón de verosimilitud (no utilizados)
            etiquetas_reales: Etiquetas verdaderas (no utilizadas)
            
        Returns:
            Umbral teórico basado en probabilidades a priori
        """
        # Umbral teórico de la regla de decisión Bayesiana
        umbral_teorico = self.prior_sana / self.prior_lesion
        return umbral_teorico
    
    def justificar(self) -> str:
        """Justifica el criterio Prior Balanced."""
        return """
        El criterio Prior Balanced ajusta el umbral considerando las probabilidades
        a priori de las clases, buscando un balance que refleje la distribución
        natural de los datos. Es apropiado cuando se desea mantener las proporciones
        observadas en la población de entrenamiento.
        """.strip()


class SelectorUmbral:
    """
    Factory para crear selectores de umbral según el criterio especificado.
    
    Proporciona una interfaz unificada para acceder a diferentes estrategias
    de selección de umbral manteniendo el principio de responsabilidad única.
    """
    
    ESTRATEGIAS_DISPONIBLES = {
        'youden': SelectorUmbralYouden,
        'equal_error': SelectorUmbralEqualError,
        'prior_balanced': SelectorUmbralPriorBalanced
    }
    
    @classmethod
    def crear(cls, criterio: str, **kwargs) -> ISelectorUmbral:
        """
        Crea un selector de umbral según el criterio especificado.
        
        Args:
            criterio: Nombre del criterio ('youden', 'equal_error', 'prior_balanced')
            **kwargs: Argumentos adicionales para criterios específicos
            
        Returns:
            Instancia del selector de umbral apropiado
            
        Raises:
            ValueError: Si el criterio no es válido
        """
        if criterio not in cls.ESTRATEGIAS_DISPONIBLES:
            criterios_validos = list(cls.ESTRATEGIAS_DISPONIBLES.keys())
            raise ValueError(f"Criterio '{criterio}' no válido. "
                           f"Opciones disponibles: {criterios_validos}")
        
        estrategia_class = cls.ESTRATEGIAS_DISPONIBLES[criterio]
        
        # Manejar argumentos específicos para cada estrategia
        if criterio == 'prior_balanced':
            if 'prior_lesion' not in kwargs or 'prior_sana' not in kwargs:
                raise ValueError("El criterio 'prior_balanced' requiere "
                               "'prior_lesion' y 'prior_sana' como argumentos")
            return estrategia_class(kwargs['prior_lesion'], kwargs['prior_sana'])
        else:
            return estrategia_class()
    
    @classmethod
    def listar_criterios(cls) -> Dict[str, str]:
        """
        Lista todos los criterios disponibles con sus descripciones.
        
        Returns:
            Diccionario con criterios y descripciones breves
        """
        return {
            'youden': 'Maximiza sensibilidad + especificidad',
            'equal_error': 'Equilibra falsos positivos y negativos',
            'prior_balanced': 'Considera probabilidades a priori'
        }
    
    @classmethod
    def obtener_justificacion(cls, criterio: str) -> str:
        """
        Obtiene la justificación de un criterio específico.
        
        Args:
            criterio: Nombre del criterio
            
        Returns:
            Justificación textual del criterio
        """
        if criterio not in cls.ESTRATEGIAS_DISPONIBLES:
            return "Criterio no reconocido"
        
        # Crear instancia temporal para obtener justificación
        if criterio == 'prior_balanced':
            selector = cls.ESTRATEGIAS_DISPONIBLES[criterio](0.5, 0.5)
        else:
            selector = cls.ESTRATEGIAS_DISPONIBLES[criterio]()
        
        return selector.justificar()