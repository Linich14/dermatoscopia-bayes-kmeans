"""
Interfaces y clases base para componentes del clasificador Bayesiano.

Este módulo define las abstracciones fundamentales que deben implementar
todos los componentes del sistema de clasificación, siguiendo principios
de inversión de dependencias y programación orientada a interfaces.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Protocol
import numpy as np


class IModelo(ABC):
    """
    Interfaz para modelos de distribución de probabilidad.
    
    Define el contrato que deben cumplir todos los modelos estadísticos
    utilizados en la clasificación Bayesiana.
    """
    
    @abstractmethod
    def entrenar(self, datos: np.ndarray) -> None:
        """
        Entrena el modelo con los datos proporcionados.
        
        Args:
            datos: Array de características de entrenamiento
        """
        pass
    
    @abstractmethod
    def calcular_verosimilitud(self, datos: np.ndarray) -> np.ndarray:
        """
        Calcula la verosimilitud de los datos bajo el modelo.
        
        Args:
            datos: Array de datos para evaluar
            
        Returns:
            Array de verosimilitudes calculadas
        """
        pass
    
    @abstractmethod
    def obtener_parametros(self) -> Dict[str, Any]:
        """
        Obtiene los parámetros del modelo entrenado.
        
        Returns:
            Diccionario con parámetros del modelo
        """
        pass


class ISelectorUmbral(ABC):
    """
    Interfaz para estrategias de selección de umbral.
    
    Define el contrato para algoritmos de optimización de umbral
    de decisión en clasificación binaria.
    """
    
    @abstractmethod
    def seleccionar(self, razones_verosimilitud: np.ndarray, 
                   etiquetas_reales: np.ndarray) -> float:
        """
        Selecciona el umbral óptimo según la estrategia implementada.
        
        Args:
            razones_verosimilitud: Valores de razón de verosimilitud
            etiquetas_reales: Etiquetas ground truth
            
        Returns:
            Umbral óptimo seleccionado
        """
        pass
    
    @abstractmethod
    def justificar(self) -> str:
        """
        Proporciona justificación textual del criterio utilizado.
        
        Returns:
            Explicación del criterio de selección
        """
        pass


class IEvaluador(ABC):
    """
    Interfaz para evaluadores de rendimiento de clasificadores.
    
    Define el contrato para componentes que calculan métricas
    de evaluación y análisis de rendimiento.
    """
    
    @abstractmethod
    def evaluar(self, predicciones: np.ndarray, 
               etiquetas_reales: np.ndarray) -> Dict[str, Any]:
        """
        Evalúa el rendimiento del clasificador.
        
        Args:
            predicciones: Predicciones del clasificador
            etiquetas_reales: Etiquetas ground truth
            
        Returns:
            Diccionario con métricas de evaluación
        """
        pass


class IClasificador(ABC):
    """
    Interfaz principal para clasificadores.
    
    Define el contrato básico que deben cumplir todos los
    clasificadores del sistema.
    """
    
    @abstractmethod
    def entrenar(self, datos_entrenamiento: List[Dict]) -> None:
        """
        Entrena el clasificador con datos de entrenamiento.
        
        Args:
            datos_entrenamiento: Lista de diccionarios con imágenes y máscaras
        """
        pass
    
    @abstractmethod
    def clasificar(self, imagen: np.ndarray) -> np.ndarray:
        """
        Clasifica una imagen.
        
        Args:
            imagen: Imagen RGB a clasificar
            
        Returns:
            Máscara binaria de clasificación
        """
        pass
    
    @abstractmethod
    def evaluar(self, datos_test: List[Dict]) -> Dict[str, Any]:
        """
        Evalúa el clasificador en datos de prueba.
        
        Args:
            datos_test: Datos de prueba
            
        Returns:
            Métricas de evaluación
        """
        pass


class ModeloBase(ABC):
    """
    Clase base abstracta para modelos estadísticos.
    
    Proporciona funcionalidad común y estructura base
    para implementaciones específicas de modelos.
    """
    
    def __init__(self):
        self._entrenado = False
        self._parametros = {}
    
    @property
    def entrenado(self) -> bool:
        """Indica si el modelo ha sido entrenado."""
        return self._entrenado
    
    @property
    def parametros(self) -> Dict[str, Any]:
        """Parámetros del modelo entrenado."""
        return self._parametros.copy()
    
    def _validar_entrenado(self) -> None:
        """Valida que el modelo esté entrenado antes de usarlo."""
        if not self._entrenado:
            raise RuntimeError("El modelo debe ser entrenado antes de usarlo")


class ClasificadorBase(ABC):
    """
    Clase base abstracta para clasificadores.
    
    Proporciona estructura común y validaciones básicas
    para implementaciones específicas de clasificadores.
    """
    
    def __init__(self):
        self._entrenado = False
    
    @property
    def entrenado(self) -> bool:
        """Indica si el clasificador ha sido entrenado."""
        return self._entrenado
    
    def _validar_entrenado(self) -> None:
        """Valida que el clasificador esté entrenado."""
        if not self._entrenado:
            raise RuntimeError("El clasificador debe ser entrenado antes de usarlo")
    
    def _validar_datos_entrenamiento(self, datos: List[Dict]) -> None:
        """Valida la estructura de datos de entrenamiento."""
        if not datos:
            raise ValueError("Se requieren datos de entrenamiento")
        
        for i, item in enumerate(datos):
            if not isinstance(item, dict):
                raise ValueError(f"Elemento {i} no es un diccionario")
            if 'imagen' not in item or 'mascara' not in item:
                raise ValueError(f"Elemento {i} debe contener 'imagen' y 'mascara'")
    
    def _validar_imagen(self, imagen: np.ndarray) -> None:
        """Valida formato de imagen de entrada."""
        if not isinstance(imagen, np.ndarray):
            raise TypeError("La imagen debe ser un array numpy")
        if len(imagen.shape) != 3 or imagen.shape[2] != 3:
            raise ValueError("La imagen debe tener forma (H, W, 3)")
        if imagen.min() < 0 or imagen.max() > 1:
            raise ValueError("Los valores de la imagen deben estar en [0, 1]")