"""
Implementación del modelo gaussiano multivariado para clasificación Bayesiana.

Este módulo contiene la implementación específica del modelo estadístico
utilizado para modelar las distribuciones de píxeles de lesión y piel sana
en el espacio de color RGB.
"""

import numpy as np
from scipy.stats import multivariate_normal
from typing import Dict, Any
from .base import IModelo, ModeloBase


class ModeloGaussianoMultivariado(ModeloBase, IModelo):
    """
    Modelo gaussiano multivariado para distribuciones RGB.
    
    Implementa un modelo de distribución gaussiana multivariada que estima
    los parámetros de media y covarianza para modelar la distribución de
    píxeles en el espacio de color RGB.
    
    Attributes:
        mu (np.ndarray): Vector de medias [R, G, B]
        cov (np.ndarray): Matriz de covarianza 3x3
        regularization (float): Factor de regularización para estabilidad numérica
    """
    
    def __init__(self, regularization: float = 1e-6):
        """
        Inicializa el modelo gaussiano multivariado.
        
        Args:
            regularization: Factor de regularización para matrices de covarianza
        """
        super().__init__()
        self.regularization = regularization
        self.mu = None
        self.cov = None
    
    def entrenar(self, datos: np.ndarray) -> None:
        """
        Entrena el modelo estimando parámetros de la distribución gaussiana.
        
        Estima los parámetros de media y covarianza de la distribución
        gaussiana multivariada que mejor modela los datos de entrenamiento.
        
        Args:
            datos: Array de forma (N, 3) con valores RGB de píxeles
            
        Raises:
            ValueError: Si los datos están vacíos o mal formateados
            np.linalg.LinAlgError: Si la matriz de covarianza es singular
        """
        # Validar datos de entrada
        if datos.size == 0:
            raise ValueError("Los datos de entrenamiento están vacíos")
        
        if len(datos.shape) != 2 or datos.shape[1] != 3:
            raise ValueError("Los datos deben tener forma (N, 3) para valores RGB")
        
        if datos.shape[0] < 4:  # Mínimo para estimar covarianza 3x3
            raise ValueError(f"Insuficientes muestras para entrenar: {datos.shape[0]} < 4")
        
        # Estimar parámetros
        self.mu = np.mean(datos, axis=0)
        self.cov = np.cov(datos.T)
        
        # Aplicar regularización para estabilidad numérica
        self.cov += self.regularization * np.eye(3)
        
        # Verificar que la matriz de covarianza es válida
        try:
            np.linalg.cholesky(self.cov)
        except np.linalg.LinAlgError:
            raise np.linalg.LinAlgError(
                "La matriz de covarianza no es definida positiva después de regularización"
            )
        
        # Actualizar estado y parámetros
        self._entrenado = True
        self._parametros = {
            'mu': self.mu.copy(),
            'cov': self.cov.copy(),
            'regularization': self.regularization,
            'num_muestras': datos.shape[0]
        }
    
    def calcular_verosimilitud(self, datos: np.ndarray) -> np.ndarray:
        """
        Calcula la verosimilitud de los datos bajo la distribución gaussiana.
        
        Evalúa la función de densidad de probabilidad de la distribución
        gaussiana multivariada en los puntos proporcionados.
        
        Args:
            datos: Array de forma (N, 3) con valores RGB a evaluar
            
        Returns:
            Array de forma (N,) con las verosimilitudes calculadas
            
        Raises:
            RuntimeError: Si el modelo no ha sido entrenado
            ValueError: Si los datos tienen formato incorrecto
        """
        self._validar_entrenado()
        
        # Validar formato de datos
        if len(datos.shape) != 2 or datos.shape[1] != 3:
            raise ValueError("Los datos deben tener forma (N, 3) para valores RGB")
        
        # Calcular verosimilitud usando scipy
        try:
            verosimilitud = multivariate_normal.pdf(datos, mean=self.mu, cov=self.cov)
        except np.linalg.LinAlgError as e:
            raise RuntimeError(f"Error al calcular verosimilitud: {e}")
        
        return verosimilitud
    
    def obtener_parametros(self) -> Dict[str, Any]:
        """
        Obtiene los parámetros del modelo entrenado.
        
        Returns:
            Diccionario con parámetros del modelo:
            - mu: Vector de medias
            - cov: Matriz de covarianza  
            - regularization: Factor de regularización
            - num_muestras: Número de muestras de entrenamiento
            
        Raises:
            RuntimeError: Si el modelo no ha sido entrenado
        """
        self._validar_entrenado()
        return self.parametros
    
    def calcular_distancia_mahalanobis(self, datos: np.ndarray) -> np.ndarray:
        """
        Calcula la distancia de Mahalanobis de los datos al centro de la distribución.
        
        La distancia de Mahalanobis es útil para detectar outliers y
        entender qué tan "típicos" son los datos bajo el modelo.
        
        Args:
            datos: Array de forma (N, 3) con valores RGB
            
        Returns:
            Array de forma (N,) con las distancias de Mahalanobis
            
        Raises:
            RuntimeError: Si el modelo no ha sido entrenado
        """
        self._validar_entrenado()
        
        # Centrar los datos
        datos_centrados = datos - self.mu
        
        # Calcular distancia de Mahalanobis
        # D²(x) = (x - μ)ᵀ Σ⁻¹ (x - μ)
        cov_inv = np.linalg.inv(self.cov)
        distancias = np.sqrt(np.sum(datos_centrados @ cov_inv * datos_centrados, axis=1))
        
        return distancias
    
    def log_verosimilitud(self, datos: np.ndarray) -> np.ndarray:
        """
        Calcula el logaritmo de la verosimilitud para mejor estabilidad numérica.
        
        Args:
            datos: Array de forma (N, 3) con valores RGB
            
        Returns:
            Array de forma (N,) con log-verosimilitudes
        """
        self._validar_entrenado()
        
        try:
            log_verosimilitud = multivariate_normal.logpdf(datos, mean=self.mu, cov=self.cov)
        except np.linalg.LinAlgError as e:
            raise RuntimeError(f"Error al calcular log-verosimilitud: {e}")
        
        return log_verosimilitud
    
    def obtener_elipse_confianza(self, nivel_confianza: float = 0.95) -> Dict[str, Any]:
        """
        Calcula parámetros de la elipse de confianza en 2D (proyección RG).
        
        Útil para visualización de la distribución en el espacio de color.
        
        Args:
            nivel_confianza: Nivel de confianza deseado (0-1)
            
        Returns:
            Diccionario con parámetros de la elipse:
            - centro: Centro de la elipse [R, G]
            - ancho: Ancho de la elipse
            - alto: Alto de la elipse  
            - angulo: Ángulo de rotación en radianes
        """
        self._validar_entrenado()
        
        from scipy.stats import chi2
        
        # Usar solo componentes R y G para visualización 2D
        mu_2d = self.mu[:2]
        cov_2d = self.cov[:2, :2]
        
        # Calcular eigenvalores y eigenvectores
        eigenvals, eigenvecs = np.linalg.eigh(cov_2d)
        
        # Factor de escala según nivel de confianza
        chi2_val = chi2.ppf(nivel_confianza, df=2)
        scale = np.sqrt(chi2_val)
        
        # Parámetros de la elipse
        ancho = 2 * scale * np.sqrt(eigenvals[1])  # Eje mayor
        alto = 2 * scale * np.sqrt(eigenvals[0])   # Eje menor
        angulo = np.arctan2(eigenvecs[1, 1], eigenvecs[0, 1])
        
        return {
            'centro': mu_2d,
            'ancho': ancho,
            'alto': alto,
            'angulo': angulo,
            'nivel_confianza': nivel_confianza
        }