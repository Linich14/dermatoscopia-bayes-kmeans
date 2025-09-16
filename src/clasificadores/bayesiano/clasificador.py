"""
Clasificador Bayesiano RGB refactorizado.

Este módulo contiene la implementación principal del clasificador Bayesiano
que orquesta todos los componentes especializados siguiendo principios
de código limpio y responsabilidad única.
"""

import numpy as np
from typing import Dict, List, Any
from .base import IClasificador, ClasificadorBase
from .modelo import ModeloGaussianoMultivariado
from .umbrales import SelectorUmbral
from .evaluacion import EvaluadorClasificador


class ClasificadorBayesianoRGB(ClasificadorBase, IClasificador):
    """
    Clasificador Bayesiano RGB modular para segmentación de lesiones dermatoscópicas.
    
    Esta implementación refactorizada utiliza el patrón de composición para
    separar responsabilidades y mejorar la mantenibilidad del código.
    
    Attributes:
        criterio_umbral (str): Criterio utilizado para selección del umbral
        modelo_lesion (ModeloGaussianoMultivariado): Modelo para píxeles de lesión
        modelo_sana (ModeloGaussianoMultivariado): Modelo para píxeles sanos
        selector_umbral (ISelectorUmbral): Estrategia de selección de umbral
        evaluador (EvaluadorClasificador): Componente de evaluación
        umbral (float): Umbral de decisión seleccionado
        prior_lesion (float): Probabilidad a priori de lesión
        prior_sana (float): Probabilidad a priori de piel sana
    """
    
    def __init__(self, criterio_umbral: str = 'youden'):
        """
        Inicializa el clasificador Bayesiano RGB modular.
        
        Args:
            criterio_umbral: Criterio para selección del umbral óptimo
                          ('youden', 'equal_error', 'prior_balanced')
        """
        super().__init__()
        
        # Validar criterio
        criterios_validos = ['youden', 'equal_error', 'prior_balanced']
        if criterio_umbral not in criterios_validos:
            raise ValueError(f"Criterio '{criterio_umbral}' no válido. "
                           f"Opciones: {criterios_validos}")
        
        self.criterio_umbral = criterio_umbral
        
        # Inicializar componentes
        self.modelo_lesion = ModeloGaussianoMultivariado()
        self.modelo_sana = ModeloGaussianoMultivariado()
        self.evaluador = EvaluadorClasificador()
        
        # Parámetros del clasificador
        self.umbral = None
        self.prior_lesion = None
        self.prior_sana = None
        self.selector_umbral = None
    
    def entrenar(self, datos_entrenamiento: List[Dict]) -> None:
        """
        Entrena el clasificador con datos de entrenamiento.
        
        Args:
            datos_entrenamiento: Lista de diccionarios con imágenes y máscaras
        """
        # Validar datos de entrada
        self._validar_datos_entrenamiento(datos_entrenamiento)
        
        print("Iniciando entrenamiento del clasificador Bayesiano...")
        
        # Extraer píxeles de cada clase
        pixels_lesion, pixels_sana = self._extraer_pixels_por_clase(datos_entrenamiento)
        
        # Entrenar modelos gaussianos
        print("Entrenando modelo para píxeles de lesión...")
        self.modelo_lesion.entrenar(pixels_lesion)
        
        print("Entrenando modelo para píxeles sanos...")
        self.modelo_sana.entrenar(pixels_sana)
        
        # Calcular probabilidades a priori
        total_pixels = len(pixels_lesion) + len(pixels_sana)
        self.prior_lesion = len(pixels_lesion) / total_pixels
        self.prior_sana = len(pixels_sana) / total_pixels
        
        print(f"Probabilidades a priori: P(lesión)={self.prior_lesion:.3f}, "
              f"P(sana)={self.prior_sana:.3f}")
        
        # Crear selector de umbral
        self._crear_selector_umbral()
        
        # Seleccionar umbral óptimo
        print(f"Seleccionando umbral óptimo con criterio '{self.criterio_umbral}'...")
        self._seleccionar_umbral_optimo(datos_entrenamiento)
        
        # Marcar como entrenado
        self._entrenado = True
        
        print("✅ Entrenamiento completado exitosamente")
    
    def clasificar(self, imagen: np.ndarray) -> np.ndarray:
        """
        Clasifica una imagen RGB.
        
        Args:
            imagen: Imagen RGB normalizada [0,1] de forma (H,W,3)
            
        Returns:
            Máscara binaria de clasificación (H,W)
        """
        self._validar_entrenado()
        self._validar_imagen(imagen)
        
        # Calcular razones de verosimilitud
        razones = self._calcular_razones_verosimilitud(imagen)
        
        # Aplicar umbral de decisión
        mascara_pred = (razones >= self.umbral).astype(np.uint8)
        
        return mascara_pred
    
    def evaluar(self, datos_test: List[Dict]) -> Dict[str, Any]:
        """
        Evalúa el clasificador en datos de prueba.
        
        Args:
            datos_test: Lista de diccionarios con imágenes y máscaras de prueba
            
        Returns:
            Diccionario con métricas de evaluación
        """
        self._validar_entrenado()
        
        return self.evaluador.evaluar_conjunto_imagenes(datos_test, self)
    
    def comparar_criterios(self, datos_validacion: List[Dict]) -> Dict[str, Dict[str, Any]]:
        """
        Compara todos los criterios de umbral disponibles.
        
        Args:
            datos_validacion: Datos para evaluar criterios
            
        Returns:
            Diccionario con resultados para cada criterio
        """
        self._validar_entrenado()
        
        # Guardar estado actual
        criterio_original = self.criterio_umbral
        umbral_original = self.umbral
        
        criterios = ['youden', 'equal_error', 'prior_balanced']
        resultados = {}
        
        print("Comparando criterios de umbral...")
        
        for criterio in criterios:
            print(f"Evaluando criterio: {criterio}")
            
            # Cambiar temporalmente el criterio
            self.criterio_umbral = criterio
            self._crear_selector_umbral()
            self._seleccionar_umbral_optimo(datos_validacion)
            
            # Evaluar con este criterio
            metricas = self.evaluador.evaluar_conjunto_imagenes(datos_validacion, self)
            
            # Guardar resultados
            resultados[criterio] = {
                'umbral': self.umbral,
                'metricas': metricas,
                'justificacion': self.selector_umbral.justificar()
            }
        
        # Restaurar estado original
        self.criterio_umbral = criterio_original
        self.umbral = umbral_original
        self._crear_selector_umbral()
        
        print("Comparación completada")
        return resultados
    
    def obtener_parametros(self) -> Dict[str, Any]:
        """
        Obtiene todos los parámetros del clasificador entrenado.
        
        Returns:
            Diccionario con parámetros completos del modelo
        """
        self._validar_entrenado()
        
        params_lesion = self.modelo_lesion.obtener_parametros()
        params_sana = self.modelo_sana.obtener_parametros()
        
        return {
            'criterio_umbral': self.criterio_umbral,
            'umbral': self.umbral,
            'prior_lesion': self.prior_lesion,
            'prior_sana': self.prior_sana,
            'mu_lesion': params_lesion['mu'],
            'cov_lesion': params_lesion['cov'],
            'mu_sana': params_sana['mu'],
            'cov_sana': params_sana['cov'],
            'entrenado': self._entrenado,
            'num_muestras_lesion': params_lesion['num_muestras'],
            'num_muestras_sana': params_sana['num_muestras']
        }
    
    def justificar_criterio_umbral(self) -> str:
        """
        Proporciona justificación del criterio de umbral utilizado.
        
        Returns:
            Explicación textual del criterio
        """
        if self.selector_umbral is None:
            return "Clasificador no entrenado"
        
        return self.selector_umbral.justificar()
    
    def _extraer_pixels_por_clase(self, datos: List[Dict]) -> tuple:
        """
        Extrae píxeles de lesión y sanos aplicando muestreo equilibrado.
        
        Implementa el requisito del proyecto de "muestreo equilibrado de píxeles 
        para entrenamiento" para evitar desbalance de clases.
        
        Args:
            datos: Datos de entrenamiento
            
        Returns:
            Tupla (pixels_lesion, pixels_sana) como arrays numpy balanceados
        """
        print("Aplicando muestreo equilibrado de píxeles...")
        
        # Importar función de muestreo equilibrado
        from ...muestreo.muestreo_equilibrado import muestreo_equilibrado
        
        # Aplicar muestreo equilibrado según condiciones del proyecto
        X_equilibrado, y_equilibrado = muestreo_equilibrado(datos)
        
        # Separar píxeles por clase
        pixels_lesion = X_equilibrado[y_equilibrado == 1]  # Píxeles de lesión
        pixels_sana = X_equilibrado[y_equilibrado == 0]    # Píxeles de piel sana
        
        print(f"Muestreo equilibrado completado: {len(pixels_lesion)} píxeles de lesión, "
              f"{len(pixels_sana)} píxeles sanos")
        
        return pixels_lesion, pixels_sana
    
    def _crear_selector_umbral(self) -> None:
        """Crea el selector de umbral según el criterio especificado."""
        if self.criterio_umbral == 'prior_balanced':
            self.selector_umbral = SelectorUmbral.crear(
                self.criterio_umbral,
                prior_lesion=self.prior_lesion,
                prior_sana=self.prior_sana
            )
        else:
            self.selector_umbral = SelectorUmbral.crear(self.criterio_umbral)
    
    def _seleccionar_umbral_optimo(self, datos_validacion: List[Dict]) -> None:
        """
        Selecciona el umbral óptimo usando datos de validación.
        
        Args:
            datos_validacion: Datos para optimizar el umbral
        """
        # Calcular razones de verosimilitud para validación
        razones_vero = []
        etiquetas_true = []
        
        for item in datos_validacion:
            imagen = item['imagen']
            mascara = item['mascara']
            
            razones = self._calcular_razones_verosimilitud(imagen)
            
            razones_vero.extend(razones.flatten())
            etiquetas_true.extend(mascara.flatten())
        
        razones_vero = np.array(razones_vero)
        etiquetas_true = np.array(etiquetas_true)
        
        # Seleccionar umbral óptimo
        self.umbral = self.selector_umbral.seleccionar(razones_vero, etiquetas_true)
        
        print(f"Umbral óptimo seleccionado: {self.umbral:.6f}")
    
    def _calcular_razones_verosimilitud(self, imagen: np.ndarray) -> np.ndarray:
        """
        Calcula razones de verosimilitud para cada píxel.
        
        Args:
            imagen: Imagen RGB de entrada
            
        Returns:
            Matriz de razones de verosimilitud
        """
        h, w, c = imagen.shape
        pixels = imagen.reshape(-1, 3)
        
        # Calcular verosimilitudes para cada clase
        likelihood_lesion = self.modelo_lesion.calcular_verosimilitud(pixels)
        likelihood_sana = self.modelo_sana.calcular_verosimilitud(pixels)
        
        # Evitar división por cero
        epsilon = 1e-10
        likelihood_sana = np.maximum(likelihood_sana, epsilon)
        
        # Calcular razón de verosimilitud
        razon_verosimilitud = likelihood_lesion / likelihood_sana
        
        # Reshape a forma original
        return razon_verosimilitud.reshape(h, w)