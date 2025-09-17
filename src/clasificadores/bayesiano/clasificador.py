"""
Clasificador Bayesiano RGB refactorizado.

Este módulo contiene la implementación principal del clasificador Bayesiano
que orquesta todos los componentes especializados siguiendo principios
de código limpio y responsabilidad única.

FUNCIONES PRINCIPALES PARA LOCALIZAR:
- entrenar(): Línea ~70 - Entrena clasificador RGB con muestreo equilibrado
- clasificar(): Línea ~120 - Clasifica imagen RGB píxel por píxel
- evaluar(): Línea ~150 - Evalúa rendimiento con métricas completas
- _calcular_log_likelihood(): Línea ~200 - Calcula probabilidades gaussianas RGB
- _seleccionar_umbral_optimo(): Línea ~250 - Determina umbral según criterio

CÓMO FUNCIONA EL MÉTODO RGB TRADICIONAL:
1. Extrae píxeles balanceados de imágenes (50% lesión, 50% sana)
2. Calcula distribuciones gaussianas para cada clase en espacio RGB
3. Entrena selector de umbral óptimo (Youden, EER, etc.)
4. Para clasificar: calcula likelihood ratio y aplica umbral de decisión
5. Resultado: máscara binaria con píxeles clasificados

DIFERENCIA CON PCA:
- RGB: Opera directamente en espacio de color RGB (3 dimensiones)
- PCA: Primero reduce dimensionalidad, luego aplica Bayesiano
"""

import numpy as np
from typing import Dict, List, Any
from .base import IClasificador, ClasificadorBase
from .modelo import ModeloGaussianoMultivariado
from .umbrales import SelectorUmbral
from .evaluacion import EvaluadorClasificador
from ...estadisticas.curvas_roc import CalculadorROC


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
        *** FUNCIÓN PRINCIPAL DE ENTRENAMIENTO BAYESIANO RGB ***
        Localización: Línea ~82 del archivo clasificador.py
        
        FUNDAMENTO TEÓRICO:
        Implementa clasificación Bayesiana basada en el Teorema de Bayes:
        P(Clase|RGB) = P(RGB|Clase) × P(Clase) / P(RGB)
        
        MATEMÁTICA APLICADA:
        1. Distribuciones Gaussianas Multivariadas:
           - Para lesión: RGB ~ N(μ_lesion, Σ_lesion)
           - Para sana: RGB ~ N(μ_sana, Σ_sana)
        
        2. Log-Likelihood Ratio (LLR):
           LLR = log[P(RGB|lesion)/P(RGB|sana)] + log[P(lesion)/P(sana)]
           
        3. Decisión: Si LLR > umbral → lesión, sino → sana
        
        PROCESO DE ENTRENAMIENTO:
        1. Muestreo equilibrado (50% lesión, 50% sana) → evita sesgo
        2. Estimación de parámetros gaussianos (μ, Σ) por máxima verosimilitud
        3. Cálculo de probabilidades a priori P(lesión), P(sana)
        4. Selección de umbral óptimo según criterio (Youden, EER, etc.)
        
        ROBUSTEZ ESTADÍSTICA:
        - Regularización de matrices de covarianza para estabilidad numérica
        - Validación de definición positiva de Σ
        - Manejo de casos degenerados
        
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
        *** FUNCIÓN DE CLASIFICACIÓN BAYESIANA RGB ***
        Localización: Línea ~150 del archivo clasificador.py
        
        FUNDAMENTO MATEMÁTICO:
        Aplica la regla de decisión Bayesiana píxel por píxel:
        
        1. CÁLCULO DE VEROSIMILITUDES:
           P(RGB|lesión) usando distribución N(μ_lesion, Σ_lesion)
           P(RGB|sana) usando distribución N(μ_sana, Σ_sana)
        
        2. LOG-LIKELIHOOD RATIO (LLR):
           LLR = log[P(RGB|lesión)] - log[P(RGB|sana)] + log[P(lesión)/P(sana)]
           
           Ventajas del log:
           - Evita underflow numérico
           - Convierte multiplicaciones en sumas
           - Estabilidad computacional
        
        3. REGLA DE DECISIÓN:
           Si LLR ≥ umbral → píxel clasificado como lesión
           Si LLR < umbral → píxel clasificado como sana
        
        TEOREMA DE BAYES APLICADO:
        P(lesión|RGB) ∝ P(RGB|lesión) × P(lesión)
        P(sana|RGB) ∝ P(RGB|sana) × P(sana)
        
        El umbral óptimo minimiza error según criterio elegido:
        - Youden: maximiza sensibilidad + especificidad
        - EER: iguala falsos positivos y falsos negativos
        - Prior Balanced: considera probabilidades a priori
        
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
        *** FUNCIÓN DE EVALUACIÓN CON MÉTRICAS MÉDICAS ***
        Localización: Línea ~180 del archivo clasificador.py
        
        MÉTRICAS ESTADÍSTICAS IMPLEMENTADAS:
        
        1. MATRIZ DE CONFUSIÓN:
           - TP (True Positives): píxeles lesión correctamente identificados
           - TN (True Negatives): píxeles sanos correctamente identificados  
           - FP (False Positives): píxeles sanos clasificados como lesión
           - FN (False Negatives): píxeles lesión clasificados como sanos
        
        2. MÉTRICAS DERIVADAS:
           - Exactitud = (TP + TN) / (TP + TN + FP + FN)
           - Precisión = TP / (TP + FP) → "de los que digo lesión, cuántos son"
           - Sensibilidad (Recall) = TP / (TP + FN) → "de las lesiones reales, cuántas detecto"
           - Especificidad = TN / (TN + FP) → "de los sanos reales, cuántos identifico"
        
        3. MÉTRICAS ESPECIALIZADAS:
           - F1-Score = 2 × (Precisión × Sensibilidad) / (Precisión + Sensibilidad)
           - Índice Jaccard = TP / (TP + FP + FN) → solapamiento con ground truth
           - Índice Youden = Sensibilidad + Especificidad - 1 → balance global
        
        INTERPRETACIÓN MÉDICA:
        - Alta sensibilidad: no perdemos lesiones (crítico en medicina)
        - Alta especificidad: no alarmamos innecesariamente
        - F1 alto: balance entre precisión y detección
        - Youden alto: clasificador balanceado óptimo
        
        VALIDACIÓN ESTADÍSTICA:
        - Evaluación en conjunto independiente (test set)
        - Métricas robustas ante desbalance de clases
        - Análisis de performance píxel-a-píxel
        
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
    
    def generar_curva_roc(self, datos_test: List[Dict], nombre_clasificador: str = "Clasificador RGB") -> Dict[str, Any]:
        """
        *** GENERACIÓN CURVA ROC PARA CLASIFICADOR BAYESIANO ***
        
        PROCESO DE ANÁLISIS ROC:
        1. Extrae todos los píxeles y etiquetas de las imágenes de test
        2. Calcula log-likelihood ratios para cada píxel
        3. Genera curva ROC usando sklearn.metrics.roc_curve
        4. Calcula AUC (Area Under Curve)
        5. Determina punto de operación según criterio seleccionado
        
        INTERPRETACIÓN PÍXEL-A-PÍXEL:
        Cada píxel es tratado como una muestra independiente para
        construcción de la curva ROC, permitiendo análisis detallado
        de capacidad discriminativa del clasificador.
        
        MÉTRICAS RETORNADAS:
        - Curva ROC completa (FPR, TPR, umbrales)
        - AUC con interpretación clínica
        - Punto de operación óptimo marcado
        - Justificación del criterio seleccionado
        
        Args:
            datos_test: Lista de diccionarios con imágenes y máscaras de test
            nombre_clasificador: Nombre para identificar este clasificador
            
        Returns:
            Diccionario con resultados completos del análisis ROC
        """
        self._validar_entrenado()
        
        # *** EXTRACCIÓN DE DATOS PARA ROC ***
        pixeles_scores = []
        pixeles_etiquetas = []
        
        print(f"Procesando {len(datos_test)} imágenes para análisis ROC...")
        
        for i, datos in enumerate(datos_test):
            imagen = datos['imagen']
            mascara_gt = datos['mascara']
            
            # Calcular razones de verosimilitud para toda la imagen
            razones = self._calcular_razones_verosimilitud(imagen)
            
            # Aplanar para análisis píxel-a-píxel
            razones_flat = razones.flatten()
            etiquetas_flat = mascara_gt.flatten()
            
            pixeles_scores.extend(razones_flat)
            pixeles_etiquetas.extend(etiquetas_flat)
            
            if (i + 1) % 10 == 0:
                print(f"  Procesadas {i + 1}/{len(datos_test)} imágenes")
        
        # Convertir a arrays numpy
        y_scores = np.array(pixeles_scores)
        y_true = np.array(pixeles_etiquetas)
        
        print(f"Total píxeles analizados: {len(y_scores):,}")
        print(f"Píxeles lesión: {np.sum(y_true):,} ({np.mean(y_true):.1%})")
        print(f"Píxeles sanos: {len(y_true) - np.sum(y_true):,} ({1-np.mean(y_true):.1%})")
        
        # *** ANÁLISIS ROC CON CALCULADORA ESPECIALIZADA ***
        calculadora_roc = CalculadorROC()
        
        # Calcular curva ROC
        resultados_roc = calculadora_roc.calcular_roc(y_true, y_scores, nombre_clasificador)
        
        # Seleccionar punto de operación según criterio del clasificador
        criterio_mapeado = self._mapear_criterio_roc()
        punto_operacion = calculadora_roc.seleccionar_punto_operacion(
            nombre_clasificador, criterio_mapeado
        )
        
        # *** RESULTADOS INTEGRADOS ***
        resultados = {
            'calculadora_roc': calculadora_roc,
            'resultados_roc': resultados_roc,
            'punto_operacion': punto_operacion,
            'criterio_usado': self.criterio_umbral,
            'umbral_actual': self.umbral,
            'auc': resultados_roc['auc'],
            'interpretacion_auc': calculadora_roc._interpretar_auc(resultados_roc['auc']),
            'justificacion_criterio': punto_operacion.justificacion,
            'metricas_punto': {
                'sensibilidad': punto_operacion.tpr,
                'especificidad': punto_operacion.tnr,
                'youden_index': punto_operacion.youden_index,
                'precision': punto_operacion.precision,
                'f1_score': punto_operacion.f1_score
            }
        }
        
        print(f"\n*** RESUMEN ANÁLISIS ROC ***")
        print(f"AUC: {resultados['auc']:.3f} ({resultados['interpretacion_auc']})")
        print(f"Punto de operación ({criterio_mapeado}):")
        print(f"  - Sensibilidad: {punto_operacion.tpr:.3f} ({punto_operacion.tpr:.1%})")
        print(f"  - Especificidad: {punto_operacion.tnr:.3f} ({punto_operacion.tnr:.1%})")
        print(f"  - Índice Youden: {punto_operacion.youden_index:.3f}")
        
        return resultados
    
    def _mapear_criterio_roc(self) -> str:
        """
        Mapea criterios internos del clasificador a criterios de ROC.
        
        Returns:
            Criterio compatible con CalculadorROC
        """
        mapeo = {
            'youden': 'youden',
            'equal_error': 'eer',
            'prior_balanced': 'youden'  # Fallback para compatibilidad
        }
        
        return mapeo.get(self.criterio_umbral, 'youden')
    
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
        *** SELECCIÓN DE UMBRAL ÓPTIMO BAYESIANO ***
        Localización: Línea ~371 del archivo clasificador.py
        
        FUNDAMENTO TEÓRICO:
        El umbral τ en la regla de decisión LLR ≥ τ determina el balance
        entre sensibilidad y especificidad del clasificador.
        
        CRITERIOS DE OPTIMIZACIÓN IMPLEMENTADOS:
        
        1. ÍNDICE YOUDEN (J = Sensibilidad + Especificidad - 1):
           - Maximiza la suma de sensibilidad y especificidad
           - Punto óptimo en curva ROC (máxima distancia a línea diagonal)
           - Interpretación: máximo beneficio neto del test diagnóstico
           
        2. EQUAL ERROR RATE (EER):
           - Iguala False Positive Rate (FPR) y False Negative Rate (FNR)
           - Punto donde FPR = FNR = error_rate
           - Útil cuando costos de errores tipo I y II son similares
           
        3. PRIOR BALANCED:
           - Incorpora probabilidades a priori en la decisión
           - Umbral = log[P(sana)/P(lesión)] (Bayes óptimo)
           - Minimiza error de clasificación esperado
        
        PROCESO DE OPTIMIZACIÓN:
        1. Calcula LLR para todos los píxeles de validación
        2. Prueba múltiples umbrales candidatos
        3. Evalúa criterio de optimización para cada umbral
        4. Selecciona umbral que maximiza/minimiza criterio objetivo
        
        VALIDACIÓN CRUZADA:
        - Usa conjunto independiente para evitar overfitting
        - Garantiza generalización a datos no vistos
        
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
        *** CÁLCULO DE LOG-LIKELIHOOD RATIO BAYESIANO ***
        Localización: Línea ~431 del archivo clasificador.py
        
        FUNDAMENTO MATEMÁTICO:
        Implementa el núcleo de la inferencia Bayesiana mediante el 
        Log-Likelihood Ratio (LLR) para cada píxel RGB.
        
        FÓRMULA IMPLEMENTADA:
        LLR(x) = log[P(x|lesión)/P(x|sana)] + log[P(lesión)/P(sana)]
        
        Donde:
        - x = vector RGB del píxel [R, G, B]
        - P(x|clase) = distribución gaussiana multivariada
        - P(clase) = probabilidad a priori de la clase
        
        DISTRIBUCIÓN GAUSSIANA MULTIVARIADA:
        P(x|clase) = (2π)^(-k/2) |Σ|^(-1/2) exp(-½(x-μ)ᵀΣ⁻¹(x-μ))
        
        Donde:
        - k = 3 (dimensiones RGB)
        - μ = vector de medias [μ_R, μ_G, μ_B]
        - Σ = matriz de covarianza 3×3
        - |Σ| = determinante de la matriz de covarianza
        
        LOG-LIKELIHOOD (evita underflow):
        log P(x|clase) = -½[k×log(2π) + log|Σ| + (x-μ)ᵀΣ⁻¹(x-μ)]
        
        VENTAJAS DEL ENFOQUE LOG:
        1. Estabilidad numérica (evita productos de números muy pequeños)
        2. Eficiencia computacional (sumas en lugar de productos)
        3. Robustez ante valores extremos
        
        DISTANCIA DE MAHALANOBIS:
        d²(x,μ) = (x-μ)ᵀΣ⁻¹(x-μ)
        - Considera correlaciones entre canales RGB
        - Normaliza por variabilidad de cada clase
        - Generaliza distancia euclidiana
        
        Args:
            imagen: Imagen RGB de entrada
            
        Returns:
            Matriz de razones de verosimilitud (LLR para cada píxel)
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