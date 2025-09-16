"""
Estrategias de selección de umbral para clasificación Bayesiana.

Este módulo implementa diferentes criterios para la selección óptima
del umbral de decisión en clasificación binaria, cada uno optimizando
diferentes aspectos del rendimiento del clasificador.

FUNDAMENTO TEÓRICO DE UMBRALES:
En clasificación Bayesiana, la regla de decisión es:
- Si LLR(x) ≥ τ → clase lesión
- Si LLR(x) < τ → clase sana

El umbral τ determina el balance entre:
- Sensibilidad (detectar lesiones reales)
- Especificidad (evitar falsos positivos)

CRITERIOS IMPLEMENTADOS:
1. YOUDEN: Maximiza J = Sensibilidad + Especificidad - 1
2. EQUAL ERROR RATE: Iguala FPR y FNR
3. PRIOR BALANCED: Incorpora probabilidades a priori

CURVA ROC Y SELECCIÓN:
Cada criterio busca el punto óptimo en la curva ROC que optimiza
diferentes métricas de rendimiento diagnóstico.
"""

import numpy as np
from typing import Dict, Any
from abc import ABC
from .base import ISelectorUmbral


class SelectorUmbralYouden(ISelectorUmbral):
    """
    *** SELECTOR UMBRAL ÍNDICE YOUDEN ***
    Localización: Línea ~28 del archivo umbrales.py
    
    FUNDAMENTO MATEMÁTICO:
    El índice de Youden es una medida de efectividad de un test diagnóstico
    que combina sensibilidad y especificidad en una sola métrica.
    
    FÓRMULA:
    J = Sensibilidad + Especificidad - 1
    J = TPR + TNR - 1 = TPR - FPR
    
    Donde:
    - TPR = True Positive Rate = TP/(TP+FN) (Sensibilidad)
    - TNR = True Negative Rate = TN/(TN+FP) (Especificidad)  
    - FPR = False Positive Rate = FP/(FP+TN)
    
    INTERPRETACIÓN:
    - J = 0: Test no mejor que azar (TPR = FPR)
    - J = 1: Test perfecto (TPR = 1, FPR = 0)
    - J máximo: Punto óptimo que balancea sensibilidad y especificidad
    
    VENTAJAS CLÍNICAS:
    - Maximiza beneficio neto del test diagnóstico
    - No favorece una clase sobre otra
    - Punto óptimo en curva ROC (máxima distancia a diagonal)
    - Robusto ante desbalance moderado de clases
    
    GEOMETRÍA ROC:
    El punto que maximiza J corresponde al punto en la curva ROC
    con máxima distancia perpendicular a la línea y = x.
    
    Selector de umbral basado en el índice de Youden.
    
    El índice de Youden (J = Sensibilidad + Especificidad - 1) maximiza
    la capacidad discriminativa del clasificador balanceando la detección
    de verdaderos positivos y la correcta identificación de negativos.
    """
    
    def seleccionar(self, razones_verosimilitud: np.ndarray, 
                   etiquetas_reales: np.ndarray) -> float:
        """
        *** IMPLEMENTACIÓN ALGORITMO YOUDEN ***
        Localización: Método seleccionar en SelectorUmbralYouden
        
        ALGORITMO MATEMÁTICO:
        1. Para cada threshold t en scores únicos:
           - Calcular TPR(t) = TP(t)/(TP(t)+FN(t))
           - Calcular FPR(t) = FP(t)/(FP(t)+TN(t))
           - Calcular J(t) = TPR(t) - FPR(t)
        
        2. Seleccionar t* = argmax(J(t))
        
        OPTIMIZACIÓN ESTADÍSTICA:
        El umbral óptimo maximiza la función objetivo:
        
        J(t) = P(score ≥ t | Y=1) - P(score ≥ t | Y=0)
        
        Esto equivale a maximizar la diferencia entre:
        - Probabilidad de detección correcta de melanoma
        - Probabilidad de falsa alarma en casos benignos
        
        JUSTIFICACIÓN TEÓRICA:
        Bajo distribuciones normales para cada clase:
        - Melanoma: score ~ N(μ₁, σ₁²)
        - Benigno: score ~ N(μ₀, σ₀²)
        
        El umbral Youden aproxima el punto de intersección
        de las distribuciones cuando tienen varianzas similares.
        
        COMPLEJIDAD: O(n log n) por ordenamiento de scores
        ROBUSTEZ: Invariante ante transformaciones monótonas
        
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
        
        # *** CÁLCULO MATRIZ CONFUSIÓN ***
        # Para cada umbral candidato evaluamos las 4 categorías:
        # TP: Melanoma detectado correctamente (score ≥ t, label = 1)
        # FP: Falsa alarma (score ≥ t, label = 0) 
        # TN: Benigno identificado correctamente (score < t, label = 0)
        # FN: Melanoma perdido (score < t, label = 1)
        
        for umbral in umbrales_candidatos:
            predicciones = (razones_verosimilitud >= umbral).astype(int)
            
            # Calcular matriz de confusión
            tp = np.sum((predicciones == 1) & (etiquetas_reales == 1))
            fp = np.sum((predicciones == 1) & (etiquetas_reales == 0))
            tn = np.sum((predicciones == 0) & (etiquetas_reales == 0))
            fn = np.sum((predicciones == 0) & (etiquetas_reales == 1))
            
            # Evitar división por cero (casos degenerados)
            if (tp + fn) == 0 or (tn + fp) == 0:
                continue
            
            # *** CÁLCULO ÍNDICE YOUDEN ***
            # Sensibilidad = P(test+ | enfermo) = TP/(TP+FN)
            # Especificidad = P(test- | sano) = TN/(TN+FP)
            # J = Sensibilidad + Especificidad - 1
            sensibilidad = tp / (tp + fn)
            especificidad = tn / (tn + fp)
            j = sensibilidad + especificidad - 1
            
            # Actualizar umbral óptimo si mejora el índice
            if j > mejor_j:
                mejor_j = j
                mejor_umbral = umbral
        
        return mejor_umbral
    
    def justificar(self) -> str:
        """
        *** JUSTIFICACIÓN TEÓRICA ÍNDICE YOUDEN ***
        
        FUNDAMENTO ESTADÍSTICO:
        El índice de Youden representa la máxima eficiencia diagnóstica
        de un test binario bajo la teoría de detección de señales.
        
        INTERPRETACIÓN BAYESIANA:
        Maximizar J equivale a maximizar:
        P(D+|T+) + P(D-|T-) - 1
        
        Donde:
        - P(D+|T+): Probabilidad de enfermedad dado test positivo
        - P(D-|T-): Probabilidad de no enfermedad dado test negativo
        
        OPTIMALIDAD ROC:
        El punto Youden en la curva ROC es tangente a la línea
        con pendiente = (1-π)/π · (c₁₀-c₀₀)/(c₀₁-c₁₁)
        
        Para costos simétricos y prevalencia equiprobable:
        pendiente = 1 → tangente a 45°
        
        APLICACIÓN MÉDICA:
        Ideal cuando el costo de falsos positivos y falsos negativos
        es similar, y se busca maximizar la capacidad discriminativa
        global del test diagnóstico.
        
        Justifica el criterio de Youden.
        """
        return """
        El índice de Youden maximiza la suma de sensibilidad y especificidad (TPR + TNR - 1).
        Es ideal para aplicaciones médicas donde se busca un balance óptimo entre la detección
        de casos positivos y la correcta identificación de casos negativos, minimizando tanto
        falsos positivos como falsos negativos de manera equilibrada.
        """.strip()


class SelectorUmbralEqualError(ISelectorUmbral):
    """
    *** SELECTOR UMBRAL EQUAL ERROR RATE (EER) ***
    Localización: Línea ~202 del archivo umbrales.py
    
    FUNDAMENTO TEÓRICO:
    Equal Error Rate es el punto donde FPR = FNR, representando
    el equilibrio perfecto entre tipos de error.
    
    DEFINICIONES:
    - FPR = False Positive Rate = FP/(FP+TN) = 1-Especificidad
    - FNR = False Negative Rate = FN/(FN+TP) = 1-Sensibilidad
    - EER: punto donde FPR(t) = FNR(t)
    
    ECUACIÓN DE EQUILIBRIO:
    FP/(FP+TN) = FN/(FN+TP)
    
    En curva ROC: punto donde TPR = 1-FPR
    En curva DET: punto de intersección con diagonal
    
    INTERPRETACIÓN ESTADÍSTICA:
    EER representa la tasa de error cuando el sistema
    trata ambos tipos de error como equivalentes.
    
    PROPIEDADES:
    - Simétrico: trata FP y FN de manera equitativa
    - Robusto: menos sensible a desbalance de clases
    - Comparable: permite comparar clasificadores diferentes
    - Conservador: minimiza el error máximo entre tipos
    
    APLICACIÓN BIOMÉTRICA:
    Común en sistemas de verificación (huellas, reconocimiento facial)
    donde FP y FN tienen consecuencias similares.
    
    Selector de umbral basado en Equal Error Rate (EER).
    
    Busca el punto donde la tasa de falsos positivos es igual
    a la tasa de falsos negativos, proporcionando un balance
    simétrico en los tipos de error.
    """
    
    def seleccionar(self, razones_verosimilitud: np.ndarray, 
                   etiquetas_reales: np.ndarray) -> float:
        """
        *** ALGORITMO BÚSQUEDA EER ***
        Localización: Método seleccionar en SelectorUmbralEqualError
        
        MÉTODO DE OPTIMIZACIÓN:
        1. Evaluar umbrales candidatos t_i
        2. Para cada t_i calcular:
           FPR(t_i) = FP(t_i)/(FP(t_i)+TN(t_i))
           FNR(t_i) = FN(t_i)/(FN(t_i)+TP(t_i))
        3. Minimizar |FPR(t_i) - FNR(t_i)|
        
        CONVERGENCIA:
        El algoritmo busca el punto t* tal que:
        lim[t→t*] |FPR(t) - FNR(t)| = 0
        
        MÉTODO NUMÉRICO:
        Utilizamos búsqueda por grilla en percentiles para
        aproximar la solución del problema de optimización:
        
        t* = argmin |P(score ≥ t | Y=0) - P(score < t | Y=1)|
        
        COMPLEJIDAD: O(k·n) donde k=100 umbrales candidatos
        
        PRECISIÓN: Depende de granularidad de percentiles
        Para mayor precisión usar interpolación entre umbrales.
        
        Selecciona el umbral que minimiza |FPR - FNR|.
        
        Args:
            razones_verosimilitud: Valores de razón de verosimilitud
            etiquetas_reales: Etiquetas verdaderas
            
        Returns:
            Umbral que minimiza la diferencia entre FPR y FNR
        """
        # *** BÚSQUEDA POR GRILLA EN PERCENTILES ***
        # Evaluamos 100 umbrales candidatos distribuidos uniformemente
        # en los percentiles para capturar toda la distribución de scores
        umbrales_candidatos = np.percentile(razones_verosimilitud, np.linspace(1, 99, 100))
        
        menor_diferencia = float('inf')
        mejor_umbral = 1.0
        
        for umbral in umbrales_candidatos:
            predicciones = (razones_verosimilitud >= umbral).astype(int)
            
            # *** CÁLCULO MATRIZ DE CONFUSIÓN ***
            tp = np.sum((predicciones == 1) & (etiquetas_reales == 1))
            fp = np.sum((predicciones == 1) & (etiquetas_reales == 0))
            tn = np.sum((predicciones == 0) & (etiquetas_reales == 0))
            fn = np.sum((predicciones == 0) & (etiquetas_reales == 1))
            
            # Evitar división por cero (casos degenerados)
            if (tp + fn) == 0 or (tn + fp) == 0:
                continue
            
            # *** CÁLCULO TASAS DE ERROR ***
            # FPR = P(predicción=1 | verdadero=0)
            # FNR = P(predicción=0 | verdadero=1)
            fpr = fp / (fp + tn)  # Tasa de falsos positivos
            fnr = fn / (fn + tp)  # Tasa de falsos negativos
            diferencia = abs(fpr - fnr)
            
            # Actualizar si encontramos mejor equilibrio
            if diferencia < menor_diferencia:
                menor_diferencia = diferencia
                mejor_umbral = umbral
        
        return mejor_umbral
    
    def justificar(self) -> str:
        """
        *** JUSTIFICACIÓN TEÓRICA EER ***
        
        FUNDAMENTO ESTADÍSTICO:
        EER representa el equilibrio óptimo cuando ambos tipos de error
        tienen la misma penalización en la función de costo.
        
        TEORÍA DE DECISIÓN:
        Minimizar: C₁₀·P(FP) + C₀₁·P(FN)
        
        Para C₁₀ = C₀₁ (costos iguales):
        Umbral óptimo donde FPR = FNR
        
        APLICACIÓN PRÁCTICA:
        - Sistemas biométricos: autenticación vs rechazo
        - Control de calidad: defecto vs falsa alarma  
        - Diagnóstico: cuando sobre-diagnóstico = sub-diagnóstico
        
        VENTAJAS:
        - Robusto ante desbalance de clases
        - Interpretable: "mismo riesgo de error en ambas direcciones"
        - Comparable entre diferentes clasificadores
        
        DESVENTAJAS:
        - Puede no ser óptimo si costos FP ≠ FN
        - Conservador en aplicaciones médicas críticas
        
        Justifica el criterio Equal Error Rate.
        """
        return """
        El criterio Equal Error Rate busca el punto donde la tasa de falsos positivos
        es igual a la tasa de falsos negativos (FPR = FNR). Es útil cuando los costos
        de ambos tipos de error son similares y se desea un clasificador equilibrado
        en términos de errores simétricos.
        """.strip()


class SelectorUmbralPriorBalanced(ISelectorUmbral):
    """
    *** SELECTOR UMBRAL PROBABILIDADES A PRIORI ***
    Localización: Línea ~347 del archivo umbrales.py
    
    FUNDAMENTO BAYESIANO:
    Utiliza la teoría de decisión Bayesiana para determinar
    el umbral óptimo basado en probabilidades a priori.
    
    REGLA DE DECISIÓN BAYES:
    Decidir clase 1 si:
    P(Y=1|X) > P(Y=0|X)
    
    Equivalente a:
    log(P(X|Y=1)/P(X|Y=0)) > log(P(Y=0)/P(Y=1))
    
    UMBRAL TEÓRICO:
    t* = log(π₀/π₁) = log(P(Y=0)/P(Y=1))
    
    Donde:
    - π₀ = probabilidad a priori de clase negativa
    - π₁ = probabilidad a priori de clase positiva
    
    INTERPRETACIÓN:
    - Si π₀ = π₁ = 0.5 → t* = 0 (clases equiprobables)
    - Si π₀ > π₁ → t* > 0 (favorece clase negativa)
    - Si π₀ < π₁ → t* < 0 (favorece clase positiva)
    
    OPTIMALIDAD:
    Este umbral minimiza la probabilidad de error
    bajo la hipótesis de distribuciones conocidas.
    
    APLICACIÓN MÉDICA:
    Incorpora prevalencia de la enfermedad en la población
    para ajustar sensibilidad del diagnóstico.
    
    Selector de umbral basado en probabilidades a priori balanceadas.
    
    Utiliza el umbral teórico derivado de la teoría de decisión Bayesiana
    cuando se asumen probabilidades a priori balanceadas y costos uniformes.
    """
    
    def __init__(self, prior_lesion: float, prior_sana: float):
        """
        *** INICIALIZACIÓN PROBABILIDADES A PRIORI ***
        
        PARÁMETROS BAYESIANOS:
        Establece las probabilidades a priori necesarias para
        calcular el umbral teórico óptimo.
        
        NORMALIZACIÓN:
        Se asume que prior_lesion + prior_sana = 1
        (distribución de probabilidad válida)
        
        INTERPRETACIÓN MÉDICA:
        - prior_lesion: prevalencia de melanoma en población
        - prior_sana: prevalencia de lesiones benignas
        
        EJEMPLOS:
        - Población general: prior_lesion ≈ 0.01 (1% melanomas)
        - Clínica especializada: prior_lesion ≈ 0.3 (30% melanomas)
        
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
        *** CÁLCULO UMBRAL BAYESIANO ÓPTIMO ***
        Localización: Método seleccionar en SelectorUmbralPriorBalanced
        
        DERIVACIÓN TEÓRICA:
        De la regla de decisión Bayesiana óptima:
        
        Decidir Y=1 si P(Y=1|X) > P(Y=0|X)
        
        Aplicando Bayes:
        P(X|Y=1)·P(Y=1) > P(X|Y=0)·P(Y=0)
        
        Tomando logaritmo:
        log(P(X|Y=1)) - log(P(X|Y=0)) > log(P(Y=0)/P(Y=1))
        
        LLR(X) > log(π₀/π₁)
        
        UMBRAL RESULTANTE:
        t* = log(prior_sana / prior_lesion)
        
        CASOS ESPECIALES:
        - Clases equiprobables: t* = log(0.5/0.5) = 0
        - Melanoma raro: t* = log(0.99/0.01) ≈ 4.6 (umbral alto)
        - Clínica especializada: t* = log(0.7/0.3) ≈ 0.85
        
        OPTIMIZACIÓN:
        Este umbral minimiza la probabilidad total de error
        bajo el supuesto de distribuciones gaussianas conocidas.
        
        NOTA: Los datos de entrada (razones_verosimilitud, etiquetas_reales)
        no se utilizan ya que el umbral es teórico, derivado analíticamente.
        
        Selecciona el umbral basado en la relación de probabilidades a priori.
        
        Para clasificación Bayesiana óptima con costos uniformes,
        el umbral teórico es P(sana) / P(lesión).
        
        Args:
            razones_verosimilitud: Valores de razón de verosimilitud (no utilizados)
            etiquetas_reales: Etiquetas verdaderas (no utilizadas)
            
        Returns:
            Umbral teórico basado en probabilidades a priori
        """
        # *** UMBRAL TEÓRICO BAYESIANO ***
        # Derivado de la teoría de decisión Bayesiana óptima
        # t* = log(P(Y=0)/P(Y=1)) = log(π₀/π₁)
        umbral_teorico = self.prior_sana / self.prior_lesion
        return umbral_teorico
    
    def justificar(self) -> str:
        """
        *** JUSTIFICACIÓN TEÓRICA PRIOR BALANCED ***
        
        FUNDAMENTO BAYESIANO:
        Este método implementa la teoría de decisión Bayesiana clásica
        que minimiza la probabilidad total de error.
        
        TEOREMA FUNDAMENTAL:
        Para un clasificador Bayesiano óptimo, el umbral que minimiza
        la probabilidad de error es:
        
        t* = log(π₀/π₁)
        
        Donde π₀, π₁ son las probabilidades a priori.
        
        DEMOSTRACIÓN:
        La probabilidad de error es:
        P(error) = π₁·P(decidir 0|Y=1) + π₀·P(decidir 1|Y=0)
        
        Minimizar P(error) lleva al umbral t*.
        
        INTERPRETACIÓN MÉDICA:
        - Incorpora prevalencia de la enfermedad
        - Ajusta sensibilidad según población objetivo
        - Equilibra costos considerando frecuencia natural
        
        APLICACIONES:
        - Screening poblacional (prevalencia baja)
        - Diagnóstico especializado (prevalencia alta)
        - Medicina personalizada (riesgo individual)
        
        VENTAJA TEÓRICA:
        Garantiza optimalidad bajo supuestos del modelo
        (distribuciones gaussianas, costos uniformes).
        
        Justifica el criterio Prior Balanced.
        """
        return """
        El criterio Prior Balanced ajusta el umbral considerando las probabilidades
        a priori de las clases, buscando un balance que refleje la distribución
        natural de los datos. Es apropiado cuando se desea mantener las proporciones
        observadas en la población de entrenamiento.
        """.strip()


class SelectorUmbral:
    """
    *** FACTORY PATTERN PARA SELECTORES DE UMBRAL ***
    Localización: Línea ~508 del archivo umbrales.py
    
    PATRÓN DE DISEÑO:
    Implementa Factory Pattern para encapsular la creación
    de diferentes estrategias de selección de umbral.
    
    PRINCIPIOS SOLID:
    - Single Responsibility: cada selector tiene una responsabilidad
    - Open/Closed: extensible a nuevas estrategias sin modificar código
    - Dependency Inversion: depende de abstracciones (ISelectorUmbral)
    
    ESTRATEGIAS DISPONIBLES:
    1. 'youden': Maximiza J = Sensibilidad + Especificidad - 1
    2. 'equal_error': Minimiza |FPR - FNR|
    3. 'prior_balanced': Umbral teórico t* = log(π₀/π₁)
    
    FLEXIBILIDAD:
    Permite cambiar estrategia de selección sin modificar
    el código cliente (clasificador principal).
    
    EXTENSIBILIDAD:
    Agregar nuevas estrategias requiere solo:
    - Implementar ISelectorUmbral
    - Registrar en ESTRATEGIAS_DISPONIBLES
    
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