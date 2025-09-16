"""
Funciones para evaluación y métricas de clasificadores.
"""
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def calcular_metricas(y_true, y_pred):
    """
    Calcula métricas de evaluación para clasificación binaria.
    
    Args:
        y_true (np.ndarray): Etiquetas reales (0=sana, 1=lesión)
        y_pred (np.ndarray): Predicciones (0=sana, 1=lesión)
        
    Returns:
        dict: Diccionario con métricas calculadas
    """
    # Matriz de confusión
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    
    # Métricas básicas
    exactitud = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    sensibilidad = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall
    especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # Índice de Jaccard (IoU)
    jaccard = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    
    # F1-Score
    f1 = 2 * (precision * sensibilidad) / (precision + sensibilidad) if (precision + sensibilidad) > 0 else 0
    
    # Índice de Youden
    youden = sensibilidad + especificidad - 1
    
    return {
        'exactitud': exactitud,
        'precision': precision,
        'sensibilidad': sensibilidad,
        'especificidad': especificidad,
        'jaccard': jaccard,
        'f1_score': f1,
        'youden': youden,
        'matriz_confusion': {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}
    }

def evaluar_clasificador_en_conjunto(clasificador, conjunto_datos):
    """
    Evalúa un clasificador en un conjunto de datos completo.
    
    Args:
        clasificador: Clasificador entrenado
        conjunto_datos (list): Lista de diccionarios con 'imagen' y 'mascara'
        
    Returns:
        dict: Métricas de evaluación agregadas
    """
    predicciones_totales = []
    etiquetas_totales = []
    
    for item in conjunto_datos:
        imagen = item['imagen']
        mascara_real = item['mascara']
        
        # Clasificar imagen
        mascara_pred = clasificador.clasificar(imagen)
        
        # Acumular predicciones y etiquetas
        predicciones_totales.extend(mascara_pred.flatten())
        etiquetas_totales.extend(mascara_real.flatten())
    
    # Convertir a arrays
    y_true = np.array(etiquetas_totales)
    y_pred = np.array(predicciones_totales)
    
    # Calcular métricas
    metricas = calcular_metricas(y_true, y_pred)
    
    return metricas

def generar_curva_roc(clasificador, conjunto_datos):
    """
    Genera curva ROC para un clasificador.
    
    Args:
        clasificador: Clasificador entrenado que puede retornar razones de verosimilitud
        conjunto_datos (list): Lista de diccionarios con 'imagen' y 'mascara'
        
    Returns:
        tuple: (fpr, tpr, auc_score, thresholds)
    """
    razones_totales = []
    etiquetas_totales = []
    
    for item in conjunto_datos:
        imagen = item['imagen']
        mascara_real = item['mascara']
        
        # Obtener razones de verosimilitud
        _, razones = clasificador._clasificar_imagen_con_razones(imagen)
        
        # Acumular
        razones_totales.extend(razones.flatten())
        etiquetas_totales.extend(mascara_real.flatten())
    
    # Convertir a arrays
    y_true = np.array(etiquetas_totales)
    y_scores = np.array(razones_totales)
    
    # Calcular curva ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc_score = auc(fpr, tpr)
    
    return fpr, tpr, auc_score, thresholds

def comparar_criterios_umbral(imagenes_entrenamiento, imagenes_validacion):
    """
    Compara diferentes criterios de selección de umbral.
    
    Returns:
        dict: Resultados de evaluación para cada criterio
    """
    from .clasificador_bayesiano import ClasificadorBayesianoRGB
    
    criterios = ['youden', 'equal_error', 'prior_balanced']
    resultados = {}
    
    for criterio in criterios:
        # Entrenar clasificador con criterio específico
        clasificador = ClasificadorBayesianoRGB(criterio_umbral=criterio)
        clasificador.entrenar(imagenes_entrenamiento)
        
        # Evaluar en conjunto de validación
        metricas = evaluar_clasificador_en_conjunto(clasificador, imagenes_validacion)
        
        # Guardar resultados
        resultados[criterio] = {
            'metricas': metricas,
            'umbral': clasificador.umbral,
            'justificacion': clasificador.justificar_criterio_umbral()
        }
    
    return resultados

def imprimir_reporte_evaluacion(metricas, titulo="Evaluación del Clasificador"):
    """
    Imprime un reporte formateado de las métricas de evaluación.
    """
    print(f"\n{'='*50}")
    print(f"{titulo:^50}")
    print(f"{'='*50}")
    print(f"Exactitud:      {metricas['exactitud']:.4f}")
    print(f"Precisión:      {metricas['precision']:.4f}")
    print(f"Sensibilidad:   {metricas['sensibilidad']:.4f}")
    print(f"Especificidad:  {metricas['especificidad']:.4f}")
    print(f"F1-Score:       {metricas['f1_score']:.4f}")
    print(f"Índice Jaccard: {metricas['jaccard']:.4f}")
    print(f"Índice Youden:  {metricas['youden']:.4f}")
    print(f"\nMatriz de Confusión:")
    mc = metricas['matriz_confusion']
    print(f"  TP: {mc['TP']:6d}  |  FN: {mc['FN']:6d}")
    print(f"  FP: {mc['FP']:6d}  |  TN: {mc['TN']:6d}")
    print(f"{'='*50}")