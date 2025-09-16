"""
Módulo de evaluación de rendimiento para clasificadores.

Este módulo proporciona herramientas especializadas para evaluar
el rendimiento de clasificadores binarios, calculando métricas
estándar y proporcionando análisis detallado de resultados.
"""

import numpy as np
from typing import Dict, Any, List, Tuple
from .base import IEvaluador


class EvaluadorClasificador(IEvaluador):
    """
    Evaluador especializado para clasificadores binarios.
    
    Calcula métricas estándar de clasificación binaria y proporciona
    análisis detallado del rendimiento del clasificador.
    """
    
    def evaluar(self, predicciones: np.ndarray, 
               etiquetas_reales: np.ndarray) -> Dict[str, Any]:
        """
        Evalúa el rendimiento del clasificador calculando métricas completas.
        
        Args:
            predicciones: Predicciones binarias del clasificador
            etiquetas_reales: Etiquetas verdaderas
            
        Returns:
            Diccionario con métricas de evaluación:
            - exactitud: Proporción de predicciones correctas
            - precision: Precisión en la detección de positivos
            - sensibilidad: Tasa de verdaderos positivos (recall)
            - especificidad: Tasa de verdaderos negativos
            - f1_score: Media armónica de precisión y recall
            - jaccard: Índice de Jaccard (IoU)
            - youden: Índice de Youden
            - matriz_confusion: Conteos de TP, TN, FP, FN
        """
        # Validar entradas
        self._validar_entradas(predicciones, etiquetas_reales)
        
        # Calcular matriz de confusión
        matriz_confusion = self._calcular_matriz_confusion(predicciones, etiquetas_reales)
        tp, tn, fp, fn = matriz_confusion['TP'], matriz_confusion['TN'], matriz_confusion['FP'], matriz_confusion['FN']
        
        # Calcular métricas básicas
        total_pixels = tp + tn + fp + fn
        exactitud = (tp + tn) / total_pixels if total_pixels > 0 else 0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        sensibilidad = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall, TPR
        especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0  # TNR
        
        # Métricas compuestas
        f1_score = 2 * (precision * sensibilidad) / (precision + sensibilidad) if (precision + sensibilidad) > 0 else 0
        jaccard = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0  # Intersection over Union
        youden = sensibilidad + especificidad - 1  # Índice de Youden
        
        return {
            'exactitud': exactitud,
            'precision': precision,
            'sensibilidad': sensibilidad,
            'especificidad': especificidad,
            'f1_score': f1_score,
            'jaccard': jaccard,
            'youden': youden,
            'matriz_confusion': matriz_confusion
        }
    
    def _calcular_matriz_confusion(self, predicciones: np.ndarray, 
                                 etiquetas_reales: np.ndarray) -> Dict[str, int]:
        """
        Calcula la matriz de confusión para clasificación binaria.
        
        Args:
            predicciones: Predicciones binarias (0/1)
            etiquetas_reales: Etiquetas verdaderas (0/1)
            
        Returns:
            Diccionario con conteos de la matriz de confusión
        """
        tp = np.sum((predicciones == 1) & (etiquetas_reales == 1))
        tn = np.sum((predicciones == 0) & (etiquetas_reales == 0))
        fp = np.sum((predicciones == 1) & (etiquetas_reales == 0))
        fn = np.sum((predicciones == 0) & (etiquetas_reales == 1))
        
        return {
            'TP': int(tp),
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn)
        }
    
    def _validar_entradas(self, predicciones: np.ndarray, 
                         etiquetas_reales: np.ndarray) -> None:
        """Valida las entradas para evaluación."""
        if not isinstance(predicciones, np.ndarray) or not isinstance(etiquetas_reales, np.ndarray):
            raise TypeError("Las predicciones y etiquetas deben ser arrays numpy")
        
        if predicciones.shape != etiquetas_reales.shape:
            raise ValueError(f"Formas no coinciden: predicciones {predicciones.shape}, "
                           f"etiquetas {etiquetas_reales.shape}")
        
        # Verificar que son binarios
        pred_unique = np.unique(predicciones)
        real_unique = np.unique(etiquetas_reales)
        
        if not set(pred_unique).issubset({0, 1}):
            raise ValueError(f"Las predicciones deben ser binarias (0/1), encontrados: {pred_unique}")
        
        if not set(real_unique).issubset({0, 1}):
            raise ValueError(f"Las etiquetas deben ser binarias (0/1), encontradas: {real_unique}")
    
    def evaluar_conjunto_imagenes(self, datos_imagenes: List[Dict], 
                                 clasificador) -> Dict[str, Any]:
        """
        Evalúa el clasificador en un conjunto completo de imágenes.
        
        Args:
            datos_imagenes: Lista de diccionarios con imágenes y máscaras
            clasificador: Clasificador a evaluar (debe tener método 'clasificar')
            
        Returns:
            Métricas de evaluación agregadas para todo el conjunto
        """
        if not datos_imagenes:
            raise ValueError("El conjunto de datos está vacío")
        
        # Acumular predicciones y etiquetas
        todas_predicciones = []
        todas_etiquetas = []
        
        for item in datos_imagenes:
            imagen = item['imagen']
            mascara_real = item['mascara']
            
            # Realizar predicción
            mascara_pred = clasificador.clasificar(imagen)
            
            # Acumular resultados
            todas_predicciones.extend(mascara_pred.flatten())
            todas_etiquetas.extend(mascara_real.flatten())
        
        # Convertir a arrays y evaluar
        predicciones_array = np.array(todas_predicciones)
        etiquetas_array = np.array(todas_etiquetas)
        
        return self.evaluar(predicciones_array, etiquetas_array)
    
    def comparar_clasificadores(self, datos_test: List[Dict], 
                              clasificadores: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Compara múltiples clasificadores en el mismo conjunto de datos.
        
        Args:
            datos_test: Conjunto de datos de prueba
            clasificadores: Diccionario de clasificadores {nombre: clasificador}
            
        Returns:
            Diccionario con métricas para cada clasificador
        """
        resultados = {}
        
        for nombre, clasificador in clasificadores.items():
            try:
                metricas = self.evaluar_conjunto_imagenes(datos_test, clasificador)
                resultados[nombre] = metricas
            except Exception as e:
                resultados[nombre] = {'error': str(e)}
        
        return resultados
    
    def generar_reporte_detallado(self, metricas: Dict[str, Any]) -> str:
        """
        Genera un reporte textual detallado de las métricas.
        
        Args:
            metricas: Diccionario con métricas de evaluación
            
        Returns:
            Reporte textual formateado
        """
        if 'error' in metricas:
            return f"Error en evaluación: {metricas['error']}"
        
        mc = metricas['matriz_confusion']
        
        reporte = f"""
=== REPORTE DE EVALUACIÓN ===

MÉTRICAS PRINCIPALES:
• Exactitud:      {metricas['exactitud']:.4f} ({metricas['exactitud']*100:.1f}%)
• Precisión:      {metricas['precision']:.4f} ({metricas['precision']*100:.1f}%)
• Sensibilidad:   {metricas['sensibilidad']:.4f} ({metricas['sensibilidad']*100:.1f}%)
• Especificidad:  {metricas['especificidad']:.4f} ({metricas['especificidad']*100:.1f}%)

MÉTRICAS COMPUESTAS:
• F1-Score:       {metricas['f1_score']:.4f}
• Índice Jaccard: {metricas['jaccard']:.4f} ({metricas['jaccard']*100:.1f}%)
• Índice Youden:  {metricas['youden']:.4f}

MATRIZ DE CONFUSIÓN:
           Predicción
         Lesión    Sana
Real Lesión  {mc['TP']:6d}  {mc['FN']:6d}
     Sana    {mc['FP']:6d}  {mc['TN']:6d}

INTERPRETACIÓN:
• De cada 100 lesiones, detecta {metricas['sensibilidad']*100:.0f} correctamente
• De cada 100 píxeles sanos, clasifica {metricas['especificidad']*100:.0f} correctamente
• Solapamiento con ground truth: {metricas['jaccard']*100:.1f}% (Jaccard)
        """
        
        return reporte.strip()
    
    def calcular_curva_roc_puntos(self, razones_verosimilitud: np.ndarray,
                                 etiquetas_reales: np.ndarray,
                                 num_puntos: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula puntos para la curva ROC.
        
        Args:
            razones_verosimilitud: Valores de razón de verosimilitud
            etiquetas_reales: Etiquetas verdaderas
            num_puntos: Número de puntos a calcular
            
        Returns:
            Tupla (fpr, tpr) con las tasas para la curva ROC
        """
        # Generar umbrales
        umbrales = np.linspace(razones_verosimilitud.min(), 
                              razones_verosimilitud.max(), 
                              num_puntos)
        
        fpr_list = []
        tpr_list = []
        
        for umbral in umbrales:
            predicciones = (razones_verosimilitud >= umbral).astype(int)
            
            tp = np.sum((predicciones == 1) & (etiquetas_reales == 1))
            fp = np.sum((predicciones == 1) & (etiquetas_reales == 0))
            tn = np.sum((predicciones == 0) & (etiquetas_reales == 0))
            fn = np.sum((predicciones == 0) & (etiquetas_reales == 1))
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        
        return np.array(fpr_list), np.array(tpr_list)
    
    def calcular_auc(self, fpr: np.ndarray, tpr: np.ndarray) -> float:
        """
        Calcula el área bajo la curva ROC usando regla del trapecio.
        
        Args:
            fpr: Tasas de falsos positivos
            tpr: Tasas de verdaderos positivos
            
        Returns:
            Valor del AUC
        """
        # Ordenar por FPR
        indices_ordenados = np.argsort(fpr)
        fpr_ordenado = fpr[indices_ordenados]
        tpr_ordenado = tpr[indices_ordenados]
        
        # Calcular AUC usando regla del trapecio
        auc = np.trapz(tpr_ordenado, fpr_ordenado)
        
        return auc