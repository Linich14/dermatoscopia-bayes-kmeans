"""
Clasificador Bayesiano con soporte para PCA.

Este módulo extiende el clasificador Bayesiano RGB para operar en el espacio
de componentes principales, implementando el nuevo requisito del proyecto de
"Clasificador Bayesiano + PCA".

FUNCIONES PRINCIPALES PARA LOCALIZAR:
- entrenar(): Línea ~65 - Entrena el clasificador con PCA automático
- clasificar(): Línea ~140 - Clasifica imagen aplicando transformación PCA
- evaluar(): Línea ~175 - Evalúa rendimiento con métricas completas
- comparar_con_rgb(): Línea ~310 - Compara PCA vs RGB equivalente
- obtener_justificacion_pca(): Línea ~285 - Obtiene justificación metodológica
- _entrenar_clasificador_base_pca(): Línea ~400 - Entrena Bayesiano en espacio PCA
- _calcular_log_likelihood_pca(): Línea ~480 - Calcula likelihood en espacio PCA

CÓMO FUNCIONA:
1. Aplica PCA a píxeles RGB con selección automática de componentes
2. Entrena clasificador Bayesiano en el nuevo espacio dimensional
3. Para clasificar: RGB → PCA → Bayesiano → Decisión
4. Incluye comparación con método RGB tradicional
"""

import numpy as np
import scipy.linalg
import scipy.stats
from typing import Dict, List, Any, Optional
from ..bayesiano.clasificador import ClasificadorBayesianoRGB
from ..bayesiano.base import ClasificadorBase, IClasificador
from ...reduccion_dimensionalidad import PCAAjustado


class ClasificadorBayesianoPCA(ClasificadorBase, IClasificador):
    """
    Clasificador Bayesiano que opera en el espacio de componentes principales.
    
    Esta implementación extiende el clasificador Bayesiano RGB para operar
    en un espacio de dimensionalidad reducida usando PCA, cumpliendo con el
    nuevo requisito del proyecto.
    
    Attributes:
        criterio_umbral (str): Criterio para selección del umbral de decisión
        criterio_pca (str): Criterio para selección de componentes PCA
        pca_ajustado (PCAAjustado): Transformador PCA entrenado
        clasificador_base (ClasificadorBayesianoRGB): Clasificador en espacio PCA
        estandarizar_pca (bool): Si estandarizar datos antes de PCA
    """
    
    def __init__(self, criterio_umbral: str = 'youden', 
                 criterio_pca: str = 'varianza',
                 estandarizar_pca: bool = True,
                 **kwargs_pca):
        """
        Inicializa el clasificador Bayesiano con PCA.
        
        Args:
            criterio_umbral: Criterio para selección de umbral ('youden', 'equal_error', 'prior_balanced')
            criterio_pca: Criterio para selección de componentes PCA ('varianza', 'codo', 'discriminativo')
            estandarizar_pca: Si estandarizar los datos antes de aplicar PCA
            **kwargs_pca: Argumentos adicionales para el selector PCA
        """
        super().__init__()
        
        self.criterio_umbral = criterio_umbral
        self.criterio_pca = criterio_pca
        self.estandarizar_pca = estandarizar_pca
        self.kwargs_pca = kwargs_pca
        
        # Componentes principales
        self.pca_ajustado = None
        self.clasificador_base = None
        
        # Información del entrenamiento
        self.num_componentes_pca = None
        self.justificacion_pca = None
        self.espacio_operacion = "PCA"  # Para distinguir de RGB
        
        # Datos originales para comparación
        self._datos_entrenamiento_originales = None
    
    def entrenar(self, datos_entrenamiento: List[Dict]) -> None:
        """
        *** FUNCIÓN PRINCIPAL DE ENTRENAMIENTO PCA ***
        Localización: Línea ~65 del archivo clasificador_pca.py
        
        PROPÓSITO: Entrena el clasificador con reducción de dimensionalidad PCA automática
        
        CÓMO FUNCIONA:
        1. Extrae píxeles balanceados de las imágenes (50% lesión, 50% sana)
        2. Aplica PCA con selección automática de componentes según criterio elegido
        3. Transforma píxeles al espacio PCA reducido (3D → 1-3D)
        4. Entrena clasificador Bayesiano gaussiano en el nuevo espacio
        5. Calcula umbral óptimo según criterio (Youden, EER, etc.)
        
        PARÁMETROS DE ENTRADA:
        - datos_entrenamiento: Lista de imágenes y máscaras para entrenar
        
        RESULTADO: Clasificador listo para usar con justificación metodológica
        
        Implementa el flujo completo:
        1. Extracción de píxeles balanceados
        2. Aplicación de PCA con selección de componentes
        3. Entrenamiento del clasificador Bayesiano en espacio PCA
        4. Selección de umbral óptimo
        
        Args:
            datos_entrenamiento: Lista de diccionarios con imágenes y máscaras
        """
        # Validar datos de entrada
        self._validar_datos_entrenamiento(datos_entrenamiento)
        
        # Guardar datos originales para comparación posterior
        self._datos_entrenamiento_originales = datos_entrenamiento
        
        print("🔄 Iniciando entrenamiento del Clasificador Bayesiano + PCA...")
        
        # Extraer píxeles balanceados (igual que el clasificador RGB)
        pixels_lesion, pixels_sana = self._extraer_pixels_por_clase(datos_entrenamiento)
        
        # Combinar datos para PCA
        print("📊 Aplicando reducción de dimensionalidad PCA...")
        todos_pixels = np.vstack([pixels_lesion, pixels_sana])
        todas_etiquetas = np.hstack([
            np.ones(len(pixels_lesion)),   # 1 = lesión
            np.zeros(len(pixels_sana))     # 0 = sana
        ])
        
        # Entrenar PCA con selección automática de componentes
        self.pca_ajustado = PCAAjustado(
            criterio_seleccion=self.criterio_pca,
            estandarizar=self.estandarizar_pca,
            **self.kwargs_pca
        )
        self.pca_ajustado.entrenar(todos_pixels, todas_etiquetas)
        
        self.num_componentes_pca = self.pca_ajustado.num_componentes
        self.justificacion_pca = self.pca_ajustado.obtener_justificacion()
        
        print(f"✅ PCA aplicado: {self.num_componentes_pca} componentes seleccionados")
        print(f"📈 Varianza preservada: {self.pca_ajustado.analisis_varianza['varianza_total_preservada']:.1%}")
        
        # Transformar datos al espacio PCA
        pixels_lesion_pca = self.pca_ajustado.transformar(pixels_lesion)
        pixels_sana_pca = self.pca_ajustado.transformar(pixels_sana)
        
        # Entrenar clasificador Bayesiano en espacio PCA
        print("🤖 Entrenando clasificador Bayesiano en espacio PCA...")
        self.clasificador_base = ClasificadorBayesianoRGB(criterio_umbral=self.criterio_umbral)
        
        # Entrenar directamente con los píxeles PCA
        try:
            self._entrenar_clasificador_base_pca(pixels_lesion_pca, pixels_sana_pca)
            
            # Marcar como entrenado SOLO si el entrenamiento fue exitoso
            self._entrenado = True
            
            print("✅ Entrenamiento completado exitosamente")
            print(f"🎯 Umbral óptimo: {self.clasificador_base.umbral:.6f}")
            
            # Imprimir justificación PCA
            print("\\n" + "="*60)
            print("JUSTIFICACIÓN DE COMPONENTES PCA:")
            print("="*60)
            print(self.justificacion_pca)
            print("="*60)
            
        except Exception as e:
            print(f"❌ Error durante el entrenamiento PCA: {e}")
            print(f"📊 Forma pixels_lesion_pca: {pixels_lesion_pca.shape}")
            print(f"📊 Forma pixels_sana_pca: {pixels_sana_pca.shape}")
            print(f"📊 Número de componentes PCA: {self.num_componentes_pca}")
            raise
    
    def clasificar(self, imagen: np.ndarray) -> np.ndarray:
        """
        *** FUNCIÓN DE CLASIFICACIÓN CON PCA ***
        Localización: Línea ~140 del archivo clasificador_pca.py
        
        PROPÓSITO: Clasifica una imagen aplicando primero transformación PCA
        
        CÓMO FUNCIONA:
        1. Toma imagen RGB (H,W,3) y la aplana a píxeles (H*W, 3)
        2. Transforma cada píxel RGB al espacio PCA usando el transformador entrenado
        3. Aplica clasificador Bayesiano en el espacio PCA reducido
        4. Calcula log-likelihood para cada clase (lesión vs sana)
        5. Aplica umbral de decisión y devuelve máscara binaria
        
        FLUJO TÉCNICO:
        RGB → PCA → Bayesiano → Decisión → Máscara
        
        RESULTADO: Máscara binaria donde True = lesión, False = sana
        
        Clasifica una imagen RGB transformándola primero al espacio PCA.
        
        Args:
            imagen: Imagen RGB normalizada [0,1] de forma (H,W,3)
            
        Returns:
            Máscara binaria de clasificación (H,W)
        """
        self._validar_entrenado()
        self._validar_imagen(imagen)
        
        h, w, c = imagen.shape
        
        # Transformar imagen al espacio PCA
        pixels_rgb = imagen.reshape(-1, 3)
        pixels_pca = self.pca_ajustado.transformar(pixels_rgb)
        
        # Clasificar píxeles en espacio PCA
        log_likelihood_lesion = self._calcular_log_likelihood_pca(pixels_pca, 'lesion')
        log_likelihood_sana = self._calcular_log_likelihood_pca(pixels_pca, 'sana')
        
        # Agregar log-priores
        log_posterior_lesion = log_likelihood_lesion + np.log(self.clasificador_base.prior_lesion)
        log_posterior_sana = log_likelihood_sana + np.log(self.clasificador_base.prior_sana)
        
        # Calcular log-ratio
        log_ratio = log_posterior_lesion - log_posterior_sana
        
        # Aplicar umbral
        mascara_plana = log_ratio > self.clasificador_base.umbral
        
        # Reshape a la forma original
        return mascara_plana.reshape(h, w)
    
    def evaluar(self, datos_test: List[Dict]) -> Dict[str, Any]:
        """
        *** FUNCIÓN DE EVALUACIÓN COMPLETA ***
        Localización: Línea ~175 del archivo clasificador_pca.py
        
        PROPÓSITO: Evalúa el rendimiento del clasificador PCA en datos de prueba
        
        CÓMO FUNCIONA:
        1. Procesa cada imagen de test aplicando clasificación PCA
        2. Compara predicciones con máscaras reales (ground truth)
        3. Calcula métricas estándar: accuracy, precision, recall, F1, AUC
        4. Calcula métricas médicas: sensibilidad, especificidad, Jaccard, Youden
        5. Genera matriz de confusión detallada
        
        MÉTRICAS CALCULADAS:
        - Exactitud, Precisión, Sensibilidad (Recall)
        - Especificidad, F1-Score, AUC-ROC
        - Índice Jaccard, Índice Youden
        - Matriz confusión (TP, TN, FP, FN)
        
        RESULTADO: Diccionario con todas las métricas para análisis completo
        
        Evalúa el clasificador en datos de prueba.
        
        Args:
            datos_test: Lista de diccionarios con imágenes y máscaras de prueba
            
        Returns:
            Diccionario con métricas de evaluación
        """
        self._validar_entrenado()
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        import time
        
        print(f"🔍 Evaluando clasificador PCA en {len(datos_test)} imágenes...")
        
        # Listas para almacenar resultados
        y_true_all = []
        y_pred_all = []
        y_prob_all = []
        
        tiempo_inicio = time.time()
        
        # Procesar cada imagen de prueba
        for i, data in enumerate(datos_test):
            imagen = data['imagen']
            mascara_real = data['mascara']
            
            # Clasificar imagen
            mascara_pred = self.clasificar(imagen)
            
            # Aplanar máscaras para métricas
            y_true = mascara_real.flatten().astype(bool)
            y_pred = mascara_pred.flatten().astype(bool)
            
            # Calcular probabilidades para AUC
            h, w, c = imagen.shape
            pixels_rgb = imagen.reshape(-1, 3)
            pixels_pca = self.pca_ajustado.transformar(pixels_rgb)
            
            log_likelihood_lesion = self._calcular_log_likelihood_pca(pixels_pca, 'lesion')
            log_likelihood_sana = self._calcular_log_likelihood_pca(pixels_pca, 'sana')
            
            log_posterior_lesion = log_likelihood_lesion + np.log(self.clasificador_base.prior_lesion)
            log_posterior_sana = log_likelihood_sana + np.log(self.clasificador_base.prior_sana)
            
            # Convertir log-posteriors a probabilidades
            log_sum = np.logaddexp(log_posterior_lesion, log_posterior_sana)
            prob_lesion = np.exp(log_posterior_lesion - log_sum)
            
            y_true_all.extend(y_true)
            y_pred_all.extend(y_pred)
            y_prob_all.extend(prob_lesion)
            
            if (i + 1) % 10 == 0:
                print(f"  Procesadas {i + 1}/{len(datos_test)} imágenes...")
        
        tiempo_evaluacion = time.time() - tiempo_inicio
        
        # Convertir a arrays numpy
        y_true_all = np.array(y_true_all)
        y_pred_all = np.array(y_pred_all)
        y_prob_all = np.array(y_prob_all)
        
        # Calcular métricas
        accuracy = accuracy_score(y_true_all, y_pred_all)
        precision = precision_score(y_true_all, y_pred_all, zero_division=0)
        recall = recall_score(y_true_all, y_pred_all, zero_division=0)
        f1 = f1_score(y_true_all, y_pred_all, zero_division=0)
        
        try:
            auc = roc_auc_score(y_true_all, y_prob_all)
        except ValueError:
            auc = 0.0  # En caso de que todas las etiquetas sean de la misma clase
        
        # Calcular especificidad y sensibilidad
        tn = np.sum((~y_true_all) & (~y_pred_all))
        fp = np.sum((~y_true_all) & y_pred_all)
        fn = np.sum(y_true_all & (~y_pred_all))
        tp = np.sum(y_true_all & y_pred_all)
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity = recall  # sensitivity = recall = tp / (tp + fn)
        
        # Calcular métricas adicionales para compatibilidad con la interfaz
        jaccard = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        youden = sensitivity + specificity - 1
        
        resultados = {
            # Métricas estándar
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'f1_score': f1,
            'auc': auc,
            
            # Métricas con nombres compatibles con la interfaz
            'exactitud': accuracy,
            'sensibilidad': sensitivity,
            'especificidad': specificity,
            'jaccard': jaccard,
            'youden': youden,
            
            # Información adicional
            'tiempo_evaluacion': tiempo_evaluacion,
            'num_imagenes': len(datos_test),
            'num_pixels_total': len(y_true_all),
            'num_pixels_lesion': np.sum(y_true_all),
            'num_pixels_sana': np.sum(~y_true_all),
            'matriz_confusion': {
                'tp': int(tp),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn)
            },
            'umbral_usado': self.clasificador_base.umbral,
            'num_componentes_pca': self.num_componentes_pca,
            'varianza_preservada': getattr(self.pca_ajustado.analisis_varianza, 'varianza_total_preservada', 0.0)
        }
        
        print(f"✅ Evaluación completada:")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall: {recall:.3f}")
        print(f"   F1-Score: {f1:.3f}")
        print(f"   AUC: {auc:.3f}")
        print(f"   Tiempo: {tiempo_evaluacion:.2f}s")
        
        return resultados
    
    def obtener_parametros(self) -> Dict[str, Any]:
        """
        Obtiene todos los parámetros del clasificador entrenado.
        
        Returns:
            Diccionario con parámetros completos del modelo PCA+Bayesiano
        """
        self._validar_entrenado()
        
        # Construir parámetros manualmente en lugar de llamar al clasificador base
        # para evitar problemas de validación cruzada
        params_base = {
            'criterio_umbral': self.criterio_umbral,
            'umbral': self.clasificador_base.umbral,
            'prior_lesion': self.clasificador_base.prior_lesion,
            'prior_sana': self.clasificador_base.prior_sana,
            'mu_lesion': self.clasificador_base.mu_lesion,
            'cov_lesion': self.clasificador_base.cov_lesion,
            'mu_sana': self.clasificador_base.mu_sana,
            'cov_sana': self.clasificador_base.cov_sana,
            'num_muestras_lesion': self.clasificador_base.num_muestras_lesion,
            'num_muestras_sana': self.clasificador_base.num_muestras_sana
        }
        
        # Obtener parámetros PCA
        params_pca = self.pca_ajustado.obtener_parametros()
        
        # Combinar información
        return {
            # Información general
            'tipo_clasificador': 'Bayesiano + PCA',
            'espacio_operacion': 'PCA',
            'entrenado': self._entrenado,
            
            # Parámetros PCA
            'criterio_pca': self.criterio_pca,
            'num_componentes_pca': self.num_componentes_pca,
            'estandarizar_pca': self.estandarizar_pca,
            'varianza_preservada': params_pca.get('varianza_total_preservada', 0),
            'reduccion_dimensional': params_pca.get('reduccion_dimensional', 'N/A'),
            
            # Parámetros del clasificador Bayesiano
            'criterio_umbral': params_base.get('criterio_umbral'),
            'umbral': params_base.get('umbral'),
            'prior_lesion': params_base.get('prior_lesion'),
            'prior_sana': params_base.get('prior_sana'),
            
            # Parámetros de los modelos gaussianos (en espacio PCA)
            'mu_lesion_pca': params_base.get('mu_lesion'),
            'cov_lesion_pca': params_base.get('cov_lesion'),
            'mu_sana_pca': params_base.get('mu_sana'),
            'cov_sana_pca': params_base.get('cov_sana'),
            
            # Información adicional
            'num_muestras_lesion': params_base.get('num_muestras_lesion'),
            'num_muestras_sana': params_base.get('num_muestras_sana')
        }
    
    def obtener_justificacion_pca(self) -> str:
        """
        Obtiene la justificación metodológica de la selección de componentes PCA.
        
        Returns:
            Justificación detallada del PCA aplicado
        """
        if not self._entrenado or not self.pca_ajustado:
            return "Clasificador no entrenado - justificación PCA no disponible"
        
        return self.justificacion_pca
    
    def obtener_analisis_pca(self) -> Dict[str, Any]:
        """
        Obtiene el análisis detallado del PCA aplicado.
        
        Returns:
            Diccionario con análisis completo del PCA
        """
        if not self._entrenado or not self.pca_ajustado:
            return {}
        
        return self.pca_ajustado.obtener_analisis_varianza()
    
    def comparar_con_rgb(self, datos_validacion: List[Dict]) -> Dict[str, Any]:
        """
        *** FUNCIÓN DE COMPARACIÓN RGB vs PCA ***
        Localización: Línea ~310 del archivo clasificador_pca.py
        
        PROPÓSITO: Compara el rendimiento PCA vs clasificador RGB equivalente
        
        CÓMO FUNCIONA:
        1. Evalúa el clasificador PCA actual en datos de validación
        2. Crea y entrena un clasificador RGB equivalente con mismo criterio
        3. Evalúa el clasificador RGB en los mismos datos
        4. Calcula diferencias entre todas las métricas
        5. Determina cuál método es superior
        
        ANÁLISIS INCLUIDO:
        - Comparación métrica por métrica (PCA vs RGB)
        - Diferencias absolutas en rendimiento
        - Recomendación automática del mejor método
        - Justificación técnica de la reducción dimensional
        
        RESULTADO: Comparación completa para toma de decisiones
        
        Compara el rendimiento con un clasificador RGB equivalente.
        
        Args:
            datos_validacion: Datos para comparación
            
        Returns:
            Diccionario con comparación detallada RGB vs PCA
        """
        self._validar_entrenado()
        
        print("⚖️ Comparando rendimiento PCA vs RGB...")
        
        # Evaluar clasificador PCA (actual)
        metricas_pca = self.evaluar(datos_validacion)
        
        # Crear y entrenar clasificador RGB equivalente
        from ..bayesiano.clasificador import ClasificadorBayesianoRGB
        clasificador_rgb = ClasificadorBayesianoRGB(criterio_umbral=self.criterio_umbral)
        
        # Usar los mismos datos de entrenamiento que se usaron para PCA
        # Necesitamos recrear los datos de entrenamiento originales
        print("🔄 Entrenando clasificador RGB equivalente...")
        clasificador_rgb.entrenar(self._datos_entrenamiento_originales)
        
        # Evaluar clasificador RGB
        metricas_rgb = clasificador_rgb.evaluar(datos_validacion)
        
        # Calcular comparación con métricas disponibles en ambos clasificadores
        metricas_comunes = ['exactitud', 'precision', 'sensibilidad', 'especificidad', 'f1_score', 'jaccard', 'youden']
        
        diferencias = {}
        for metrica in metricas_comunes:
            if metrica in metricas_pca and metrica in metricas_rgb:
                diferencias[metrica] = metricas_pca[metrica] - metricas_rgb[metrica]
        
        # AUC solo está disponible en PCA
        if 'auc' in metricas_pca:
            diferencias['auc'] = metricas_pca['auc']  # Sin comparación ya que RGB no la tiene
        
        comparacion = {
            'pca': {
                'metricas': metricas_pca,
                'dimensiones': self.num_componentes_pca,
                'varianza_preservada': self.pca_ajustado.analisis_varianza['varianza_total_preservada'],
                'criterio_seleccion': self.criterio_pca
            },
            'rgb': {
                'metricas': metricas_rgb,
                'dimensiones': 3,
                'varianza_preservada': 1.0,
                'criterio_seleccion': 'N/A'
            },
            'diferencias': diferencias,
            'recomendacion': 'PCA' if metricas_pca['f1_score'] > metricas_rgb['f1_score'] else 'RGB',
            'justificacion_pca': self.justificacion_pca
        }
        
        print("✅ Comparación completada")
        return comparacion
    
    def generar_reporte_comparativo(self, comparacion: Dict[str, Any]) -> str:
        """
        Genera reporte textual de la comparación RGB vs PCA.
        
        Args:
            comparacion: Resultado de comparar_con_rgb()
            
        Returns:
            Reporte formateado
        """
        pca_data = comparacion['pca']
        rgb_data = comparacion['rgb']
        diff = comparacion['diferencias']
        
        # Determinar qué método es mejor
        mejor_metodo = "PCA" if diff['youden'] > 0 else "RGB"
        ventaja_youden = abs(diff['youden'])
        
        return f"""
=== COMPARACIÓN: CLASIFICADOR BAYESIANO RGB vs PCA ===

CONFIGURACIÓN PCA:
• Criterio selección: {pca_data['criterio_seleccion']}
• Componentes: {pca_data['dimensiones']} (de 3 originales)
• Varianza preservada: {pca_data['varianza_preservada']:.1%}
• Reducción dimensional: {((3 - pca_data['dimensiones']) / 3 * 100):.1f}%

RENDIMIENTO COMPARATIVO:
                    RGB        PCA        Diferencia
Exactitud:      {rgb_data['metricas']['exactitud']:.4f}     {pca_data['metricas']['exactitud']:.4f}     {diff['exactitud']:+.4f}
Precisión:      {rgb_data['metricas']['precision']:.4f}     {pca_data['metricas']['precision']:.4f}     {diff['precision']:+.4f}
Sensibilidad:   {rgb_data['metricas']['sensibilidad']:.4f}     {pca_data['metricas']['sensibilidad']:.4f}     {diff['sensibilidad']:+.4f}
Especificidad:  {rgb_data['metricas']['especificidad']:.4f}     {pca_data['metricas']['especificidad']:.4f}     {diff['especificidad']:+.4f}
F1-Score:       {rgb_data['metricas']['f1_score']:.4f}     {pca_data['metricas']['f1_score']:.4f}     {diff['f1_score']:+.4f}
Jaccard:        {rgb_data['metricas']['jaccard']:.4f}     {pca_data['metricas']['jaccard']:.4f}     {diff['jaccard']:+.4f}
Youden:         {rgb_data['metricas']['youden']:.4f}     {pca_data['metricas']['youden']:.4f}     {diff['youden']:+.4f}

CONCLUSIÓN:
• Método superior: {mejor_metodo}
• Ventaja (Índice Youden): {ventaja_youden:.4f} puntos
• Eficiencia dimensional: PCA opera en {pca_data['dimensiones']}D vs 3D del RGB

JUSTIFICACIÓN METODOLÓGICA:
{"El PCA preserva" if diff['youden'] > 0 else "A pesar de que el PCA preserva"} {pca_data['varianza_preservada']:.1%} de la varianza original 
{"y logra mejor rendimiento" if diff['youden'] > 0 else "pero tiene menor rendimiento"} que el espacio RGB completo. 
{"Esto confirma que la reducción dimensional es efectiva" if diff['youden'] > 0 else "La reducción dimensional afecta ligeramente el rendimiento"} 
y {"justifica" if diff['youden'] > 0 else "requiere evaluar si justifica"} la complejidad adicional del preprocesamiento PCA.
        """.strip()
    
    def _extraer_pixels_por_clase(self, datos: List[Dict]) -> tuple:
        """
        Extrae píxeles de lesión y sanos aplicando muestreo equilibrado.
        
        Reutiliza la misma lógica del clasificador RGB base.
        """
        print("🎯 Aplicando muestreo equilibrado de píxeles...")
        
        # Importar función de muestreo equilibrado
        from ...muestreo.muestreo_equilibrado import muestreo_equilibrado
        
        # Aplicar muestreo equilibrado según condiciones del proyecto
        X_equilibrado, y_equilibrado = muestreo_equilibrado(datos)
        
        # Separar píxeles por clase
        pixels_lesion = X_equilibrado[y_equilibrado == 1]  # Píxeles de lesión
        pixels_sana = X_equilibrado[y_equilibrado == 0]    # Píxeles de piel sana
        
        print(f"✅ Muestreo equilibrado: {len(pixels_lesion)} píxeles de lesión, "
              f"{len(pixels_sana)} píxeles sanos")
        
        # Guardar datos originales para comparación posterior
        self._datos_entrenamiento_originales = datos
        
        return pixels_lesion, pixels_sana
    
    def _entrenar_clasificador_base_pca(self, pixels_lesion_pca: np.ndarray, pixels_sana_pca: np.ndarray):
        """
        *** FUNCIÓN INTERNA: ENTRENAMIENTO BAYESIANO EN ESPACIO PCA ***
        Localización: Línea ~400 del archivo clasificador_pca.py
        
        PROPÓSITO: Entrena el clasificador Bayesiano directamente en espacio PCA
        
        CÓMO FUNCIONA:
        1. Calcula estadísticas para cada clase en espacio PCA:
           - Media (μ) para píxeles lesión y sana
           - Matriz covarianza (Σ) para cada clase
        2. Aplica regularización para matrices definidas positivas
        3. Calcula probabilidades a priori de cada clase
        4. Determina umbral óptimo según criterio seleccionado
        
        TÉCNICAS APLICADAS:
        - Regularización numérica (1e-6) para estabilidad
        - Cálculo robusto de covarianzas
        - Estimación gaussiana multivariada
        
        RESULTADO: Modelo Bayesiano entrenado en espacio PCA reducido
        
        Entrena el clasificador base directamente con datos PCA.
        
        Args:
            pixels_lesion_pca: Píxeles de lesión en espacio PCA
            pixels_sana_pca: Píxeles sanos en espacio PCA
        """
        # Calcular estadísticas para cada clase en espacio PCA
        self.clasificador_base.mu_lesion = np.mean(pixels_lesion_pca, axis=0)
        self.clasificador_base.mu_sana = np.mean(pixels_sana_pca, axis=0)
        
        self.clasificador_base.cov_lesion = np.cov(pixels_lesion_pca, rowvar=False)
        self.clasificador_base.cov_sana = np.cov(pixels_sana_pca, rowvar=False)
        
        # Asegurar que las matrices de covarianza sean definidas positivas
        reg = 1e-6
        self.clasificador_base.cov_lesion += reg * np.eye(self.clasificador_base.cov_lesion.shape[0])
        self.clasificador_base.cov_sana += reg * np.eye(self.clasificador_base.cov_sana.shape[0])
        
        # Calcular priores
        n_lesion = len(pixels_lesion_pca)
        n_sana = len(pixels_sana_pca)
        total = n_lesion + n_sana
        
        self.clasificador_base.prior_lesion = n_lesion / total
        self.clasificador_base.prior_sana = n_sana / total
        
        # Calcular umbral óptimo usando el criterio especificado
        self._calcular_umbral_optimo_pca(pixels_lesion_pca, pixels_sana_pca)
        
        # Marcar como entrenado
        self.clasificador_base._entrenado = True
        self.clasificador_base.num_muestras_lesion = n_lesion
        self.clasificador_base.num_muestras_sana = n_sana
        
    def _calcular_umbral_optimo_pca(self, pixels_lesion_pca: np.ndarray, pixels_sana_pca: np.ndarray):
        """
        Calcula el umbral óptimo para clasificación en espacio PCA.
        
        Args:
            pixels_lesion_pca: Píxeles de lesión en espacio PCA
            pixels_sana_pca: Píxeles sanos en espacio PCA
        """
        # Calcular log-likelihood para cada píxel
        ll_lesion_lesion = self._calcular_log_likelihood_pca(pixels_lesion_pca, 'lesion')
        ll_lesion_sana = self._calcular_log_likelihood_pca(pixels_lesion_pca, 'sana')
        
        ll_sana_lesion = self._calcular_log_likelihood_pca(pixels_sana_pca, 'lesion')
        ll_sana_sana = self._calcular_log_likelihood_pca(pixels_sana_pca, 'sana')
        
        # Calcular log-ratio
        log_ratio_lesion = ll_lesion_lesion - ll_lesion_sana
        log_ratio_sana = ll_sana_lesion - ll_sana_sana
        
        # Aplicar criterio de umbral
        if self.criterio_umbral == 'youden':
            from sklearn.metrics import roc_curve
            y_true = np.hstack([np.ones(len(log_ratio_lesion)), np.zeros(len(log_ratio_sana))])
            y_scores = np.hstack([log_ratio_lesion, log_ratio_sana])
            
            fpr, tpr, thresholds = roc_curve(y_true, y_scores)
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            self.clasificador_base.umbral = thresholds[optimal_idx]
        
        elif self.criterio_umbral == 'bayes':
            # Umbral que minimiza error bayesiano
            self.clasificador_base.umbral = np.log(self.clasificador_base.prior_sana / self.clasificador_base.prior_lesion)
        
        elif self.criterio_umbral == 'media':
            # Umbral como media de log-ratios
            todos_log_ratios = np.hstack([log_ratio_lesion, log_ratio_sana])
            self.clasificador_base.umbral = np.mean(todos_log_ratios)
        
        else:  # 'cero' o cualquier otro
            self.clasificador_base.umbral = 0.0
    
    def _calcular_log_likelihood_pca(self, pixels: np.ndarray, clase: str) -> np.ndarray:
        """
        *** FUNCIÓN INTERNA: CÁLCULO DE LOG-LIKELIHOOD EN PCA ***
        Localización: Línea ~480 del archivo clasificador_pca.py
        
        PROPÓSITO: Calcula probabilidad log-likelihood para píxeles en espacio PCA
        
        CÓMO FUNCIONA:
        1. Usa distribución gaussiana multivariada en espacio PCA
        2. Calcula distancia de Mahalanobis: (x-μ)ᵀ Σ⁻¹ (x-μ)
        3. Aplica fórmula log-likelihood: -½[k·log(2π) + log|Σ| + d²]
        4. Usa descomposición Cholesky para eficiencia numérica
        5. Fallback robusto para casos de matrices singulares
        
        TÉCNICAS AVANZADAS:
        - Descomposición Cholesky para estabilidad
        - Pseudoinversa como fallback
        - Manejo de casos degenerados
        
        RESULTADO: Array de log-likelihoods para cada píxel
        
        Calcula log-likelihood de píxeles en espacio PCA para una clase.
        
        Args:
            pixels: Píxeles en espacio PCA
            clase: 'lesion' o 'sana'
            
        Returns:
            Log-likelihood para cada píxel
        """
        if clase == 'lesion':
            mu = self.clasificador_base.mu_lesion
            cov = self.clasificador_base.cov_lesion
        else:
            mu = self.clasificador_base.mu_sana
            cov = self.clasificador_base.cov_sana
        
        # Calcular log-likelihood multivariada gaussiana
        diff = pixels - mu
        
        try:
            # Usar descomposición de Cholesky para eficiencia y estabilidad numérica
            L = np.linalg.cholesky(cov)
            solve_triangular = scipy.linalg.solve_triangular
            
            # Resolver L * y = diff.T
            y = solve_triangular(L, diff.T, lower=True)
            
            # Calcular término cuadrático: (x-mu)^T * Sigma^-1 * (x-mu)
            maha_sq = np.sum(y**2, axis=0)
            
            # Log-determinante usando Cholesky: 2 * sum(log(diag(L)))
            log_det = 2 * np.sum(np.log(np.diag(L)))
            
            # Dimensión
            k = len(mu)
            
            # Log-likelihood
            log_likelihood = -0.5 * (k * np.log(2 * np.pi) + log_det + maha_sq)
            
        except np.linalg.LinAlgError:
            # Fallback a método más robusto pero menos eficiente
            try:
                cov_inv = np.linalg.pinv(cov)
                sign, log_det = np.linalg.slogdet(cov)
                if sign <= 0:
                    log_det = np.log(np.linalg.det(cov + 1e-6 * np.eye(cov.shape[0])))
                
                maha_sq = np.sum(diff @ cov_inv * diff, axis=1)
                k = len(mu)
                log_likelihood = -0.5 * (k * np.log(2 * np.pi) + log_det + maha_sq)
                
            except:
                # Último recurso: asumir distribución independiente
                log_likelihood = np.sum(scipy.stats.norm.logpdf(pixels, mu, np.sqrt(np.diag(cov))), axis=1)
        
        return log_likelihood
    
