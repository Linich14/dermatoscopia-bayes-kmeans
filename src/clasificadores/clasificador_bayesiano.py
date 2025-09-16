"""
Clasificador Bayesiano RGB para segmentación de lesiones dermatoscópicas.

Este módulo implementa un clasificador Bayesiano que utiliza la razón de verosimilitud
para discriminar entre píxeles de lesión y píxeles de piel sana en imágenes dermatoscópicas.
El clasificador modela las distribuciones de probabilidad de los valores RGB usando
distribuciones gaussianas multivariadas y permite diferentes criterios para la selección
del umbral óptimo de clasificación.


"""

import numpy as np
from scipy.stats import multivariate_normal
import cv2

class ClasificadorBayesianoRGB:
    """
    Clasificador Bayesiano para segmentación de lesiones dermatoscópicas en espacio RGB.
    
    Este clasificador implementa un enfoque probabilístico para la segmentación de lesiones
    dermatoscópicas basado en la teoría de decisión bayesiana. Modela las distribuciones
    de probabilidad de los píxeles de lesión y piel sana como distribuciones gaussianas
    multivariadas en el espacio de color RGB.
    
    El clasificador utiliza la razón de verosimilitud (likelihood ratio) para tomar
    decisiones de clasificación y ofrece tres criterios diferentes para la selección
    del umbral óptimo:
    
    - Criterio de Youden: Maximiza la suma de sensibilidad y especificidad
    - Equal Error Rate: Equilibra las tasas de error de falsos positivos y negativos  
    - Prior Balanced: Considera probabilidades a priori balanceadas
    
    Attributes:
        criterio_umbral (str): Criterio utilizado para selección del umbral
        mu_lesion (np.ndarray): Vector de medias de la distribución de píxeles de lesión
        cov_lesion (np.ndarray): Matriz de covarianza de píxeles de lesión
        mu_sana (np.ndarray): Vector de medias de la distribución de píxeles sanos
        cov_sana (np.ndarray): Matriz de covarianza de píxeles sanos
        prior_lesion (float): Probabilidad a priori de píxeles de lesión
        prior_sana (float): Probabilidad a priori de píxeles sanos
        umbral (float): Umbral de decisión seleccionado
        entrenado (bool): Indica si el modelo ha sido entrenado
    """
    
    def __init__(self, criterio_umbral='youden'):
        """
        Inicializa el clasificador Bayesiano RGB.
        
        Args:
            criterio_umbral (str, optional): Criterio para selección del umbral óptimo.
                Opciones disponibles:
                - 'youden': Maximiza el índice de Youden (J = Sensibilidad + Especificidad - 1)
                - 'equal_error': Minimiza la diferencia entre FPR y FNR (Equal Error Rate)
                - 'prior_balanced': Utiliza probabilidades a priori balanceadas (P(lesión) = P(sana) = 0.5)
                Por defecto es 'youden'.
                
        Note:
            El criterio de Youden es especialmente útil en aplicaciones médicas donde
            se busca un balance entre la capacidad de detectar lesiones (sensibilidad)
            y evitar falsos positivos (especificidad).
        """
        # Validar criterio de umbral
        criterios_validos = ['youden', 'equal_error', 'prior_balanced']
        if criterio_umbral not in criterios_validos:
            raise ValueError(f"Criterio de umbral '{criterio_umbral}' no válido. "
                           f"Opciones válidas: {criterios_validos}")
        
        self.criterio_umbral = criterio_umbral
        
        # Parámetros del modelo gaussiano multivariado para píxeles de lesión
        self.mu_lesion = None      # Vector de medias RGB [R_mean, G_mean, B_mean]
        self.cov_lesion = None     # Matriz de covarianza 3x3 
        
        # Parámetros del modelo gaussiano multivariado para píxeles sanos
        self.mu_sana = None        # Vector de medias RGB [R_mean, G_mean, B_mean]
        self.cov_sana = None       # Matriz de covarianza 3x3
        
        # Probabilidades a priori estimadas de los datos de entrenamiento
        self.prior_lesion = None   # P(lesión) 
        self.prior_sana = None     # P(sana) = 1 - P(lesión)
        
        # Umbral de decisión calculado según el criterio seleccionado
        self.umbral = None
        
        # Estado del modelo
        self.entrenado = False
        
    def entrenar(self, imagenes_entrenamiento):
        """
        Entrena el clasificador estimando los parámetros de las distribuciones gaussianas.
        
        Este método realiza el entrenamiento del clasificador bayesiano mediante:
        1. Extracción de vectores de características RGB de píxeles etiquetados
        2. Estimación de parámetros de distribuciones gaussianas multivariadas
        3. Cálculo de probabilidades a priori
        4. Selección del umbral óptimo según el criterio especificado
        
        Args:
            imagenes_entrenamiento (list): Lista de diccionarios, cada uno conteniendo:
                - 'imagen': np.ndarray de forma (H,W,3) con valores RGB normalizados [0,1]
                - 'mascara': np.ndarray de forma (H,W) con valores binarios (0=sana, 1=lesión)
                
        Raises:
            ValueError: Si las imágenes de entrenamiento están vacías o mal formateadas
            np.linalg.LinAlgError: Si las matrices de covarianza no son invertibles
            
        Note:
            - Se requiere un mínimo de píxeles de cada clase para estimar parámetros robustos
            - Las matrices de covarianza se regularizan para evitar singularidades
            - El método actualiza automáticamente el estado 'entrenado' al completarse
        """
        if not imagenes_entrenamiento:
            raise ValueError("Se requieren imágenes de entrenamiento para entrenar el modelo")
        
        # Inicializar listas para almacenar píxeles de cada clase
        pixels_lesion = []  # Píxeles etiquetados como lesión
        pixels_sana = []    # Píxeles etiquetados como piel sana
        
        print("Extrayendo vectores de características RGB...")
        
        # Extraer vectores de características RGB de todas las imágenes de entrenamiento
        for i, item in enumerate(imagenes_entrenamiento):
            img = item['imagen']    # Imagen RGB normalizada [0,1]
            mask = item['mascara']  # Máscara binaria ground truth
            
            # Validar dimensiones de entrada
            if img.shape[:2] != mask.shape:
                raise ValueError(f"Imagen {i}: dimensiones no coinciden - "
                               f"imagen {img.shape[:2]}, máscara {mask.shape}")
            
            # Extraer píxeles de lesión (donde máscara = 1)
            idx_lesion = np.where(mask == 1)
            if len(idx_lesion[0]) > 0:
                # Obtener valores RGB de píxeles de lesión
                rgb_lesion = img[idx_lesion[0], idx_lesion[1], :]  # Shape: (N_lesion, 3)
                pixels_lesion.extend(rgb_lesion)
            
            # Extraer píxeles de piel sana (donde máscara = 0)
            idx_sana = np.where(mask == 0)
            if len(idx_sana[0]) > 0:
                # Obtener valores RGB de píxeles sanos
                rgb_sana = img[idx_sana[0], idx_sana[1], :]  # Shape: (N_sana, 3)
                pixels_sana.extend(rgb_sana)
        
        # Convertir listas a arrays numpy para cálculos eficientes
        pixels_lesion = np.array(pixels_lesion)  # Shape: (N_total_lesion, 3)
        pixels_sana = np.array(pixels_sana)      # Shape: (N_total_sana, 3)
        
        # Validar que tenemos suficientes muestras para entrenar
        min_samples = 10  # Mínimo de píxeles por clase
        if len(pixels_lesion) < min_samples:
            raise ValueError(f"Insuficientes píxeles de lesión para entrenar: {len(pixels_lesion)} < {min_samples}")
        if len(pixels_sana) < min_samples:
            raise ValueError(f"Insuficientes píxeles sanos para entrenar: {len(pixels_sana)} < {min_samples}")
        
        print(f"Datos extraídos: {len(pixels_lesion)} píxeles de lesión, {len(pixels_sana)} píxeles sanos")
        
        # Estimación de parámetros de la distribución gaussiana multivariada para píxeles de lesión
        print("Estimando parámetros de distribución gaussiana para píxeles de lesión...")
        self.mu_lesion = np.mean(pixels_lesion, axis=0)     # Vector de medias [R_mean, G_mean, B_mean]
        self.cov_lesion = np.cov(pixels_lesion.T)           # Matriz de covarianza 3x3
        
        # Estimación de parámetros de la distribución gaussiana multivariada para píxeles sanos
        print("Estimando parámetros de distribución gaussiana para píxeles sanos...")
        self.mu_sana = np.mean(pixels_sana, axis=0)         # Vector de medias [R_mean, G_mean, B_mean]
        self.cov_sana = np.cov(pixels_sana.T)               # Matriz de covarianza 3x3
        
        # Regularización de matrices de covarianza para evitar singularidades
        # Agregar pequeño valor en la diagonal para estabilidad numérica
        regularization = 1e-6
        self.cov_lesion += regularization * np.eye(3)
        self.cov_sana += regularization * np.eye(3)
        
        # Cálculo de probabilidades a priori basadas en la proporción de píxeles en datos de entrenamiento
        total_pixels = len(pixels_lesion) + len(pixels_sana)
        self.prior_lesion = len(pixels_lesion) / total_pixels    # P(lesión)
        self.prior_sana = len(pixels_sana) / total_pixels        # P(sana) = 1 - P(lesión)
        
        print(f"Probabilidades a priori: P(lesión)={self.prior_lesion:.3f}, P(sana)={self.prior_sana:.3f}")
        
        # Marcar modelo como entrenado ANTES de seleccionar umbral
        self.entrenado = True
        
        # Selección del umbral óptimo usando datos de entrenamiento como validación
        print(f"Seleccionando umbral óptimo usando criterio '{self.criterio_umbral}'...")
        self._seleccionar_umbral(imagenes_entrenamiento)
        
        print("Entrenamiento completado exitosamente.")
        
    def _seleccionar_umbral(self, imagenes_validacion):
        """
        Selecciona el umbral óptimo de decisión según el criterio especificado.
        
        Este método evalúa diferentes umbrales de decisión para encontrar aquel que
        optimiza el criterio seleccionado (Youden, Equal Error Rate, o Prior Balanced).
        
        Args:
            imagenes_validacion (list): Conjunto de imágenes para evaluar umbrales
                Misma estructura que en el método entrenar()
                
        Note:
            - Se evalúan múltiples umbrales candidatos en un rango razonable
            - Para cada umbral se calculan métricas de rendimiento (TPR, FPR, etc.)
            - Se selecciona el umbral que optimiza el criterio especificado
        """
        print("Calculando razones de verosimilitud para selección de umbral...")
        
        # Inicializar listas para almacenar razones de verosimilitud y etiquetas verdaderas
        razones_vero = []   # Valores de razón de verosimilitud calculados
        etiquetas_true = [] # Etiquetas verdaderas (0=sana, 1=lesión)
        
        # Calcular razones de verosimilitud para todos los píxeles de validación
        for i, item in enumerate(imagenes_validacion):
            img = item['imagen']    # Imagen RGB normalizada
            mask = item['mascara']  # Ground truth
            
            # Calcular razón de verosimilitud para cada píxel de la imagen
            # Razón = P(RGB|lesión) / P(RGB|sana)
            razones = self._calcular_razones_verosimilitud(img)
            
            # Aplanar las matrices para trabajar con vectores 1D
            razones_flat = razones.flatten()
            mask_flat = mask.flatten()
            
            # Agregar a las listas principales
            razones_vero.extend(razones_flat)
            etiquetas_true.extend(mask_flat)
        
        # Convertir a arrays numpy para cálculos eficientes
        razones_vero = np.array(razones_vero)
        etiquetas_true = np.array(etiquetas_true)
        
        print(f"Evaluando umbrales en {len(razones_vero)} píxeles...")
        
        # Generar conjunto de umbrales candidatos para evaluar
        # Usar percentiles de las razones de verosimilitud como candidatos
        umbrales_candidatos = np.percentile(razones_vero, np.linspace(1, 99, 100))
        
        mejor_umbral = None
        mejor_score = -np.inf
        
        # Evaluar cada umbral candidato
        for umbral in umbrales_candidatos:
            # Clasificar píxeles usando el umbral actual
            predicciones = (razones_vero >= umbral).astype(int)
            
            # Calcular métricas de rendimiento
            tp = np.sum((predicciones == 1) & (etiquetas_true == 1))  # Verdaderos positivos
            fp = np.sum((predicciones == 1) & (etiquetas_true == 0))  # Falsos positivos
            tn = np.sum((predicciones == 0) & (etiquetas_true == 0))  # Verdaderos negativos
            fn = np.sum((predicciones == 0) & (etiquetas_true == 1))  # Falsos negativos
            
            # Evitar división por cero
            if (tp + fn) == 0 or (tn + fp) == 0:
                continue
                
            # Calcular tasas de rendimiento
            sensibilidad = tp / (tp + fn)  # True Positive Rate (TPR)
            especificidad = tn / (tn + fp)  # True Negative Rate (TNR)
            fpr = fp / (fp + tn)           # False Positive Rate
            fnr = fn / (fn + tp)           # False Negative Rate
            
            # Evaluar según el criterio seleccionado
            if self.criterio_umbral == 'youden':
                # Criterio de Youden: maximizar J = Sensibilidad + Especificidad - 1
                score = sensibilidad + especificidad - 1
                
            elif self.criterio_umbral == 'equal_error':
                # Equal Error Rate: minimizar |FPR - FNR|
                score = -abs(fpr - fnr)  # Negativo porque queremos minimizar
                
            elif self.criterio_umbral == 'prior_balanced':
                # Prior Balanced: usar umbral que considera probabilidades a priori balanceadas
                # Usar umbral teórico cuando P(lesión) = P(sana) = 0.5
                umbral_teorico = self.prior_lesion / self.prior_sana
                score = -abs(umbral - umbral_teorico)  # Negativo porque queremos minimizar distancia
            
            # Actualizar mejor umbral si se encuentra mejor score
            if score > mejor_score:
                mejor_score = score
                mejor_umbral = umbral
        
        # Asignar el umbral óptimo encontrado
        self.umbral = mejor_umbral
        
        print(f"Umbral óptimo seleccionado: {self.umbral:.6f} (criterio: {self.criterio_umbral})")
        
    def _calcular_razones_verosimilitud(self, imagen):
        """
        Calcula la razón de verosimilitud para cada píxel de la imagen.
        
        La razón de verosimilitud se define como:
        L(x) = P(RGB|lesión) / P(RGB|sana)
        
        Donde P(RGB|clase) es la densidad de probabilidad de la distribución
        gaussiana multivariada correspondiente a cada clase.
        
        Args:
            imagen (np.ndarray): Imagen RGB de forma (H,W,3) con valores normalizados [0,1]
            
        Returns:
            np.ndarray: Matriz de razones de verosimilitud de forma (H,W)
                       Valores > 1 indican mayor probabilidad de lesión
                       Valores < 1 indican mayor probabilidad de piel sana
                       
        Note:
            - Se utiliza scipy.stats.multivariate_normal para cálculo eficiente
            - Se agrega pequeño valor epsilon para evitar división por cero
        """
        if not self.entrenado:
            raise RuntimeError("El modelo debe ser entrenado antes de calcular razones de verosimilitud")
        
        # Obtener dimensiones de la imagen
        h, w, c = imagen.shape
        
        # Reshape imagen para procesar todos los píxeles como un batch
        pixels = imagen.reshape(-1, 3)  # Shape: (H*W, 3)
        
        # Calcular densidades de probabilidad para cada clase usando distribuciones gaussianas multivariadas
        # P(RGB|lesión) - probabilidad de observar estos valores RGB dado que es lesión
        prob_lesion = multivariate_normal.pdf(pixels, mean=self.mu_lesion, cov=self.cov_lesion)
        
        # P(RGB|sana) - probabilidad de observar estos valores RGB dado que es piel sana  
        prob_sana = multivariate_normal.pdf(pixels, mean=self.mu_sana, cov=self.cov_sana)
        
        # Evitar división por cero agregando pequeño epsilon
        epsilon = 1e-10
        prob_sana = np.maximum(prob_sana, epsilon)
        
        # Calcular razón de verosimilitud: L(x) = P(x|lesión) / P(x|sana)
        razon_verosimilitud = prob_lesion / prob_sana
        
        # Reshape de vuelta a forma de imagen
        razon_verosimilitud = razon_verosimilitud.reshape(h, w)
        
        return razon_verosimilitud
    
    def clasificar(self, imagen):
        """
        Clasifica una imagen utilizando el modelo entrenado.
        
        Aplica el clasificador bayesiano entrenado para segmentar lesiones en una nueva imagen.
        Cada píxel se clasifica como lesión (1) o piel sana (0) basándose en la comparación
        de la razón de verosimilitud con el umbral óptimo.
        
        Args:
            imagen (np.ndarray): Imagen RGB de forma (H,W,3) con valores normalizados [0,1]
            
        Returns:
            np.ndarray: Máscara binaria de forma (H,W) donde:
                       1 = píxel clasificado como lesión
                       0 = píxel clasificado como piel sana
                       
        Raises:
            RuntimeError: Si el modelo no ha sido entrenado
            
        Note:
            La decisión de clasificación se basa en:
            - Si L(x) >= umbral → píxel clasificado como lesión
            - Si L(x) < umbral → píxel clasificado como piel sana
        """
        if not self.entrenado:
            raise RuntimeError("El modelo debe ser entrenado antes de clasificar")
        
        # Calcular razones de verosimilitud para todos los píxeles
        razones = self._calcular_razones_verosimilitud(imagen)
        
        # Aplicar umbral de decisión
        # razones >= umbral → lesión (1), razones < umbral → sana (0)
        mascara_predicha = (razones >= self.umbral).astype(np.uint8)
        
        return mascara_predicha
    
    def comparar_criterios(self, imagenes_validacion):
        """
        Compara el rendimiento de los tres criterios de umbral disponibles.
        
        Este método entrena temporalmente el clasificador con cada uno de los tres
        criterios de umbral y evalúa su rendimiento en un conjunto de validación,
        permitiendo una comparación objetiva entre ellos.
        
        Args:
            imagenes_validacion (list): Conjunto de imágenes para evaluar criterios
                Misma estructura que en el método entrenar()
                
        Returns:
            dict: Diccionario con resultados para cada criterio:
                {
                    'criterio1': {
                        'umbral': float,
                        'metricas': dict con métricas de rendimiento
                    },
                    'criterio2': {...},
                    'criterio3': {...}
                }
                
        Note:
            - Restaura el criterio original después de la comparación
            - Útil para análisis y selección informada del mejor criterio
        """
        if not self.entrenado:
            raise RuntimeError("El modelo debe estar entrenado para comparar criterios")
        
        # Guardar criterio original para restaurarlo después
        criterio_original = self.criterio_umbral
        umbral_original = self.umbral
        
        criterios = ['youden', 'equal_error', 'prior_balanced']
        resultados = {}
        
        print("Comparando criterios de umbral...")
        
        for criterio in criterios:
            print(f"Evaluando criterio: {criterio}")
            
            # Cambiar temporalmente el criterio
            self.criterio_umbral = criterio
            
            # Recalcular umbral para este criterio
            self._seleccionar_umbral(imagenes_validacion)
            
            # Evaluar rendimiento con este criterio
            metricas = self.evaluar(imagenes_validacion)
            
            # Guardar resultados
            resultados[criterio] = {
                'umbral': self.umbral,
                'metricas': metricas
            }
        
        # Restaurar criterio original
        self.criterio_umbral = criterio_original
        self.umbral = umbral_original
        
        print("Comparación de criterios completada.")
        return resultados
    
    def evaluar(self, imagenes_test):
        """
        Evalúa el rendimiento del clasificador en un conjunto de prueba.
        
        Calcula métricas completas de rendimiento del clasificador incluyendo
        exactitud, precisión, sensibilidad, especificidad, F1-score, índice
        de Jaccard, índice de Youden y matriz de confusión.
        
        Args:
            imagenes_test (list): Conjunto de imágenes de prueba con ground truth
                Misma estructura que en el método entrenar()
                
        Returns:
            dict: Diccionario con métricas de evaluación:
                {
                    'exactitud': float,
                    'precision': float, 
                    'sensibilidad': float,
                    'especificidad': float,
                    'f1_score': float,
                    'jaccard': float,
                    'youden': float,
                    'matriz_confusion': dict con TP, TN, FP, FN
                }
                
        Raises:
            RuntimeError: Si el modelo no ha sido entrenado
            
        Note:
            - Todas las métricas se calculan a nivel de píxel
            - La matriz de confusión proporciona conteos absolutos
            - Las métricas están normalizadas entre 0 y 1
        """
        if not self.entrenado:
            raise RuntimeError("El modelo debe estar entrenado para evaluar")
        
        # Inicializar contadores para matriz de confusión
        tp_total = 0  # Verdaderos positivos
        tn_total = 0  # Verdaderos negativos  
        fp_total = 0  # Falsos positivos
        fn_total = 0  # Falsos negativos
        
        print(f"Evaluando modelo en {len(imagenes_test)} imágenes...")
        
        # Procesar cada imagen de prueba
        for i, item in enumerate(imagenes_test):
            imagen = item['imagen']
            mascara_true = item['mascara']
            
            # Realizar predicción
            mascara_pred = self.clasificar(imagen)
            
            # Calcular elementos de matriz de confusión para esta imagen
            tp = np.sum((mascara_pred == 1) & (mascara_true == 1))
            tn = np.sum((mascara_pred == 0) & (mascara_true == 0))
            fp = np.sum((mascara_pred == 1) & (mascara_true == 0))
            fn = np.sum((mascara_pred == 0) & (mascara_true == 1))
            
            # Acumular contadores
            tp_total += tp
            tn_total += tn
            fp_total += fp
            fn_total += fn
        
        # Calcular métricas de rendimiento
        # Evitar división por cero
        total_positivos = tp_total + fn_total  # Total de píxeles de lesión en ground truth
        total_negativos = tn_total + fp_total  # Total de píxeles sanos en ground truth
        total_pixels = tp_total + tn_total + fp_total + fn_total
        
        if total_positivos == 0 or total_negativos == 0 or total_pixels == 0:
            raise ValueError("Datos de prueba insuficientes para evaluar métricas")
        
        # Métricas básicas
        exactitud = (tp_total + tn_total) / total_pixels
        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
        sensibilidad = tp_total / total_positivos  # Recall, True Positive Rate
        especificidad = tn_total / total_negativos  # True Negative Rate
        
        # F1-Score: media armónica de precisión y sensibilidad
        f1_score = 2 * (precision * sensibilidad) / (precision + sensibilidad) if (precision + sensibilidad) > 0 else 0
        
        # Índice de Jaccard: intersección sobre unión
        jaccard = tp_total / (tp_total + fp_total + fn_total) if (tp_total + fp_total + fn_total) > 0 else 0
        
        # Índice de Youden: J = Sensibilidad + Especificidad - 1
        youden = sensibilidad + especificidad - 1
        
        # Crear diccionario con todas las métricas
        metricas = {
            'exactitud': exactitud,
            'precision': precision,
            'sensibilidad': sensibilidad,
            'especificidad': especificidad,
            'f1_score': f1_score,
            'jaccard': jaccard,
            'youden': youden,
            'matriz_confusion': {
                'TP': int(tp_total),
                'TN': int(tn_total),
                'FP': int(fp_total),
                'FN': int(fn_total)
            }
        }
        
        print("Evaluación completada.")
        return metricas
    
    def obtener_parametros(self):
        """
        Obtiene los parámetros del modelo entrenado.
        
        Returns:
            dict: Diccionario con parámetros del modelo:
                {
                    'mu_lesion': np.ndarray,
                    'cov_lesion': np.ndarray,
                    'mu_sana': np.ndarray,
                    'cov_sana': np.ndarray,
                    'prior_lesion': float,
                    'prior_sana': float,
                    'umbral': float,
                    'criterio_umbral': str,
                    'entrenado': bool
                }
                
        Note:
            Útil para análisis, visualización y documentación del modelo
        """
        return {
            'mu_lesion': self.mu_lesion,
            'cov_lesion': self.cov_lesion,
            'mu_sana': self.mu_sana,
            'cov_sana': self.cov_sana,
            'prior_lesion': self.prior_lesion,
            'prior_sana': self.prior_sana,
            'umbral': self.umbral,
            'criterio_umbral': self.criterio_umbral,
            'entrenado': self.entrenado
        }
    
    def justificar_criterio_umbral(self):
        """
        Proporciona justificación textual del criterio de umbral seleccionado.
        
        Returns:
            str: Explicación detallada del criterio utilizado y su aplicabilidad clínica
        """
        justificaciones = {
            'youden': """
            El índice de Youden maximiza la suma de sensibilidad y especificidad (TPR + TNR - 1).
            Es ideal para aplicaciones médicas donde se busca un balance óptimo entre la detección
            de casos positivos y la correcta identificación de casos negativos, minimizando tanto
            falsos positivos como falsos negativos de manera equilibrada.
            """,
            
            'equal_error': """
            El criterio Equal Error Rate busca el punto donde la tasa de falsos positivos
            es igual a la tasa de falsos negativos (FPR = FNR). Es útil cuando los costos
            de ambos tipos de error son similares y se desea un clasificador equilibrado
            en términos de errores simétricos.
            """,
            
            'prior_balanced': """
            El criterio Prior Balanced ajusta el umbral considerando las probabilidades
            a priori de las clases, buscando un balance que refleje la distribución
            natural de los datos. Es apropiado cuando se desea mantener las proporciones
            observadas en la población de entrenamiento.
            """
        }
        
        return justificaciones.get(self.criterio_umbral, "Criterio no reconocido").strip()