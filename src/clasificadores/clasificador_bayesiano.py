"""
Clasificador Bayesiano RGB con razón de verosimilitud.
"""
import numpy as np
from scipy.stats import multivariate_normal
import cv2

class ClasificadorBayesianoRGB:
    """
    Clasificador Bayesiano que utiliza la razón de verosimilitud para clasificar
    píxeles RGB en lesión/no-lesión usando distribuciones gaussianas multivariadas.
    """
    
    def __init__(self, criterio_umbral='youden'):
        """
        Inicializa el clasificador.
        
        Args:
            criterio_umbral (str): Criterio para selección del umbral
                - 'youden': Maximiza J = Sensibilidad + Especificidad - 1
                - 'equal_error': Minimiza |FPR - FNR|
                - 'prior_balanced': Usa P(lesión) = P(sana) = 0.5
        """
        self.criterio_umbral = criterio_umbral
        self.mu_lesion = None
        self.cov_lesion = None
        self.mu_sana = None
        self.cov_sana = None
        self.prior_lesion = None
        self.prior_sana = None
        self.umbral = None
        self.entrenado = False
        
    def entrenar(self, imagenes_entrenamiento):
        """
        Entrena el clasificador estimando parámetros de las distribuciones gaussianas.
        
        Args:
            imagenes_entrenamiento (list): Lista de diccionarios con 'imagen' y 'mascara'
        """
        # Extraer vectores de características RGB
        pixels_lesion = []
        pixels_sana = []
        
        for item in imagenes_entrenamiento:
            img = item['imagen']  # (H,W,3) normalizada [0,1]
            mask = item['mascara']  # (H,W) binaria
            
            # Píxeles de lesión
            idx_lesion = np.where(mask == 1)
            if len(idx_lesion[0]) > 0:
                rgb_lesion = img[idx_lesion[0], idx_lesion[1], :]  # (N,3)
                pixels_lesion.extend(rgb_lesion)
            
            # Píxeles sanos
            idx_sana = np.where(mask == 0)
            if len(idx_sana[0]) > 0:
                rgb_sana = img[idx_sana[0], idx_sana[1], :]  # (M,3)
                pixels_sana.extend(rgb_sana)
        
        # Convertir a arrays
        pixels_lesion = np.array(pixels_lesion)
        pixels_sana = np.array(pixels_sana)
        
        # Estimar parámetros gaussianos multivariados
        self.mu_lesion = np.mean(pixels_lesion, axis=0)
        self.cov_lesion = np.cov(pixels_lesion.T)
        self.mu_sana = np.mean(pixels_sana, axis=0)
        self.cov_sana = np.cov(pixels_sana.T)
        
        # Estimar probabilidades a priori
        total_pixels = len(pixels_lesion) + len(pixels_sana)
        self.prior_lesion = len(pixels_lesion) / total_pixels
        self.prior_sana = len(pixels_sana) / total_pixels
        
        # Seleccionar umbral usando datos de entrenamiento
        self._seleccionar_umbral(imagenes_entrenamiento)
        
        self.entrenado = True
        
    def _seleccionar_umbral(self, imagenes_validacion):
        """
        Selecciona el umbral óptimo según el criterio especificado.
        """
        # Calcular razones de verosimilitud para datos de validación
        razones_vero = []
        etiquetas_reales = []
        
        for item in imagenes_validacion:
            img = item['imagen']
            mask = item['mascara']
            
            # Calcular razones de verosimilitud sin umbral
            razones = self._calcular_razones_verosimilitud(img)
            
            # Obtener etiquetas reales
            etiquetas = mask.flatten()
            
            razones_vero.extend(razones.flatten())
            etiquetas_reales.extend(etiquetas)
        
        razones_vero = np.array(razones_vero)
        etiquetas_reales = np.array(etiquetas_reales)
        
        # Probar diferentes umbrales
        umbrales = np.logspace(-3, 3, 100)  # Umbrales de 0.001 a 1000
        
        if self.criterio_umbral == 'youden':
            self.umbral = self._umbral_youden(razones_vero, etiquetas_reales, umbrales)
        elif self.criterio_umbral == 'equal_error':
            self.umbral = self._umbral_equal_error(razones_vero, etiquetas_reales, umbrales)
        elif self.criterio_umbral == 'prior_balanced':
            self.umbral = self.prior_sana / self.prior_lesion
        else:
            raise ValueError(f"Criterio no reconocido: {self.criterio_umbral}")
    
    def _umbral_youden(self, razones_vero, etiquetas_reales, umbrales):
        """
        Selecciona umbral que maximiza el índice de Youden (J = Sens + Spec - 1).
        """
        mejor_j = -1
        mejor_umbral = 1.0
        
        for umbral in umbrales:
            predicciones = (razones_vero > umbral).astype(int)
            tn, fp, fn, tp = self._confusion_matrix(etiquetas_reales, predicciones)
            
            if (tp + fn) > 0 and (tn + fp) > 0:
                sensibilidad = tp / (tp + fn)
                especificidad = tn / (tn + fp)
                j = sensibilidad + especificidad - 1
                
                if j > mejor_j:
                    mejor_j = j
                    mejor_umbral = umbral
        
        return mejor_umbral
    
    def _umbral_equal_error(self, razones_vero, etiquetas_reales, umbrales):
        """
        Selecciona umbral que minimiza |FPR - FNR| (Equal Error Rate).
        """
        menor_diferencia = float('inf')
        mejor_umbral = 1.0
        
        for umbral in umbrales:
            predicciones = (razones_vero > umbral).astype(int)
            tn, fp, fn, tp = self._confusion_matrix(etiquetas_reales, predicciones)
            
            if (tp + fn) > 0 and (tn + fp) > 0:
                fpr = fp / (fp + tn)  # Tasa de falsos positivos
                fnr = fn / (fn + tp)  # Tasa de falsos negativos
                diferencia = abs(fpr - fnr)
                
                if diferencia < menor_diferencia:
                    menor_diferencia = diferencia
                    mejor_umbral = umbral
        
        return mejor_umbral
    
    def _confusion_matrix(self, y_true, y_pred):
        """
        Calcula matriz de confusión.
        """
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))
        return tn, fp, fn, tp
    
    def _calcular_razones_verosimilitud(self, imagen):
        """
        Calcula las razones de verosimilitud para cada píxel sin aplicar umbral.
        
        Args:
            imagen (np.ndarray): Imagen RGB normalizada (H,W,3)
            
        Returns:
            np.ndarray: Razones de verosimilitud (H,W)
        """
        H, W, _ = imagen.shape
        
        # Reshapear para procesamiento vectorizado
        pixels = imagen.reshape(-1, 3)
        
        # Calcular verosimilitudes para cada clase
        try:
            likelihood_lesion = multivariate_normal.pdf(pixels, self.mu_lesion, self.cov_lesion)
            likelihood_sana = multivariate_normal.pdf(pixels, self.mu_sana, self.cov_sana)
        except np.linalg.LinAlgError:
            # Si hay problemas con la matriz de covarianza, usar regularización
            reg = 1e-6 * np.eye(3)
            likelihood_lesion = multivariate_normal.pdf(pixels, self.mu_lesion, self.cov_lesion + reg)
            likelihood_sana = multivariate_normal.pdf(pixels, self.mu_sana, self.cov_sana + reg)
        
        # Calcular razón de verosimilitud
        # LR = P(x|lesión) * P(lesión) / (P(x|sana) * P(sana))
        epsilon = 1e-10  # Evitar división por cero
        razon_verosimilitud = (likelihood_lesion * self.prior_lesion) / (likelihood_sana * self.prior_sana + epsilon)
        
        # Reshapear a forma original
        razones = razon_verosimilitud.reshape(H, W)
        
        return razones
    
    def clasificar(self, imagen):
        """
        Clasifica una imagen completa usando el modelo entrenado.
        
        Args:
            imagen (np.ndarray): Imagen RGB normalizada (H,W,3)
            
        Returns:
            np.ndarray: Máscara de clasificación binaria (H,W)
        """
        if not self.entrenado:
            raise ValueError("El clasificador debe ser entrenado primero")
        
        mascara_pred, _ = self._clasificar_imagen_con_razones(imagen)
        return mascara_pred
    
    def _clasificar_imagen_con_razones(self, imagen):
        """
        Clasifica imagen y retorna tanto la máscara como las razones de verosimilitud.
        """
        # Calcular razones de verosimilitud
        razones = self._calcular_razones_verosimilitud(imagen)
        
        # Aplicar umbral para clasificación
        predicciones = (razones > self.umbral).astype(np.uint8)
        
        return predicciones, razones
    
    def obtener_parametros(self):
        """
        Retorna los parámetros del modelo entrenado.
        """
        return {
            'mu_lesion': self.mu_lesion,
            'cov_lesion': self.cov_lesion,
            'mu_sana': self.mu_sana,
            'cov_sana': self.cov_sana,
            'prior_lesion': self.prior_lesion,
            'prior_sana': self.prior_sana,
            'umbral': self.umbral,
            'criterio_umbral': self.criterio_umbral
        }
        
    def justificar_criterio_umbral(self):
        """
        Justifica la selección del criterio de umbral usado.
        """
        justificaciones = {
            'youden': """
            Criterio de Youden (J = Sensibilidad + Especificidad - 1):
            - Maximiza la capacidad discriminativa balanceando ambas métricas
            - Ideal cuando se busca un punto óptimo sin sesgo hacia ninguna clase
            - Ampliamente usado en diagnóstico médico
            - Valor J ∈ [-1, 1], donde J=1 es discriminación perfecta
            """,
            'equal_error': """
            Equal Error Rate (EER):
            - Minimiza la diferencia entre tasa de falsos positivos y falsos negativos
            - Busca un punto de operación balanceado en términos de errores
            - Útil cuando el costo de ambos tipos de error es similar
            - Proporciona un punto de referencia estándar en sistemas biométricos
            """,
            'prior_balanced': """
            Umbral basado en probabilidades a priori:
            - Usa la relación P(sana)/P(lesión) como umbral
            - Refleja la distribución natural de las clases en los datos
            - Minimiza el error de clasificación cuando las distribuciones son gaussianas
            - Óptimo en términos bayesianos con costos iguales de clasificación errónea
            """
        }
        return justificaciones.get(self.criterio_umbral, "Criterio no reconocido")


def clasificar_bayes(imagen):
    """
    Función legacy para compatibilidad.
    """
    # Convertir PIL a numpy si es necesario
    if hasattr(imagen, 'convert'):
        arr = np.array(imagen.convert('RGB')) / 255.0
    else:
        arr = imagen
    
    # Retornar clasificación simulada por ahora
    return np.zeros(arr.shape[:2], dtype=np.uint8)