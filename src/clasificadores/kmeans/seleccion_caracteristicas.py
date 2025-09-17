"""
Sistema modular de selección de características para K-Means.

Este módulo proporciona un sistema flexible para extraer y seleccionar
diferentes tipos de características de imágenes dermatoscópicas para
el análisis K-Means.

CARACTERÍSTICAS IMPLEMENTADAS:
- RGB: Canales de color directo (R, G, B)
- HSV: Espacio de color HSV (Hue, Saturation, Value)
- LAB: Espacio de color LAB (L*, a*, b*)
- Textura: Descriptores de textura (GLCM, LBP)
- Forma: Descriptores de forma de lesiones
- Estadísticas: Momentos estadísticos de distribuciones

COMBINACIONES EVALUADAS:
- RGB simple
- HSV + textura
- LAB + RGB
- Todas las características
- Selección automática basada en varianza
"""

import numpy as np
import cv2
from enum import Enum
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
try:
    from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
    from skimage.measure import regionprops, label
    from skimage.color import rgb2hsv, rgb2lab
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("⚠️ Advertencia: skimage no disponible. Funciones de textura avanzada deshabilitadas.")
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


class TipoCaracteristica(Enum):
    """Tipos de características disponibles para extracción."""
    RGB = "rgb"
    HSV = "hsv" 
    LAB = "lab"
    TEXTURA_GLCM = "textura_glcm"
    TEXTURA_LBP = "textura_lbp"
    FORMA = "forma"
    ESTADISTICAS = "estadisticas"
    GRADIENTES = "gradientes"


@dataclass
class ConfiguracionCaracteristicas:
    """Configuración para extracción de características."""
    tipos_activos: List[TipoCaracteristica]
    normalizar: bool = True
    reducir_dimensionalidad: bool = False
    max_componentes: int = 50
    
    # Parámetros específicos de textura
    glcm_distancias: List[int] = None
    glcm_angulos: List[float] = None
    lbp_radius: int = 3
    lbp_n_points: int = 24
    
    def __post_init__(self):
        if self.glcm_distancias is None:
            self.glcm_distancias = [1, 2, 3]
        if self.glcm_angulos is None:
            self.glcm_angulos = [0, np.pi/4, np.pi/2, 3*np.pi/4]


class SelectorCaracteristicas:
    """
    *** SELECTOR MODULAR DE CARACTERÍSTICAS ***
    
    Extrae y combina diferentes tipos de características de imágenes
    dermatoscópicas para análisis K-Means con selección automática
    de la mejor combinación.
    
    FUNCIONALIDAD:
    1. Extracción de múltiples tipos de características
    2. Evaluación de combinaciones de características
    3. Selección automática de mejor combinación
    4. Normalización y reducción dimensional opcional
    """
    
    def __init__(self, configuracion: ConfiguracionCaracteristicas):
        """
        Inicializa el selector de características.
        
        Args:
            configuracion: Configuración de extracción de características
        """
        self.config = configuracion
        self._extractores = {
            TipoCaracteristica.RGB: self._extraer_rgb,
            TipoCaracteristica.HSV: self._extraer_hsv,
            TipoCaracteristica.LAB: self._extraer_lab,
            TipoCaracteristica.TEXTURA_GLCM: self._extraer_textura_glcm,
            TipoCaracteristica.TEXTURA_LBP: self._extraer_textura_lbp,
            TipoCaracteristica.FORMA: self._extraer_forma,
            TipoCaracteristica.ESTADISTICAS: self._extraer_estadisticas,
            TipoCaracteristica.GRADIENTES: self._extraer_gradientes
        }
        
        # Cache para optimización
        self._cache_caracteristicas = {}
        
        # Información de la última extracción
        self.ultima_info_extraccion = None
    
    def extraer_caracteristicas(self, imagen: np.ndarray, 
                              mascara: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Extrae características según la configuración especificada.
        
        Args:
            imagen: Imagen RGB normalizada [0,1]
            mascara: Máscara opcional de región de interés
            
        Returns:
            Vector de características combinadas
        """
        if imagen.shape[-1] != 3:
            raise ValueError("La imagen debe tener 3 canales (RGB)")
        
        caracteristicas_extraidas = []
        info_extraccion = {
            'tipos_usados': [],
            'dimensiones_por_tipo': {},
            'total_dimensiones': 0
        }
        
        # Aplicar máscara si está disponible
        if mascara is not None:
            imagen_masked = imagen.copy()
            mascara_bool = mascara.astype(bool)
            # Solo procesar región de interés
            if np.any(mascara_bool):
                region_coords = np.where(mascara_bool)
                y_min, y_max = region_coords[0].min(), region_coords[0].max()
                x_min, x_max = region_coords[1].min(), region_coords[1].max()
                imagen_roi = imagen[y_min:y_max+1, x_min:x_max+1]
                mascara_roi = mascara[y_min:y_max+1, x_min:x_max+1]
            else:
                imagen_roi = imagen
                mascara_roi = None
        else:
            imagen_roi = imagen
            mascara_roi = None
        
        # Extraer cada tipo de característica
        for tipo in self.config.tipos_activos:
            try:
                caracteristicas = self._extractores[tipo](imagen_roi, mascara_roi)
                
                if caracteristicas is not None and len(caracteristicas) > 0:
                    caracteristicas_extraidas.append(caracteristicas)
                    info_extraccion['tipos_usados'].append(tipo.value)
                    info_extraccion['dimensiones_por_tipo'][tipo.value] = len(caracteristicas)
                    
            except Exception as e:
                print(f"⚠️ Error extrayendo {tipo.value}: {e}")
                continue
        
        if not caracteristicas_extraidas:
            raise ValueError("No se pudieron extraer características válidas")
        
        # Combinar todas las características
        vector_final = np.concatenate(caracteristicas_extraidas)
        
        # Normalizar si se solicita
        if self.config.normalizar:
            vector_final = self._normalizar_vector(vector_final)
        
        # Reducción dimensional si se solicita
        if self.config.reducir_dimensionalidad and len(vector_final) > self.config.max_componentes:
            vector_final = self._reducir_dimensionalidad(vector_final)
        
        # Actualizar información
        info_extraccion['total_dimensiones'] = len(vector_final)
        self.ultima_info_extraccion = info_extraccion
        
        return vector_final
    
    def extraer_caracteristicas_dataset(self, datos_imagenes: List[Dict]) -> Tuple[np.ndarray, Dict]:
        """
        Extrae características de un conjunto completo de imágenes.
        
        Args:
            datos_imagenes: Lista de diccionarios con 'imagen' y 'mascara'
            
        Returns:
            Matriz de características (n_samples, n_features) e información
        """
        caracteristicas_dataset = []
        
        print(f"🔍 Extrayendo características de {len(datos_imagenes)} imágenes...")
        print(f"📊 Tipos activos: {[t.value for t in self.config.tipos_activos]}")
        
        for i, dato in enumerate(datos_imagenes):
            try:
                imagen = dato['imagen']
                mascara = dato.get('mascara', None)
                
                # Extraer características de esta imagen
                caracteristicas = self.extraer_caracteristicas(imagen, mascara)
                caracteristicas_dataset.append(caracteristicas)
                
                if (i + 1) % 10 == 0 or (i + 1) == len(datos_imagenes):
                    print(f"   Procesadas: {i + 1}/{len(datos_imagenes)} imágenes")
                    
            except Exception as e:
                print(f"❌ Error procesando imagen {i}: {e}")
                # Usar vector de ceros para mantener consistencia
                if caracteristicas_dataset:
                    dim = len(caracteristicas_dataset[0])
                    caracteristicas_dataset.append(np.zeros(dim))
                continue
        
        if not caracteristicas_dataset:
            raise ValueError("No se pudieron extraer características de ninguna imagen")
        
        # Convertir a matriz numpy
        matriz_caracteristicas = np.array(caracteristicas_dataset)
        
        # Información del dataset
        info_dataset = {
            'n_imagenes': len(datos_imagenes),
            'n_caracteristicas': matriz_caracteristicas.shape[1],
            'tipos_caracteristicas': [t.value for t in self.config.tipos_activos],
            'dimensiones_por_tipo': self.ultima_info_extraccion['dimensiones_por_tipo'] if self.ultima_info_extraccion else {},
            'configuracion': self.config
        }
        
        print(f"✅ Características extraídas: {matriz_caracteristicas.shape}")
        
        return matriz_caracteristicas, info_dataset
    
    def _extraer_rgb(self, imagen: np.ndarray, mascara: Optional[np.ndarray] = None) -> np.ndarray:
        """Extrae características RGB básicas."""
        if mascara is not None and np.any(mascara):
            # Estadísticas de píxeles de lesión
            mascara_bool = mascara.astype(bool)
            pixeles_lesion = imagen[mascara_bool]
            
            # Estadísticas por canal
            caracteristicas = []
            for canal in range(3):  # R, G, B
                valores_canal = pixeles_lesion[:, canal]
                caracteristicas.extend([
                    np.mean(valores_canal),      # Media
                    np.std(valores_canal),       # Desviación estándar
                    np.median(valores_canal),    # Mediana
                    np.percentile(valores_canal, 25),  # Q1
                    np.percentile(valores_canal, 75),  # Q3
                ])
        else:
            # Estadísticas globales de la imagen
            caracteristicas = []
            for canal in range(3):
                valores_canal = imagen[:, :, canal].flatten()
                caracteristicas.extend([
                    np.mean(valores_canal),
                    np.std(valores_canal),
                    np.median(valores_canal),
                    np.percentile(valores_canal, 25),
                    np.percentile(valores_canal, 75),
                ])
        
        return np.array(caracteristicas)
    
    def _extraer_hsv(self, imagen: np.ndarray, mascara: Optional[np.ndarray] = None) -> np.ndarray:
        """Extrae características del espacio HSV."""
        # Convertir a HSV
        imagen_hsv = rgb2hsv(imagen)
        
        if mascara is not None and np.any(mascara):
            mascara_bool = mascara.astype(bool)
            pixeles_hsv = imagen_hsv[mascara_bool]
            
            caracteristicas = []
            nombres_canales = ['H', 'S', 'V']
            for canal in range(3):
                valores_canal = pixeles_hsv[:, canal]
                caracteristicas.extend([
                    np.mean(valores_canal),
                    np.std(valores_canal),
                    np.median(valores_canal),
                ])
        else:
            caracteristicas = []
            for canal in range(3):
                valores_canal = imagen_hsv[:, :, canal].flatten()
                caracteristicas.extend([
                    np.mean(valores_canal),
                    np.std(valores_canal),
                    np.median(valores_canal),
                ])
        
        return np.array(caracteristicas)
    
    def _extraer_lab(self, imagen: np.ndarray, mascara: Optional[np.ndarray] = None) -> np.ndarray:
        """Extrae características del espacio LAB."""
        if not SKIMAGE_AVAILABLE:
            print("⚠️ Error extrayendo lab: skimage no disponible")
            # Fallback: usar conversión básica RGB
            return self._extraer_rgb(imagen, mascara)
        
        try:
            # Convertir a LAB
            imagen_lab = rgb2lab(imagen)
        except NameError:
            print("⚠️ Error extrayendo lab: name 'rgb2lab' is not defined")
            return self._extraer_rgb(imagen, mascara)
        
        if mascara is not None and np.any(mascara):
            mascara_bool = mascara.astype(bool)
            pixeles_lab = imagen_lab[mascara_bool]
            
            caracteristicas = []
            for canal in range(3):
                valores_canal = pixeles_lab[:, canal]
                caracteristicas.extend([
                    np.mean(valores_canal),
                    np.std(valores_canal),
                ])
        else:
            caracteristicas = []
            for canal in range(3):
                valores_canal = imagen_lab[:, :, canal].flatten()
                caracteristicas.extend([
                    np.mean(valores_canal),
                    np.std(valores_canal),
                ])
        
        return np.array(caracteristicas)
    
    def _extraer_textura_glcm(self, imagen: np.ndarray, mascara: Optional[np.ndarray] = None) -> np.ndarray:
        """Extrae características de textura usando GLCM."""
        # Convertir a escala de grises
        if len(imagen.shape) == 3:
            gris = np.dot(imagen[...,:3], [0.2989, 0.5870, 0.1140])
        else:
            gris = imagen
        
        # Convertir a enteros para GLCM
        gris_int = (gris * 255).astype(np.uint8)
        
        caracteristicas = []
        
        try:
            # Calcular GLCM para diferentes distancias y ángulos
            for distancia in self.config.glcm_distancias:
                for angulo in self.config.glcm_angulos:
                    glcm = graycomatrix(gris_int, [distancia], [angulo], 
                                     levels=256, symmetric=True, normed=True)
                    
                    # Extraer propiedades de textura
                    contraste = graycoprops(glcm, 'contrast')[0, 0]
                    homogeneidad = graycoprops(glcm, 'homogeneity')[0, 0]
                    energia = graycoprops(glcm, 'energy')[0, 0]
                    correlacion = graycoprops(glcm, 'correlation')[0, 0]
                    
                    caracteristicas.extend([contraste, homogeneidad, energia, correlacion])
                    
        except Exception as e:
            print(f"⚠️ Error calculando GLCM: {e}")
            # Retornar características por defecto
            caracteristicas = [0.0] * (len(self.config.glcm_distancias) * 
                                     len(self.config.glcm_angulos) * 4)
        
        return np.array(caracteristicas)
    
    def _extraer_textura_lbp(self, imagen: np.ndarray, mascara: Optional[np.ndarray] = None) -> np.ndarray:
        """Extrae características de textura usando LBP."""
        # Convertir a escala de grises
        if len(imagen.shape) == 3:
            gris = np.dot(imagen[...,:3], [0.2989, 0.5870, 0.1140])
        else:
            gris = imagen
        
        # Convertir a entero para evitar warnings de LBP
        gris = (gris * 255).astype(np.uint8)
        
        try:
            # Calcular LBP
            lbp = local_binary_pattern(gris, self.config.lbp_n_points, 
                                     self.config.lbp_radius, method='uniform')
            
            # Aplicar máscara si está disponible
            if mascara is not None and np.any(mascara):
                mascara_bool = mascara.astype(bool)
                lbp_valores = lbp[mascara_bool]
            else:
                lbp_valores = lbp.flatten()
            
            # Calcular histograma LBP
            hist, _ = np.histogram(lbp_valores, bins=self.config.lbp_n_points + 2, 
                                 range=(0, self.config.lbp_n_points + 2))
            
            # Normalizar histograma
            hist = hist.astype(float)
            if np.sum(hist) > 0:
                hist /= np.sum(hist)
            
            return hist
            
        except Exception as e:
            print(f"⚠️ Error calculando LBP: {e}")
            return np.zeros(self.config.lbp_n_points + 2)
    
    def _extraer_forma(self, imagen: np.ndarray, mascara: Optional[np.ndarray] = None) -> np.ndarray:
        """Extrae características de forma de la lesión."""
        if mascara is None or not np.any(mascara):
            return np.array([0.0] * 8)  # Características por defecto
        
        try:
            # Etiquetar regiones conectadas
            etiquetas = label(mascara)
            regiones = regionprops(etiquetas)
            
            if not regiones:
                return np.array([0.0] * 8)
            
            # Tomar la región más grande
            region_principal = max(regiones, key=lambda r: r.area)
            
            caracteristicas = [
                region_principal.area,                    # Área
                region_principal.perimeter,               # Perímetro
                region_principal.eccentricity,            # Excentricidad
                region_principal.solidity,                # Solidez
                region_principal.extent,                  # Extensión
                region_principal.major_axis_length,       # Eje mayor
                region_principal.minor_axis_length,       # Eje menor
                4 * np.pi * region_principal.area / (region_principal.perimeter ** 2)  # Circularidad
            ]
            
            return np.array(caracteristicas)
            
        except Exception as e:
            print(f"⚠️ Error calculando forma: {e}")
            return np.array([0.0] * 8)
    
    def _extraer_estadisticas(self, imagen: np.ndarray, mascara: Optional[np.ndarray] = None) -> np.ndarray:
        """Extrae estadísticas generales de la imagen."""
        if mascara is not None and np.any(mascara):
            mascara_bool = mascara.astype(bool)
            pixeles = imagen[mascara_bool]
        else:
            pixeles = imagen.reshape(-1, imagen.shape[-1])
        
        caracteristicas = []
        
        # Estadísticas por canal
        for canal in range(pixeles.shape[1]):
            valores = pixeles[:, canal]
            caracteristicas.extend([
                np.mean(valores),
                np.var(valores),
                np.skew(valores) if len(valores) > 2 else 0.0,
                np.kurtosis(valores) if len(valores) > 3 else 0.0,
            ])
        
        return np.array(caracteristicas)
    
    def _extraer_gradientes(self, imagen: np.ndarray, mascara: Optional[np.ndarray] = None) -> np.ndarray:
        """Extrae características de gradientes."""
        # Convertir a escala de grises
        if len(imagen.shape) == 3:
            gris = np.dot(imagen[...,:3], [0.2989, 0.5870, 0.1140])
        else:
            gris = imagen
        
        # Calcular gradientes
        grad_x = np.gradient(gris, axis=1)
        grad_y = np.gradient(gris, axis=0)
        magnitud = np.sqrt(grad_x**2 + grad_y**2)
        
        if mascara is not None and np.any(mascara):
            mascara_bool = mascara.astype(bool)
            magnitud_region = magnitud[mascara_bool]
        else:
            magnitud_region = magnitud.flatten()
        
        caracteristicas = [
            np.mean(magnitud_region),
            np.std(magnitud_region),
            np.max(magnitud_region),
            np.percentile(magnitud_region, 90),
        ]
        
        return np.array(caracteristicas)
    
    def _normalizar_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normaliza el vector de características."""
        if np.std(vector) == 0:
            return vector
        return (vector - np.mean(vector)) / np.std(vector)
    
    def _reducir_dimensionalidad(self, vector: np.ndarray) -> np.ndarray:
        """Reduce la dimensionalidad del vector si es necesario."""
        # Implementación simple: tomar las primeras componentes
        # En una implementación más avanzada se podría usar PCA
        return vector[:self.config.max_componentes]


def crear_configuraciones_combinaciones() -> List[Tuple[str, ConfiguracionCaracteristicas]]:
    """
    Crea diferentes configuraciones de características para evaluar.
    
    Returns:
        Lista de tuplas (nombre, configuración) para probar
    """
    configuraciones = []
    
    # 1. RGB básico
    config_rgb = ConfiguracionCaracteristicas(
        tipos_activos=[TipoCaracteristica.RGB],
        normalizar=True
    )
    configuraciones.append(("RGB_basico", config_rgb))
    
    # 2. HSV + Textura
    config_hsv_tex = ConfiguracionCaracteristicas(
        tipos_activos=[TipoCaracteristica.HSV, TipoCaracteristica.TEXTURA_LBP],
        normalizar=True
    )
    configuraciones.append(("HSV_Textura", config_hsv_tex))
    
    # 3. LAB + RGB
    config_lab_rgb = ConfiguracionCaracteristicas(
        tipos_activos=[TipoCaracteristica.LAB, TipoCaracteristica.RGB],
        normalizar=True
    )
    configuraciones.append(("LAB_RGB", config_lab_rgb))
    
    # 4. Características completas
    config_completo = ConfiguracionCaracteristicas(
        tipos_activos=[
            TipoCaracteristica.RGB,
            TipoCaracteristica.HSV,
            TipoCaracteristica.TEXTURA_LBP,
            TipoCaracteristica.FORMA,
            TipoCaracteristica.ESTADISTICAS
        ],
        normalizar=True,
        reducir_dimensionalidad=True,
        max_componentes=30
    )
    configuraciones.append(("Completo", config_completo))
    
    # 5. Solo textura avanzada
    config_textura = ConfiguracionCaracteristicas(
        tipos_activos=[TipoCaracteristica.TEXTURA_GLCM, TipoCaracteristica.TEXTURA_LBP],
        normalizar=True
    )
    configuraciones.append(("Textura_Avanzada", config_textura))
    
    return configuraciones


# Importaciones adicionales para estadísticas
try:
    from scipy.stats import skew, kurtosis
    np.skew = skew
    np.kurtosis = kurtosis
except ImportError:
    # Fallback si scipy no está disponible
    def skew_fallback(x):
        return 0.0
    def kurtosis_fallback(x):
        return 0.0
    np.skew = skew_fallback
    np.kurtosis = kurtosis_fallback