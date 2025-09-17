"""
Clasificador K-Means para análisis dermatoscópico no supervisado.

Este módulo implementa el algoritmo K-Means aplicado a imágenes dermatoscópicas
según los requisitos de la pauta del proyecto.

FUNCIONALIDAD PRINCIPAL:
- Aplicar K-Means sobre cada imagen del conjunto de test
- Evaluar múltiples combinaciones de características
- Reportar la mejor combinación encontrada
- Integración con la interfaz gráfica del proyecto

REQUISITOS CUMPLIDOS (según pauta):
✅ Aplicar K-Means sobre cada imagen del conjunto de test
✅ Considerar selección de características  
✅ Reportar resultado obtenido con mejor combinación de características
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import time
import warnings

from .seleccion_caracteristicas import SelectorCaracteristicas, crear_configuraciones_combinaciones

warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class ResultadoKMeans:
    """Resultado de aplicar K-Means a una imagen."""
    imagen_nombre: str
    n_clusters: int
    etiquetas: np.ndarray
    centros: np.ndarray
    inercia: float
    silhouette: float
    calinski_harabasz: float
    davies_bouldin: float
    caracteristicas_usadas: str
    tiempo_procesamiento: float
    
    def obtener_metricas(self) -> Dict[str, float]:
        """Obtiene las métricas de evaluación como diccionario."""
        return {
            'inercia': self.inercia,
            'silhouette': self.silhouette,
            'calinski_harabasz': self.calinski_harabasz,
            'davies_bouldin': self.davies_bouldin,
            'tiempo': self.tiempo_procesamiento
        }


@dataclass
class EvaluacionCombinacion:
    """Evaluación de una combinación de características."""
    nombre_combinacion: str
    configuracion: Any
    resultados_imagenes: List[ResultadoKMeans]
    metricas_promedio: Dict[str, float]
    metricas_std: Dict[str, float]
    score_total: float
    mejor_imagen: str
    peor_imagen: str
    k_distribution_entrenamiento: Dict[int, int] = None  # Distribución de K durante entrenamiento


class KMeansClasificador:
    """
    *** CLASIFICADOR K-MEANS PARA DERMATOSCOPIA ***
    
    Implementa análisis K-Means no supervisado sobre imágenes dermatoscópicas
    con selección automática de características según requisitos de la pauta.
    
    CARACTERÍSTICAS:
    1. Aplicación K-Means sobre conjunto de test
    2. Evaluación de múltiples combinaciones de características
    3. Selección automática de mejor combinación
    4. Métricas de evaluación comprehensivas
    5. Integración modular con GUI
    """
    
    def __init__(self, n_clusters_opciones: List[int] = None, random_state: int = 42):
        """
        Inicializa el clasificador K-Means.
        
        Args:
            n_clusters_opciones: Lista de números de clusters a probar
            random_state: Semilla para reproducibilidad
        """
        self.n_clusters_opciones = n_clusters_opciones or [2, 3, 4, 5]
        self.random_state = random_state
        
        # Resultados del análisis
        self.evaluaciones_combinaciones = []
        self.mejor_combinacion = None
        self.datos_train = None
        self.datos_test = None
        
        # Estado del análisis
        self.analisis_completado = False
        self.tiempo_total_analisis = 0.0
        
        print(f"🎯 KMeansClasificador inicializado")
        print(f"   Clusters a probar: {self.n_clusters_opciones}")
    
    def ejecutar_analisis_completo(self, datos_train: List[Dict], datos_test: List[Dict]) -> Dict[str, Any]:
        """
        *** FUNCIÓN PRINCIPAL - ANÁLISIS K-MEANS COMPLETO ***
        
        Ejecuta el análisis K-Means completo según requisitos de la pauta:
        1. Entrena con conjunto de entrenamiento (60%) para encontrar mejores parámetros
        2. Evalúa con conjunto de test (20%) para validar rendimiento
        3. Reporta la mejor combinación encontrada
        
        Args:
            datos_train: Lista de imágenes de entrenamiento con estructura {'imagen', 'mascara', 'nombre'}
            datos_test: Lista de imágenes de test con estructura {'imagen', 'mascara', 'nombre'}
            
        Returns:
            Diccionario con resultados completos del análisis
        """
        print("🚀 INICIANDO ANÁLISIS K-MEANS COMPLETO")
        print("="*60)
        print(f"� Conjunto de entrenamiento: {len(datos_train)} imágenes")
        print(f"�📊 Conjunto de test: {len(datos_test)} imágenes")
        
        inicio_tiempo = time.time()
        self.datos_train = datos_train
        self.datos_test = datos_test
        
        # Obtener configuraciones de características a evaluar
        configuraciones = crear_configuraciones_combinaciones()
        print(f"🔍 Evaluando {len(configuraciones)} combinaciones de características:")
        for nombre, _ in configuraciones:
            print(f"   • {nombre}")
        
        # Evaluar cada combinación de características
        for nombre_config, configuracion in configuraciones:
            print(f"\\n🔄 Evaluando combinación: {nombre_config}")
            
            try:
                evaluacion = self._evaluar_combinacion_caracteristicas(
                    nombre_config, configuracion, datos_train, datos_test
                )
                self.evaluaciones_combinaciones.append(evaluacion)
                
                print(f"   ✅ Score total: {evaluacion.score_total:.3f}")
                
            except Exception as e:
                print(f"   ❌ Error en {nombre_config}: {e}")
                continue
        
        # Seleccionar mejor combinación
        if self.evaluaciones_combinaciones:
            self.mejor_combinacion = max(
                self.evaluaciones_combinaciones, 
                key=lambda x: x.score_total
            )
            
            print(f"\\n🏆 MEJOR COMBINACIÓN ENCONTRADA: {self.mejor_combinacion.nombre_combinacion}")
            print(f"   Score: {self.mejor_combinacion.score_total:.3f}")
            print(f"   Métricas promedio:")
            for metrica, valor in self.mejor_combinacion.metricas_promedio.items():
                print(f"     {metrica}: {valor:.3f}")
        
        self.tiempo_total_analisis = time.time() - inicio_tiempo
        self.analisis_completado = True
        
        print(f"\\n⏱️ Tiempo total de análisis: {self.tiempo_total_analisis:.2f} segundos")
        print("✅ ANÁLISIS K-MEANS COMPLETADO")
        
        return self._generar_reporte_completo()
    
    def _evaluar_combinacion_caracteristicas(self, nombre: str, configuracion, 
                                           datos_train: List[Dict], datos_test: List[Dict]) -> EvaluacionCombinacion:
        """
        Evalúa una combinación específica de características.
        Entrena con datos_train y evalúa con datos_test.
        
        Args:
            nombre: Nombre de la combinación
            configuracion: Configuración de características
            datos_train: Datos de entrenamiento para encontrar mejores parámetros
            datos_test: Datos de test para evaluación final
            
        Returns:
            Evaluación completa de la combinación
        """
        selector = SelectorCaracteristicas(configuracion)
        
        # FASE 1: ENTRENAMIENTO - Encontrar mejores parámetros con datos_train
        print(f"   📚 ENTRENAMIENTO: Procesando {len(datos_train)} imágenes...")
        mejores_parametros = self._encontrar_mejores_parametros(selector, datos_train)
        
        # FASE 2: EVALUACIÓN - Aplicar en datos_test con parámetros optimizados  
        print(f"   📊 EVALUACIÓN: Procesando {len(datos_test)} imágenes...")
        resultados_imagenes = []
        
        # Aplicar K-Means a cada imagen del conjunto de test usando mejores parámetros
        for i, dato_imagen in enumerate(datos_test):
            try:
                # Extraer características de esta imagen
                caracteristicas = selector.extraer_caracteristicas(
                    dato_imagen['imagen'], 
                    dato_imagen.get('mascara', None)
                )
                
                # Usar mejores parámetros encontrados en entrenamiento
                mejor_resultado = self._aplicar_mejores_parametros(
                    caracteristicas, dato_imagen['nombre'], nombre, mejores_parametros
                )
                
                resultados_imagenes.append(mejor_resultado)
                
                if (i + 1) % 5 == 0:
                    print(f"     Procesadas: {i + 1}/{len(datos_test)}")
                    
            except Exception as e:
                print(f"   ⚠️ Error procesando {dato_imagen.get('nombre', f'imagen_{i}')}: {e}")
                continue
        
        # Calcular métricas agregadas
        metricas_promedio, metricas_std = self._calcular_metricas_agregadas(resultados_imagenes)
        
        # Calcular score total (combinación de métricas)
        score_total = self._calcular_score_total(metricas_promedio)
        
        # Encontrar mejor y peor imagen
        mejor_imagen = max(resultados_imagenes, key=lambda x: x.silhouette).imagen_nombre
        peor_imagen = min(resultados_imagenes, key=lambda x: x.silhouette).imagen_nombre
        
        return EvaluacionCombinacion(
            nombre_combinacion=nombre,
            configuracion=configuracion,
            resultados_imagenes=resultados_imagenes,
            metricas_promedio=metricas_promedio,
            metricas_std=metricas_std,
            score_total=score_total,
            mejor_imagen=mejor_imagen,
            peor_imagen=peor_imagen,
            k_distribution_entrenamiento=mejores_parametros.get("k_distribution", {})
        )
    
    def _encontrar_mejor_k_para_imagen(self, caracteristicas: np.ndarray, 
                                     nombre_imagen: str, tipo_caracteristicas: str) -> ResultadoKMeans:
        """
        Encuentra el mejor número de clusters para una imagen específica.
        
        Args:
            caracteristicas: Vector de características de la imagen
            nombre_imagen: Nombre de la imagen
            tipo_caracteristicas: Tipo de características usadas
            
        Returns:
            Mejor resultado K-Means para esta imagen
        """
        inicio_tiempo = time.time()
        
        # Expandir características para clustering (crear puntos artificiales)
        # Esto simula tener múltiples puntos de datos de la imagen
        caracteristicas_expandidas = self._expandir_caracteristicas_para_clustering(caracteristicas)
        
        mejores_resultados = []
        
        # Probar diferentes números de clusters
        for k in self.n_clusters_opciones:
            if len(caracteristicas_expandidas) < k:
                continue  # No se puede hacer clustering con más clusters que puntos
                
            try:
                # Aplicar K-Means
                kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                etiquetas = kmeans.fit_predict(caracteristicas_expandidas)
                
                # Calcular métricas
                if len(np.unique(etiquetas)) > 1:  # Necesario para silhouette score
                    silhouette = silhouette_score(caracteristicas_expandidas, etiquetas)
                    calinski = calinski_harabasz_score(caracteristicas_expandidas, etiquetas)
                    davies_bouldin = davies_bouldin_score(caracteristicas_expandidas, etiquetas)
                else:
                    silhouette = -1.0
                    calinski = 0.0
                    davies_bouldin = float('inf')
                
                resultado = ResultadoKMeans(
                    imagen_nombre=nombre_imagen,
                    n_clusters=k,
                    etiquetas=etiquetas,
                    centros=kmeans.cluster_centers_,
                    inercia=kmeans.inertia_,
                    silhouette=silhouette,
                    calinski_harabasz=calinski,
                    davies_bouldin=davies_bouldin,
                    caracteristicas_usadas=tipo_caracteristicas,
                    tiempo_procesamiento=time.time() - inicio_tiempo
                )
                
                mejores_resultados.append(resultado)
                
            except Exception as e:
                print(f"     ⚠️ Error con k={k}: {e}")
                continue
        
        # Seleccionar mejor resultado basado en silhouette score
        if mejores_resultados:
            return max(mejores_resultados, key=lambda x: x.silhouette)
        else:
            # Resultado por defecto si falla todo
            return ResultadoKMeans(
                imagen_nombre=nombre_imagen,
                n_clusters=2,
                etiquetas=np.array([0, 1]),
                centros=np.zeros((2, len(caracteristicas))),
                inercia=0.0,
                silhouette=-1.0,
                calinski_harabasz=0.0,
                davies_bouldin=float('inf'),
                caracteristicas_usadas=tipo_caracteristicas,
                tiempo_procesamiento=time.time() - inicio_tiempo
            )
    
    def _expandir_caracteristicas_para_clustering(self, caracteristicas: np.ndarray, 
                                                n_puntos: int = 50) -> np.ndarray:
        """
        Expande un vector de características para crear múltiples puntos de clustering.
        
        Esto es necesario porque K-Means requiere múltiples puntos de datos,
        pero nosotros tenemos un vector de características por imagen.
        
        Args:
            caracteristicas: Vector de características de una imagen
            n_puntos: Número de puntos a generar
            
        Returns:
            Matriz de puntos para clustering (n_puntos, n_caracteristicas)
        """
        # Crear variaciones del vector original añadiendo ruido gaussiano
        puntos = []
        
        # Punto original
        puntos.append(caracteristicas)
        
        # Generar variaciones con ruido
        std_ruido = np.std(caracteristicas) * 0.1  # 10% del std como ruido
        
        for _ in range(n_puntos - 1):
            ruido = np.random.normal(0, std_ruido, size=caracteristicas.shape)
            punto_variado = caracteristicas + ruido
            puntos.append(punto_variado)
        
        return np.array(puntos)
    
    def _calcular_metricas_agregadas(self, resultados: List[ResultadoKMeans]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Calcula métricas agregadas de todos los resultados.
        
        Args:
            resultados: Lista de resultados K-Means
            
        Returns:
            Tupla con (métricas_promedio, métricas_std)
        """
        if not resultados:
            return {}, {}
        
        # Extraer todas las métricas
        metricas_todas = {
            'inercia': [r.inercia for r in resultados],
            'silhouette': [r.silhouette for r in resultados],
            'calinski_harabasz': [r.calinski_harabasz for r in resultados],
            'davies_bouldin': [r.davies_bouldin for r in resultados if r.davies_bouldin != float('inf')],
            'tiempo': [r.tiempo_procesamiento for r in resultados]
        }
        
        # Calcular promedios y desviaciones estándar
        metricas_promedio = {}
        metricas_std = {}
        
        for metrica, valores in metricas_todas.items():
            if valores:  # Solo si hay valores válidos
                metricas_promedio[metrica] = np.mean(valores)
                metricas_std[metrica] = np.std(valores)
            else:
                metricas_promedio[metrica] = 0.0
                metricas_std[metrica] = 0.0
        
        return metricas_promedio, metricas_std
    
    def _calcular_score_total(self, metricas_promedio: Dict[str, float]) -> float:
        """
        Calcula un score total combinando múltiples métricas.
        
        Args:
            metricas_promedio: Diccionario con métricas promedio
            
        Returns:
            Score total combinado
        """
        # Pesos para cada métrica (ajustables según importancia)
        pesos = {
            'silhouette': 0.4,      # Mayor peso a silhouette (rango [-1, 1])
            'calinski_harabasz': 0.3, # Calinski-Harabasz (mayor es mejor)
            'davies_bouldin': -0.2,   # Davies-Bouldin (menor es mejor, por eso negativo)
            'inercia': -0.1          # Inercia (menor es mejor)
        }
        
        score = 0.0
        
        for metrica, peso in pesos.items():
            if metrica in metricas_promedio:
                valor = metricas_promedio[metrica]
                
                # Normalizar según el tipo de métrica
                if metrica == 'silhouette':
                    # Silhouette ya está en [-1, 1], mapear a [0, 1]
                    valor_norm = (valor + 1) / 2
                elif metrica == 'davies_bouldin':
                    # Davies-Bouldin: menor es mejor, aplicar transformación
                    valor_norm = 1 / (1 + valor) if valor > 0 else 0
                elif metrica == 'calinski_harabasz':
                    # Normalizar usando función sigmoide
                    valor_norm = 1 / (1 + np.exp(-valor/100))
                elif metrica == 'inercia':
                    # Normalizar inercia (menor es mejor)
                    valor_norm = 1 / (1 + valor/1000) if valor > 0 else 1
                else:
                    valor_norm = valor
                
                score += peso * valor_norm
        
        return max(0.0, score)  # Asegurar que el score no sea negativo
    
    def _generar_reporte_completo(self) -> Dict[str, Any]:
        """
        Genera un reporte completo de todos los resultados del análisis.
        
        Returns:
            Diccionario con reporte completo
        """
        if not self.analisis_completado:
            return {"error": "Análisis no completado"}
        
        reporte = {
            "resumen_general": {
                "n_imagenes_analizadas": len(self.datos_test) if self.datos_test else 0,
                "n_combinaciones_evaluadas": len(self.evaluaciones_combinaciones),
                "tiempo_total_segundos": self.tiempo_total_analisis,
                "analisis_completado": self.analisis_completado
            },
            "mejor_combinacion": {
                "nombre": self.mejor_combinacion.nombre_combinacion if self.mejor_combinacion else "N/A",
                "score_total": self.mejor_combinacion.score_total if self.mejor_combinacion else 0.0,
                "metricas_promedio": self.mejor_combinacion.metricas_promedio if self.mejor_combinacion else {},
                "mejor_imagen": self.mejor_combinacion.mejor_imagen if self.mejor_combinacion else "N/A",
                "peor_imagen": self.mejor_combinacion.peor_imagen if self.mejor_combinacion else "N/A"
            },
            "todas_las_combinaciones": []
        }
        
        # Agregar información de todas las combinaciones evaluadas
        for evaluacion in self.evaluaciones_combinaciones:
            info_combinacion = {
                "nombre": evaluacion.nombre_combinacion,
                "score_total": evaluacion.score_total,
                "metricas_promedio": evaluacion.metricas_promedio,
                "metricas_std": evaluacion.metricas_std,
                "n_imagenes_procesadas": len(evaluacion.resultados_imagenes)
            }
            reporte["todas_las_combinaciones"].append(info_combinacion)
        
        # Ordenar combinaciones por score
        reporte["todas_las_combinaciones"].sort(key=lambda x: x["score_total"], reverse=True)
        
        return reporte
    
    def _encontrar_mejores_parametros(self, selector: 'SelectorCaracteristicas', datos_train: List[Dict]) -> Dict:
        """
        Encuentra los mejores parámetros de K-Means usando el conjunto de entrenamiento.
        
        Args:
            selector: Selector de características configurado
            datos_train: Datos de entrenamiento
            
        Returns:
            Diccionario con mejores parámetros encontrados
        """
        print(f"     🔍 Buscando mejores parámetros en {len(datos_train)} imágenes de entrenamiento...")
        
        mejores_scores = []
        
        # Usar TODAS las imágenes de entrenamiento disponibles para máxima precisión
        # Esto es la metodología ML correcta: entrenar con todos los datos disponibles
        muestra_size = len(datos_train)  # Usar las 90 imágenes completas
        print(f"     📊 Utilizando {muestra_size} imágenes de entrenamiento para optimización...")
        
        indices_muestra = list(range(len(datos_train)))  # Usar todas las imágenes
        
        print(f"     🎯 Imágenes seleccionadas para entrenamiento: {[datos_train[i]['nombre'] for i in indices_muestra[:5]]}{'...' if muestra_size > 5 else ''}")
        
        for i, idx in enumerate(indices_muestra):
            dato_imagen = datos_train[idx]
            try:
                print(f"       🔄 Entrenando con imagen {i+1}/{muestra_size}: {dato_imagen['nombre']}")
                
                # Extraer características
                caracteristicas = selector.extraer_caracteristicas(
                    dato_imagen['imagen'], 
                    dato_imagen.get('mascara', None)
                )
                
                print(f"         ✅ Características extraídas: {len(caracteristicas)} features")
                
                # Probar diferentes valores de K
                mejor_k = self._encontrar_mejor_k_para_imagen(
                    caracteristicas, dato_imagen['nombre'], "entrenamiento"
                )
                
                print(f"         🎯 Mejor K encontrado: {mejor_k.n_clusters} (silhouette: {mejor_k.silhouette:.3f})")
                mejores_scores.append(mejor_k)
                
            except Exception as e:
                print(f"       ⚠️ Error en imagen {dato_imagen.get('nombre', 'unknown')}: {e}")
                continue
        
        if not mejores_scores:
            print(f"       ❌ FALLO EN ENTRENAMIENTO: No se pudieron procesar imágenes de entrenamiento")
            # Valores por defecto si falló el entrenamiento
            return {"mejor_k": 2, "metodo": "fallback"}
        
        # Encontrar el K más común en los mejores resultados
        k_values = [resultado.n_clusters for resultado in mejores_scores]
        k_counts = {}
        for k in k_values:
            k_counts[k] = k_counts.get(k, 0) + 1
        
        mejor_k_global = max(k_counts.items(), key=lambda x: x[1])[0]
        
        # Estadísticas de entrenamiento
        silhouettes = [resultado.silhouette for resultado in mejores_scores]
        promedio_silhouette = np.mean(silhouettes)
        
        print(f"     ✅ ENTRENAMIENTO COMPLETADO:")
        print(f"       🎯 Mejor K global: {mejor_k_global}")
        print(f"       📊 Distribución de K: {k_counts}")
        print(f"       🔍 Silhouette promedio: {promedio_silhouette:.3f}")
        print(f"       📈 Imágenes procesadas exitosamente: {len(mejores_scores)}/{muestra_size}")
        
        return {
            "mejor_k": mejor_k_global,
            "k_distribution": k_counts,
            "promedio_silhouette": promedio_silhouette,
            "imagenes_entrenamiento": len(mejores_scores),
            "metodo": "entrenamiento"
        }
    
    def _aplicar_mejores_parametros(self, caracteristicas: np.ndarray, nombre_imagen: str, 
                                  tipo_caracteristicas: str, mejores_parametros: Dict) -> 'ResultadoKMeans':
        """
        Aplica los mejores parámetros encontrados en entrenamiento a una imagen de test.
        
        Args:
            caracteristicas: Características extraídas de la imagen
            nombre_imagen: Nombre de la imagen
            tipo_caracteristicas: Tipo de características usadas
            mejores_parametros: Parámetros optimizados del entrenamiento
            
        Returns:
            Resultado de K-Means para esta imagen
        """
        inicio_tiempo = time.time()
        
        # Usar el mejor K encontrado en entrenamiento
        k_optimal = mejores_parametros.get("mejor_k", 2)
        metodo_entrenamiento = mejores_parametros.get("metodo", "unknown")
        
        print(f"         🧪 EVALUANDO: {nombre_imagen} con K={k_optimal} (método: {metodo_entrenamiento})")
        
        # Expandir características para clustering
        caracteristicas_expandidas = self._expandir_caracteristicas_para_clustering(caracteristicas)
        
        try:
            # Aplicar K-Means con parámetros optimizados
            kmeans = KMeans(n_clusters=k_optimal, random_state=self.random_state, n_init=10)
            etiquetas = kmeans.fit_predict(caracteristicas_expandidas)
            
            # Calcular métricas
            if len(np.unique(etiquetas)) > 1:
                silhouette = silhouette_score(caracteristicas_expandidas, etiquetas)
                calinski = calinski_harabasz_score(caracteristicas_expandidas, etiquetas)
                davies_bouldin = davies_bouldin_score(caracteristicas_expandidas, etiquetas)
            else:
                silhouette = -1.0
                calinski = 0.0
                davies_bouldin = float('inf')
            
            resultado = ResultadoKMeans(
                imagen_nombre=nombre_imagen,
                n_clusters=k_optimal,
                etiquetas=etiquetas,
                centros=kmeans.cluster_centers_,
                inercia=kmeans.inertia_,
                silhouette=silhouette,
                calinski_harabasz=calinski,
                davies_bouldin=davies_bouldin,
                caracteristicas_usadas=tipo_caracteristicas,
                tiempo_procesamiento=time.time() - inicio_tiempo
            )
            
            print(f"         ✅ Evaluación exitosa: silhouette={silhouette:.3f}, K={k_optimal}")
            
            return resultado
            
        except Exception as e:
            print(f"       ⚠️ Error aplicando K-Means en {nombre_imagen}: {e}")
            # Resultado por defecto en caso de error
            return ResultadoKMeans(
                imagen_nombre=nombre_imagen,
                n_clusters=k_optimal,
                etiquetas=np.array([0] * min(50, len(caracteristicas))),
                centros=np.zeros((k_optimal, caracteristicas.shape[1] if len(caracteristicas.shape) > 1 else 1)),
                inercia=float('inf'),
                silhouette=-1.0,
                calinski_harabasz=0.0,
                davies_bouldin=float('inf'),
                caracteristicas_usadas=tipo_caracteristicas,
                tiempo_procesamiento=time.time() - inicio_tiempo
            )
    
    def obtener_resultado_mejor_combinacion(self) -> Optional[EvaluacionCombinacion]:
        """
        Obtiene el resultado de la mejor combinación de características.
        
        Returns:
            Evaluación de la mejor combinación o None si no hay resultados
        """
        return self.mejor_combinacion
    
    def obtener_resultados_todas_combinaciones(self) -> List[EvaluacionCombinacion]:
        """
        Obtiene los resultados de todas las combinaciones evaluadas.
        
        Returns:
            Lista con todas las evaluaciones realizadas
        """
        return self.evaluaciones_combinaciones.copy()