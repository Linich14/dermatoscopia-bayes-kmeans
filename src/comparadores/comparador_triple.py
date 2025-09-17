"""
Comparador Triple: RGB + PCA + K-Means

Este módulo implementa la comparación simultánea de los tres clasificadores:
- Bayesiano RGB
- Bayesiano PCA  
- K-Means

Permite evaluar el rendimiento comparativo y generar reportes unificados.
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# Importar clasificadores
from ..clasificadores.bayesiano.clasificador import ClasificadorBayesianoRGB
from ..clasificadores.bayesiano.clasificador_pca import ClasificadorBayesianoPCA
from ..clasificadores.kmeans.clasificador import KMeansClasificador


@dataclass
class ResultadoComparacion:
    """Resultado de comparación de un clasificador individual."""
    nombre_clasificador: str
    tiempo_entrenamiento: float
    tiempo_evaluacion: float
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    metricas_especificas: Dict[str, Any]
    mejor_configuracion: str
    imagen_mejor: str
    imagen_peor: str
    notas: str = ""


@dataclass
class ReporteTriple:
    """Reporte completo de comparación de los tres clasificadores."""
    resultados: List[ResultadoComparacion]
    tiempo_total: float
    clasificador_ganador: str
    ranking: List[str]
    resumen_comparativo: Dict[str, Any]
    recomendaciones: List[str]


class ComparadorTriple:
    """
    Comparador que evalúa los tres clasificadores de forma simultánea.
    
    Funcionalidades:
    - Entrenamiento paralelo de clasificadores
    - Evaluación con las mismas métricas
    - Generación de reporte comparativo
    - Recomendaciones automáticas
    """
    
    def __init__(self, datos_train, datos_val, datos_test, stats_rgb=None):
        """
        Inicializa el comparador con los datos particionados.
        
        Args:
            datos_train: Datos de entrenamiento
            datos_val: Datos de validación  
            datos_test: Datos de test
            stats_rgb: Estadísticas RGB precalculadas (opcional)
        """
        self.datos_train = datos_train
        self.datos_val = datos_val
        self.datos_test = datos_test
        self.stats_rgb = stats_rgb
        
        # Configuraciones por defecto
        self.config_rgb = {
            'criterio': 'youden'
        }
        
        self.config_pca = {
            'criterio': 'youden',
            'criterio_pca': 'varianza'
        }
        
        self.config_kmeans = {
            'n_clusters': [2, 3],
            'caracteristicas': 'Textura_Avanzada',
            'random_state': 42
        }
        
        # Callbacks para progreso
        self.callback_progreso = None
        self.callback_status = None
    
    def set_callbacks(self, callback_progreso=None, callback_status=None):
        """Establece callbacks para reportar progreso."""
        self.callback_progreso = callback_progreso
        self.callback_status = callback_status
    
    def actualizar_configuraciones(self, config_rgb=None, config_pca=None, config_kmeans=None):
        """Actualiza las configuraciones de los clasificadores."""
        if config_rgb:
            self.config_rgb.update(config_rgb)
        if config_pca:
            self.config_pca.update(config_pca)
        if config_kmeans:
            self.config_kmeans.update(config_kmeans)
    
    def ejecutar_comparacion_completa(self, usar_paralelismo=True) -> ReporteTriple:
        """
        Ejecuta la comparación completa de los tres clasificadores.
        
        Args:
            usar_paralelismo: Si usar threads paralelos para acelerar
            
        Returns:
            ReporteTriple con todos los resultados
        """
        inicio_total = time.time()
        
        if self.callback_status:
            self.callback_status("🚀 Iniciando comparación triple RGB + PCA + K-Means...")
        
        resultados = []
        
        if usar_paralelismo:
            resultados = self._ejecutar_paralelo()
        else:
            resultados = self._ejecutar_secuencial()
        
        tiempo_total = time.time() - inicio_total
        
        # Verificar que tenemos resultados
        if not resultados:
            if self.callback_status:
                self.callback_status("❌ Error: No se pudieron obtener resultados de ningún clasificador")
            # Crear un reporte vacío
            return ReporteTriple(
                resultados=[],
                tiempo_total=tiempo_total,
                clasificador_ganador="Ninguno",
                ranking=[],
                resumen_comparativo={},
                recomendaciones=["No se pudieron evaluar los clasificadores. Verifique los datos."]
            )
        
        # Calcular ranking y análisis
        ranking = self._calcular_ranking(resultados)
        ganador = ranking[0] if ranking else "Ninguno"
        resumen = self._generar_resumen_comparativo(resultados)
        recomendaciones = self._generar_recomendaciones(resultados, resumen)
        
        if self.callback_status:
            self.callback_status(f"✅ Comparación completada en {tiempo_total:.2f}s. Ganador: {ganador}")
        
        return ReporteTriple(
            resultados=resultados,
            tiempo_total=tiempo_total,
            clasificador_ganador=ganador,
            ranking=ranking,
            resumen_comparativo=resumen,
            recomendaciones=recomendaciones
        )
    
    def _ejecutar_paralelo(self) -> List[ResultadoComparacion]:
        """Ejecuta los clasificadores en paralelo usando ThreadPoolExecutor."""
        resultados = []
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Enviar tareas
            futures = {
                executor.submit(self._evaluar_bayesiano_rgb): "RGB",
                executor.submit(self._evaluar_bayesiano_pca): "PCA", 
                executor.submit(self._evaluar_kmeans): "K-Means"
            }
            
            # Recoger resultados conforme van completando
            for future in as_completed(futures):
                clasificador = futures[future]
                try:
                    resultado = future.result()
                    resultados.append(resultado)
                    
                    if self.callback_progreso:
                        progreso = len(resultados) / 3 * 100
                        self.callback_progreso(f"Completado: {clasificador}", progreso)
                        
                except Exception as e:
                    print(f"❌ ERROR CRÍTICO en {clasificador}: {e}")
                    import traceback
                    print(f"🔍 TRACEBACK: {traceback.format_exc()}")
                    # Crear resultado de error
                    resultado_error = self._crear_resultado_error(clasificador, str(e))
                    resultados.append(resultado_error)
        
        return resultados
    
    def _ejecutar_secuencial(self) -> List[ResultadoComparacion]:
        """Ejecuta los clasificadores de forma secuencial."""
        resultados = []
        
        # Bayesiano RGB
        if self.callback_status:
            self.callback_status("🔴 Evaluando Bayesiano RGB...")
        resultado_rgb = self._evaluar_bayesiano_rgb()
        resultados.append(resultado_rgb)
        
        if self.callback_progreso:
            self.callback_progreso("Bayesiano RGB completado", 33.3)
        
        # Bayesiano PCA
        if self.callback_status:
            self.callback_status("🔵 Evaluando Bayesiano PCA...")
        resultado_pca = self._evaluar_bayesiano_pca()
        resultados.append(resultado_pca)
        
        if self.callback_progreso:
            self.callback_progreso("Bayesiano PCA completado", 66.6)
        
        # K-Means
        if self.callback_status:
            self.callback_status("🟡 Evaluando K-Means...")
        resultado_kmeans = self._evaluar_kmeans()
        resultados.append(resultado_kmeans)
        
        if self.callback_progreso:
            self.callback_progreso("K-Means completado", 100.0)
        
        return resultados
    
    def _evaluar_bayesiano_rgb(self) -> ResultadoComparacion:
        """Evalúa el clasificador Bayesiano RGB."""
        inicio = time.time()
        
        try:
            # Crear y configurar clasificador
            clasificador = ClasificadorBayesianoRGB(
                criterio_umbral=self.config_rgb['criterio']
            )
            
            # Entrenar
            inicio_entrenamiento = time.time()
            clasificador.entrenar(self.datos_train)
            tiempo_entrenamiento = time.time() - inicio_entrenamiento
            
            # Evaluar
            inicio_evaluacion = time.time()
            metricas = clasificador.evaluar(self.datos_test)
            tiempo_evaluacion = time.time() - inicio_evaluacion
            
            # Extraer métricas (usar las keys correctas del evaluador Bayesiano)
            precision = metricas.get('precision', 0.0)
            recall = metricas.get('sensibilidad', 0.0)  # sensibilidad = recall
            f1 = metricas.get('f1_score', 0.0)
            accuracy = metricas.get('exactitud', 0.0)  # exactitud = accuracy
            
            return ResultadoComparacion(
                nombre_clasificador="Bayesiano RGB",
                tiempo_entrenamiento=tiempo_entrenamiento,
                tiempo_evaluacion=tiempo_evaluacion,
                precision=precision,
                recall=recall,
                f1_score=f1,
                accuracy=accuracy,
                metricas_especificas={
                    'criterio': self.config_rgb['criterio']
                },
                mejor_configuracion=f"Criterio: {self.config_rgb['criterio']}",
                imagen_mejor=metricas.get('mejor_imagen', 'N/A'),
                imagen_peor=metricas.get('peor_imagen', 'N/A')
            )
            
        except Exception as e:
            print(f"❌ ERROR CRÍTICO en Bayesiano RGB: {e}")
            import traceback
            print(f"🔍 TRACEBACK RGB: {traceback.format_exc()}")
            return self._crear_resultado_error("Bayesiano RGB", str(e))
    
    def _evaluar_bayesiano_pca(self) -> ResultadoComparacion:
        """Evalúa el clasificador Bayesiano PCA."""
        inicio = time.time()
        
        try:
            # Crear y configurar clasificador
            clasificador = ClasificadorBayesianoPCA(
                criterio_umbral=self.config_pca['criterio'],
                criterio_pca=self.config_pca['criterio_pca']
            )
            
            # Entrenar
            inicio_entrenamiento = time.time()
            clasificador.entrenar(self.datos_train)
            tiempo_entrenamiento = time.time() - inicio_entrenamiento
            
            # Evaluar
            inicio_evaluacion = time.time()
            metricas = clasificador.evaluar(self.datos_test)
            tiempo_evaluacion = time.time() - inicio_evaluacion
            
            # Extraer métricas (usar las keys correctas del evaluador Bayesiano)
            precision = metricas.get('precision', 0.0)
            recall = metricas.get('sensibilidad', 0.0)  # sensibilidad = recall
            f1 = metricas.get('f1_score', 0.0)
            accuracy = metricas.get('exactitud', 0.0)  # exactitud = accuracy
            
            return ResultadoComparacion(
                nombre_clasificador="Bayesiano PCA",
                tiempo_entrenamiento=tiempo_entrenamiento,
                tiempo_evaluacion=tiempo_evaluacion,
                precision=precision,
                recall=recall,
                f1_score=f1,
                accuracy=accuracy,
                metricas_especificas={
                    'criterio': self.config_pca['criterio'],
                    'criterio_pca': self.config_pca['criterio_pca']
                },
                mejor_configuracion=f"Criterio: {self.config_pca['criterio']}, PCA: {self.config_pca['criterio_pca']}",
                imagen_mejor=metricas.get('mejor_imagen', 'N/A'),
                imagen_peor=metricas.get('peor_imagen', 'N/A')
            )
            
        except Exception as e:
            print(f"❌ ERROR CRÍTICO en Bayesiano PCA: {e}")
            import traceback
            print(f"🔍 TRACEBACK PCA: {traceback.format_exc()}")
            return self._crear_resultado_error("Bayesiano PCA", str(e))
    
    def _evaluar_kmeans(self) -> ResultadoComparacion:
        """Evalúa el clasificador K-Means."""
        inicio = time.time()
        
        try:
            # Crear y configurar clasificador
            clasificador = KMeansClasificador(
                n_clusters_opciones=self.config_kmeans['n_clusters'],
                random_state=self.config_kmeans['random_state']
            )
            
            # Entrenar y evaluar (K-Means hace ambos en ejecutar_analisis_completo)
            inicio_evaluacion = time.time()
            reporte = clasificador.ejecutar_analisis_completo(self.datos_train, self.datos_test)
            tiempo_total_kmeans = time.time() - inicio_evaluacion
            
            # K-Means no tiene separación clara entre entrenamiento y evaluación
            tiempo_entrenamiento = tiempo_total_kmeans * 0.7  # Estimación
            tiempo_evaluacion = tiempo_total_kmeans * 0.3
            
            print(f"🔍 DEBUG K-Means - Reporte recibido: {reporte}")
            
            # Extraer métricas del mejor resultado (estructura corregida)
            if reporte and "mejor_combinacion" in reporte and reporte["mejor_combinacion"]["score_total"] > 0:
                mejor_combinacion_info = reporte["mejor_combinacion"]
                metricas_promedio = mejor_combinacion_info.get("metricas_promedio", {})
                
                # Las métricas de K-Means son diferentes (silhouette, inercia, etc.)
                # Necesitamos adaptarlas a las métricas estándar de clasificación
                precision = metricas_promedio.get('silhouette', 0.0)
                recall = max(0, 1.0 - metricas_promedio.get('davies_bouldin', 5.0) / 5.0) if metricas_promedio.get('davies_bouldin', 0) > 0 else 0.0
                f1 = mejor_combinacion_info.get("score_total", 0.0)
                accuracy = min(metricas_promedio.get('calinski_harabasz', 0.0) / 100.0, 1.0)  # Normalizar y limitar
                
                return ResultadoComparacion(
                    nombre_clasificador="K-Means",
                    tiempo_entrenamiento=tiempo_entrenamiento,
                    tiempo_evaluacion=tiempo_evaluacion,
                    precision=precision,
                    recall=recall,
                    f1_score=f1,
                    accuracy=accuracy,
                    metricas_especificas={
                        'silhouette': metricas_promedio.get('silhouette', 0.0),
                        'inercia': metricas_promedio.get('inercia', 0.0),
                        'calinski_harabasz': metricas_promedio.get('calinski_harabasz', 0.0),
                        'davies_bouldin': metricas_promedio.get('davies_bouldin', 0.0),
                        'score_total': f1,
                        'caracteristicas': self.config_kmeans['caracteristicas'],
                        'n_imagenes_analizadas': reporte.get("resumen_general", {}).get("n_imagenes_analizadas", 0)
                    },
                    mejor_configuracion=f"Características: {mejor_combinacion_info.get('nombre', 'N/A')}, Clusters: {self.config_kmeans['n_clusters']}",
                    imagen_mejor=mejor_combinacion_info.get("mejor_imagen", "N/A"),
                    imagen_peor=mejor_combinacion_info.get("peor_imagen", "N/A"),
                    notas="Métricas adaptadas de clustering no supervisado"
                )
            else:
                print(f"❌ DEBUG K-Means - No hay resultados válidos en el reporte")
                return self._crear_resultado_error("K-Means", "No se obtuvieron resultados válidos en el reporte")
                
        except Exception as e:
            print(f"❌ ERROR CRÍTICO en K-Means: {e}")
            import traceback
            print(f"🔍 TRACEBACK K-Means: {traceback.format_exc()}")
            return self._crear_resultado_error("K-Means", str(e))
    
    def _crear_resultado_error(self, nombre_clasificador: str, error: str) -> ResultadoComparacion:
        """Crea un resultado de error para un clasificador que falló."""
        return ResultadoComparacion(
            nombre_clasificador=nombre_clasificador,
            tiempo_entrenamiento=0.0,
            tiempo_evaluacion=0.0,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            accuracy=0.0,
            metricas_especificas={'error': error},
            mejor_configuracion="ERROR",
            imagen_mejor="N/A",
            imagen_peor="N/A",
            notas=f"Error durante ejecución: {error}"
        )
    
    def _calcular_ranking(self, resultados: List[ResultadoComparacion]) -> List[str]:
        """Calcula el ranking de clasificadores basado en F1-Score."""
        if not resultados:
            return []
        
        resultados_validos = [r for r in resultados if r.f1_score > 0]
        
        # Si no hay resultados válidos, usar todos los resultados
        if not resultados_validos:
            resultados_validos = resultados
        
        resultados_ordenados = sorted(resultados_validos, key=lambda x: x.f1_score, reverse=True)
        return [r.nombre_clasificador for r in resultados_ordenados]
    
    def _generar_resumen_comparativo(self, resultados: List[ResultadoComparacion]) -> Dict[str, Any]:
        """Genera un resumen comparativo de los resultados."""
        if not resultados:
            return {}
        
        # Calcular promedios
        precision_promedio = np.mean([r.precision for r in resultados])
        recall_promedio = np.mean([r.recall for r in resultados])
        f1_promedio = np.mean([r.f1_score for r in resultados])
        accuracy_promedio = np.mean([r.accuracy for r in resultados])
        tiempo_promedio = np.mean([r.tiempo_entrenamiento + r.tiempo_evaluacion for r in resultados])
        
        # Encontrar mejores en cada métrica
        mejor_precision = max(resultados, key=lambda x: x.precision)
        mejor_recall = max(resultados, key=lambda x: x.recall)
        mejor_f1 = max(resultados, key=lambda x: x.f1_score)
        mejor_accuracy = max(resultados, key=lambda x: x.accuracy)
        mas_rapido = min(resultados, key=lambda x: x.tiempo_entrenamiento + x.tiempo_evaluacion)
        
        return {
            'promedios': {
                'precision': precision_promedio,
                'recall': recall_promedio,
                'f1_score': f1_promedio,
                'accuracy': accuracy_promedio,
                'tiempo_total': tiempo_promedio
            },
            'mejores_por_metrica': {
                'precision': {'clasificador': mejor_precision.nombre_clasificador, 'valor': mejor_precision.precision},
                'recall': {'clasificador': mejor_recall.nombre_clasificador, 'valor': mejor_recall.recall},
                'f1_score': {'clasificador': mejor_f1.nombre_clasificador, 'valor': mejor_f1.f1_score},
                'accuracy': {'clasificador': mejor_accuracy.nombre_clasificador, 'valor': mejor_accuracy.accuracy},
                'velocidad': {'clasificador': mas_rapido.nombre_clasificador, 'tiempo': mas_rapido.tiempo_entrenamiento + mas_rapido.tiempo_evaluacion}
            }
        }
    
    def _generar_recomendaciones(self, resultados: List[ResultadoComparacion], resumen: Dict[str, Any]) -> List[str]:
        """Genera recomendaciones automáticas basadas en los resultados."""
        recomendaciones = []
        
        if not resultados or not resumen:
            return ["❌ No hay suficientes resultados para generar recomendaciones"]
        
        # Recomendación basada en mejor F1-Score
        mejor_f1 = max(resultados, key=lambda x: x.f1_score)
        recomendaciones.append(f"🏆 Para mejor rendimiento general, usar {mejor_f1.nombre_clasificador} (F1-Score: {mejor_f1.f1_score:.3f})")
        
        # Recomendación basada en velocidad
        mas_rapido = min(resultados, key=lambda x: x.tiempo_entrenamiento + x.tiempo_evaluacion)
        tiempo_total = mas_rapido.tiempo_entrenamiento + mas_rapido.tiempo_evaluacion
        recomendaciones.append(f"⚡ Para aplicaciones en tiempo real, usar {mas_rapido.nombre_clasificador} (Tiempo: {tiempo_total:.2f}s)")
        
        # Recomendación basada en precision vs recall
        mejor_precision = max(resultados, key=lambda x: x.precision)
        mejor_recall = max(resultados, key=lambda x: x.recall)
        
        if mejor_precision.precision > 0.8:
            recomendaciones.append(f"🎯 Para minimizar falsos positivos, usar {mejor_precision.nombre_clasificador} (Precisión: {mejor_precision.precision:.3f})")
        
        if mejor_recall.recall > 0.8:
            recomendaciones.append(f"🔍 Para detectar más lesiones, usar {mejor_recall.nombre_clasificador} (Recall: {mejor_recall.recall:.3f})")
        
        # Análisis de variabilidad
        f1_scores = [r.f1_score for r in resultados]
        if len(f1_scores) > 1:
            std_f1 = np.std(f1_scores)
            if std_f1 < 0.05:
                recomendaciones.append("📊 Los clasificadores tienen rendimiento similar. Considera criterios secundarios como velocidad e interpretabilidad.")
            else:
                recomendaciones.append("📊 Hay diferencias significativas entre clasificadores. La elección impacta el rendimiento.")
        
        return recomendaciones


def ejecutar_comparacion_rapida(datos_train, datos_val, datos_test, stats_rgb=None) -> ReporteTriple:
    """
    Función utilitaria para ejecutar una comparación rápida con configuraciones por defecto.
    
    Args:
        datos_train: Datos de entrenamiento
        datos_val: Datos de validación
        datos_test: Datos de test
        stats_rgb: Estadísticas RGB (opcional)
        
    Returns:
        ReporteTriple con resultados
    """
    comparador = ComparadorTriple(datos_train, datos_val, datos_test, stats_rgb)
    resultado = comparador.ejecutar_comparacion_completa(usar_paralelismo=True)
    
    return resultado