"""
Controladores para la lógica de aplicación.        self.parent = parent_window
        self.clasificador = None
        self.entrenado = False
        
        # Datos del proyecto
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
        # K-Means
        self.kmeans_clasificador = None
        self.kmeans_resultados = None
        self.kmeans_analisis_completado = False
        
        # Callbacks para UI
        self.status_callback: Optional[Callable[[str, str], None]] = None
        self.progress_callback: Optional[Callable[[str], None]] = Noneulo implementa el patrón MVC separando la lógica de negocio
de la presentación en la interfaz gráfica.

FUNCIONES PRINCIPALES PARA LOCALIZAR:
- train_classifier(): Línea ~60 - Entrena clasificador RGB o PCA según selección
- evaluate_classifier(): Línea ~120 - Evalúa clasificador entrenado
- compare_criteria(): Línea ~140 - Compara diferentes criterios de umbral
- compare_rgb_vs_pca(): Línea ~265 - Compara RGB vs PCA directamente
- show_pca_justification(): Línea ~307 - Muestra justificación metodológica PCA
- classify_image(): Línea ~180 - Clasifica imagen individual seleccionada por usuario

CÓMO FUNCIONA EL PATRÓN MVC:
- Modelo: Clasificadores Bayesianos (RGB y PCA)
- Vista: Interfaz gráfica (ventana_modular.py)
- Controlador: Esta clase (maneja lógica y comunicación)

FLUJO DE ENTRENAMIENTO:
1. Usuario selecciona criterios en interfaz
2. train_classifier() recibe parámetros
3. Ejecuta en hilo separado para no bloquear UI
4. Actualiza estado y progreso via callbacks
5. Notifica a la interfaz cuando termina
"""

import threading
from typing import Dict, List, Any, Optional, Callable
import tkinter as tk
from ..dialogs import EvaluationDialog, ComparisonDialog, RGBvsPCADialog


class ClassifierController:
    """
    Controlador para operaciones del clasificador Bayesiano.
    
    Maneja toda la lógica relacionada con el entrenamiento, evaluación
    y clasificación de modelos, separando la lógica de la interfaz.
    """
    
    def __init__(self, parent_window):
        """
        Inicializa el controlador del clasificador.
        
        Args:
            parent_window: Ventana principal de la aplicación
        """
        self.parent = parent_window
        self.clasificador = None
        self.entrenado = False
        
        # Datos del modelo
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
        # Callbacks para actualizar UI
        self.status_callback: Optional[Callable[[str, str], None]] = None
        self.progress_callback: Optional[Callable[[str], None]] = None
    
    def set_status_callback(self, callback: Callable[[str, str], None]):
        """Establece callback para actualizar estado en la UI."""
        self.status_callback = callback
    
    def set_progress_callback(self, callback: Callable[[str], None]):
        """Establece callback para actualizar progreso en la UI."""
        self.progress_callback = callback
    
    def _update_status(self, state: str, message: str):
        """Actualiza el estado en la UI si hay callback configurado."""
        if self.status_callback:
            self.status_callback(state, message)
    
    def _update_progress(self, message: str):
        """Actualiza el progreso en la UI si hay callback configurado."""
        if self.progress_callback:
            self.progress_callback(message)
    
    def train_classifier(self, criterio_umbral: str, usar_pca: bool = False, 
                        criterio_pca: str = 'varianza', **kwargs_pca):
        """
        *** FUNCIÓN PRINCIPAL DE ENTRENAMIENTO ***
        Localización: Línea ~60 del archivo controllers/__init__.py
        
        PROPÓSITO: Entrena clasificador RGB o PCA según selección del usuario
        
        CÓMO FUNCIONA:
        1. Verifica parámetros de entrada (criterio umbral, usar PCA, etc.)
        2. Carga datos de entrenamiento si no están disponibles
        3. Decide qué tipo de clasificador crear (RGB vs PCA)
        4. Ejecuta entrenamiento en hilo separado (no bloquea interfaz)
        5. Actualiza progreso y estado via callbacks a la UI
        
        TIPOS DE CLASIFICADOR:
        - usar_pca=False: ClasificadorBayesianoRGB (método tradicional)
        - usar_pca=True: ClasificadorBayesianoPCA (método con reducción dimensional)
        
        PARÁMETROS PCA:
        - criterio_pca: 'varianza', 'codo', 'discriminativo'
        - kwargs_pca: argumentos específicos (umbral_varianza, etc.)
        
        RESULTADO: Clasificador entrenado y listo para usar
        
        Entrena el clasificador en un hilo separado.
        
        Args:
            criterio_umbral: Criterio para selección de umbral
            usar_pca: Si usar PCA para reducción de dimensionalidad
            criterio_pca: Criterio para selección de componentes PCA
            **kwargs_pca: Argumentos adicionales para PCA
        """
        def train_worker():
            try:
                self._update_status('working', 'Cargando datos...')
                
                # Cargar datos si no están disponibles
                if not self.train_data:
                    self._load_data()
                
                # Seleccionar tipo de clasificador
                if usar_pca:
                    self._update_status('working', 'Entrenando clasificador Bayesiano + PCA...')
                    from src.clasificadores.bayesiano import ClasificadorBayesianoPCA
                    self.clasificador = ClasificadorBayesianoPCA(
                        criterio_umbral=criterio_umbral,
                        criterio_pca=criterio_pca,
                        **kwargs_pca
                    )
                    tipo_clasificador = f"Bayesiano + PCA ({criterio_pca})"
                else:
                    self._update_status('working', 'Entrenando clasificador Bayesiano RGB...')
                    from src.clasificadores.bayesiano import ClasificadorBayesianoRGB
                    self.clasificador = ClasificadorBayesianoRGB(criterio_umbral=criterio_umbral)
                    tipo_clasificador = "Bayesiano RGB"
                
                # Entrenar
                self.clasificador.entrenar(self.train_data)
                
                # Verificar entrenamiento
                if not self.clasificador.entrenado:
                    raise RuntimeError("El entrenamiento del clasificador falló")
                
                self.entrenado = True
                parametros = self.clasificador.obtener_parametros()
                
                # Mensaje de progreso específico según tipo
                if usar_pca:
                    num_componentes = parametros.get('num_componentes_pca', 'N/A')
                    varianza = parametros.get('varianza_preservada', 0)
                    progreso_msg = f"✅ {tipo_clasificador}\\nComponentes: {num_componentes}\\nVarianza: {varianza:.1%}\\nUmbral: {parametros['umbral']:.4f}"
                else:
                    progreso_msg = f"✅ {tipo_clasificador}\\nCriterio: {criterio_umbral}\\nUmbral: {parametros['umbral']:.4f}"
                
                # Actualizar UI con éxito
                self._update_status('success', 'Entrenamiento completado')
                self._update_progress(progreso_msg)
                
            except Exception as e:
                self.entrenado = False
                self._update_status('error', f'Error: {str(e)[:50]}...')
                self._update_progress(f"❌ Error: {str(e)[:30]}...")
                print(f"Error entrenando clasificador: {e}")
        
        # Ejecutar en hilo separado
        thread = threading.Thread(target=train_worker, daemon=True)
        thread.start()
    
    def evaluate_classifier(self):
        """
        Evalúa el clasificador y muestra resultados.
        """
        if not self.entrenado or not self.clasificador:
            self._show_error("Debe entrenar el clasificador primero")
            return
        
        def evaluate_worker():
            try:
                self._update_status('working', 'Evaluando modelo...')
                
                # Usar datos de test o validación
                test_data = self.test_data or self.val_data
                if not test_data:
                    raise ValueError("No hay datos de prueba disponibles")
                
                # Evaluar modelo
                metricas = self.clasificador.evaluar(test_data)
                parametros = self.clasificador.obtener_parametros()
                
                # Mostrar resultados en UI principal
                self.parent.after(0, lambda: self._show_evaluation_results(metricas, parametros))
                
                self._update_status('success', 'Evaluación completada')
                
            except Exception as e:
                self._update_status('error', f'Error: {str(e)[:50]}...')
                print(f"Error evaluando clasificador: {e}")
        
        # Ejecutar en hilo separado
        thread = threading.Thread(target=evaluate_worker, daemon=True)
        thread.start()
    
    def compare_criteria(self):
        """
        Compara diferentes criterios de umbral.
        """
        def compare_worker():
            try:
                self._update_status('working', 'Comparando criterios...')
                
                # Cargar datos si no están disponibles
                if not self.train_data:
                    self._load_data()
                
                # Crear clasificador temporal para comparación
                from src.clasificadores.bayesiano import ClasificadorBayesianoRGB
                clasificador_temporal = ClasificadorBayesianoRGB(criterio_umbral='youden')
                clasificador_temporal.entrenar(self.train_data)
                
                # Comparar criterios
                resultados = clasificador_temporal.comparar_criterios(self.val_data or self.train_data)
                
                # Mostrar resultados en UI principal
                self.parent.after(0, lambda: self._show_comparison_results(resultados))
                
                self._update_status('success', 'Comparación completada')
                
            except Exception as e:
                self._update_status('error', f'Error: {str(e)[:50]}...')
                print(f"Error comparando criterios: {e}")
        
        # Ejecutar en hilo separado
        thread = threading.Thread(target=compare_worker, daemon=True)
        thread.start()
    
    def classify_image(self):
        """
        Permite al usuario seleccionar y clasificar una imagen.
        """
        if not self.entrenado or not self.clasificador:
            self._show_error("Debe entrenar el clasificador primero")
            return
        
        try:
            from tkinter import filedialog
            import numpy as np
            from PIL import Image
            
            # Seleccionar archivo
            archivo = filedialog.askopenfilename(
                title="Seleccionar imagen para clasificar",
                filetypes=[
                    ("Imágenes", "*.jpg *.jpeg *.png *.bmp"),
                    ("JPEG", "*.jpg *.jpeg"),
                    ("PNG", "*.png"),
                    ("Todos los archivos", "*.*")
                ]
            )
            
            if not archivo:
                return
            
            self._update_status('working', 'Clasificando imagen...')
            
            # Cargar y procesar imagen
            imagen_pil = Image.open(archivo).convert('RGB')
            imagen_pil = imagen_pil.resize((256, 256))  # Tamaño estándar
            imagen_array = np.array(imagen_pil) / 255.0
            
            # Clasificar
            mascara_pred = self.clasificador.clasificar(imagen_array)
            
            # Mostrar resultado
            self._show_classification_result(imagen_array, mascara_pred, archivo)
            
            self._update_status('success', 'Clasificación completada')
            
        except Exception as e:
            self._update_status('error', f'Error: {str(e)[:50]}...')
            print(f"Error clasificando imagen: {e}")
    
    def _load_data(self):
        """Carga los datos de entrenamiento, validación y test."""
        try:
            from src.preprocesamiento.carga import cargar_imagenes_y_mascaras
            from src.preprocesamiento.particion import particionar_datos
            
            imagenes = cargar_imagenes_y_mascaras()
            self.train_data, self.val_data, self.test_data = particionar_datos(imagenes)
            
        except Exception as e:
            raise RuntimeError(f"Error cargando datos: {e}")
    
    def _show_evaluation_results(self, metricas: Dict[str, Any], modelo_info: Dict[str, Any]):
        """Muestra los resultados de evaluación en un diálogo."""
        dialog = EvaluationDialog(self.parent, metricas, modelo_info)
    
    def _show_comparison_results(self, resultados: Dict[str, Dict[str, Any]]):
        """Muestra los resultados de comparación en un diálogo."""
        dialog = ComparisonDialog(self.parent, resultados)
    
    def _show_classification_result(self, imagen_original, mascara_pred, nombre_archivo):
        """Muestra el resultado de clasificación usando la función modular."""
        from ..graficos import mostrar_resultado_clasificacion
        mostrar_resultado_clasificacion(self.parent, imagen_original, mascara_pred, nombre_archivo)
    
    def compare_rgb_vs_pca(self, criterio_umbral: str = 'youden', 
                          criterio_pca: str = 'varianza'):
        """
        *** FUNCIÓN DE COMPARACIÓN RGB vs PCA ***
        Localización: Línea ~265 del archivo controllers/__init__.py
        
        PROPÓSITO: Compara directamente el rendimiento RGB vs PCA
        
        CÓMO FUNCIONA:
        1. Carga datos de validación si no están disponibles
        2. Crea y entrena clasificador PCA con criterios especificados
        3. El clasificador PCA internamente compara con RGB equivalente
        4. Ejecuta comparación completa en hilo separado
        5. Muestra resultados en diálogo especializado RGBvsPCADialog
        
        ANÁLISIS INCLUIDO:
        - Métricas lado a lado (exactitud, precisión, etc.)
        - Diferencias absolutas en rendimiento
        - Información sobre reducción dimensional
        - Recomendación automática del mejor método
        - Justificación metodológica completa
        
        RESULTADO: Diálogo con comparación detallada para toma de decisiones
        
        Compara rendimiento RGB vs PCA.
        
        Args:
            criterio_umbral: Criterio para selección de umbral
            criterio_pca: Criterio para selección de componentes PCA
        """
        def compare_worker():
            try:
                self._update_status('working', 'Comparando RGB vs PCA...')
                
                # Cargar datos si no están disponibles
                if not self.train_data:
                    self._load_data()
                
                # Entrenar clasificador PCA
                from src.clasificadores.bayesiano import ClasificadorBayesianoPCA
                clasificador_pca = ClasificadorBayesianoPCA(
                    criterio_umbral=criterio_umbral,
                    criterio_pca=criterio_pca
                )
                clasificador_pca.entrenar(self.train_data)
                
                # Comparar con RGB
                resultados = clasificador_pca.comparar_con_rgb(self.val_data or self.train_data)
                
                # Mostrar resultados en UI principal
                self.parent.after(0, lambda: self._show_rgb_vs_pca_results(resultados, clasificador_pca))
                
                self._update_status('success', 'Comparación RGB vs PCA completada')
                
            except Exception as e:
                self._update_status('error', f'Error: {str(e)[:50]}...')
                print(f"Error comparando RGB vs PCA: {e}")
        
        # Ejecutar en hilo separado
        thread = threading.Thread(target=compare_worker, daemon=True)
        thread.start()
    
    def show_pca_justification(self):
        """
        *** FUNCIÓN PARA MOSTRAR JUSTIFICACIÓN PCA ***
        Localización: Línea ~307 del archivo controllers/__init__.py
        
        PROPÓSITO: Muestra la justificación metodológica detallada del PCA
        
        CÓMO FUNCIONA:
        1. Verifica que hay un clasificador PCA entrenado
        2. Valida que el clasificador actual tiene capacidades PCA
        3. Obtiene justificación completa del método obtener_justificacion_pca()
        4. Muestra en diálogo informativo con formato académico
        
        CONTENIDO DE LA JUSTIFICACIÓN:
        - Criterio de selección utilizado (varianza, codo, discriminativo)
        - Número de componentes seleccionados y por qué
        - Porcentaje de varianza preservada
        - Análisis de reducción dimensional
        - Justificación técnica y metodológica
        
        CUÁNDO USAR: Después de entrenar clasificador PCA para explicar decisiones
        
        Muestra la justificación detallada de la selección de componentes PCA.
        """
        if not self.entrenado or not self.clasificador:
            self._show_error("Debe entrenar un clasificador PCA primero")
            return
        
        # Verificar si es clasificador PCA
        if not hasattr(self.clasificador, 'obtener_justificacion_pca'):
            self._show_error("El clasificador actual no usa PCA")
            return
        
        try:
            justificacion = self.clasificador.obtener_justificacion_pca()
            self._show_pca_justification_dialog(justificacion)
            
        except Exception as e:
            self._show_error(f"Error obteniendo justificación PCA: {e}")
    
    def analyze_roc(self):
        """
        *** ANÁLISIS ROC COMPLETO ***
        
        Ejecuta análisis ROC comparativo entre RGB y PCA según requisitos de la pauta:
        1. Genera curvas ROC para ambos clasificadores
        2. Calcula AUC y puntos de operación
        3. Muestra visualización comparativa en área principal
        4. Aplica criterio Youden para selección de punto óptimo
        """
        if not self.entrenado or not self.clasificador:
            self._show_error("Debe entrenar un clasificador primero")
            return
        
        def roc_worker():
            try:
                self._update_status('working', 'Iniciando análisis ROC...')
                
                # Cargar datos de prueba si no están disponibles
                if not self.test_data:
                    self._load_data()
                
                # Determinar tipo de clasificador actual
                usar_pca = hasattr(self.clasificador, 'clasificador_base')
                criterio_actual = self.clasificador.criterio_umbral
                
                self._update_status('working', 'Entrenando clasificadores para comparación...')
                
                # *** ENTRENAR AMBOS CLASIFICADORES PARA COMPARACIÓN ***
                from src.clasificadores.bayesiano.clasificador import ClasificadorBayesianoRGB
                from src.clasificadores.bayesiano.clasificador_pca import ClasificadorBayesianoPCA
                
                # Entrenar RGB
                clasificador_rgb = ClasificadorBayesianoRGB(criterio_umbral=criterio_actual)
                clasificador_rgb.entrenar(self.train_data)
                
                # Entrenar PCA
                clasificador_pca = ClasificadorBayesianoPCA(
                    criterio_umbral=criterio_actual,
                    criterio_pca='varianza',
                    umbral_varianza=0.95
                )
                clasificador_pca.entrenar(self.train_data)
                
                self._update_status('working', 'Generando curvas ROC...')
                
                # *** GENERAR CURVAS ROC ***
                resultados_rgb = clasificador_rgb.generar_curva_roc(
                    self.test_data, "Clasificador Bayesiano RGB"
                )
                
                resultados_pca = clasificador_pca.generar_curva_roc(
                    self.test_data, "Clasificador Bayesiano PCA"
                )
                
                # *** PREPARAR DATOS PARA VISUALIZACIÓN ***
                datos_roc = {
                    'rgb': resultados_rgb,
                    'pca': resultados_pca,
                    'criterio': criterio_actual
                }
                
                # Mostrar en área principal
                self.parent.after(0, lambda: self._show_roc_analysis(datos_roc))
                
                self._update_status('success', f'Análisis ROC completado - AUC RGB: {resultados_rgb["auc"]:.3f}, PCA: {resultados_pca["auc"]:.3f}')
                
            except Exception as e:
                self._update_status('error', f'Error en análisis ROC: {str(e)[:50]}...')
                print(f"Error en análisis ROC: {e}")
                import traceback
                traceback.print_exc()
        
        # Ejecutar en hilo separado
        thread = threading.Thread(target=roc_worker, daemon=True)
        thread.start()
    
    def _show_roc_analysis(self, datos_roc: Dict[str, Any]):
        """
        Muestra el análisis ROC en el área principal de la interfaz.
        
        Args:
            datos_roc: Diccionario con resultados ROC de ambos clasificadores
        """
        try:
            # Importar función de graficado ROC
            from ..graficos import mostrar_analisis_roc
            
            # Mostrar en área principal
            mostrar_analisis_roc(
                self.parent.graph_frame.inner_frame,
                datos_roc['rgb'],
                datos_roc['pca'],
                datos_roc['criterio']
            )
            
        except Exception as e:
            print(f"Error mostrando análisis ROC: {e}")
            self._show_error(f"Error mostrando gráfico ROC: {e}")
    
    def _show_rgb_vs_pca_results(self, comparacion: Dict[str, Any], clasificador_pca):
        """Muestra los resultados de comparación RGB vs PCA."""
        dialog = RGBvsPCADialog(self.parent, comparacion, clasificador_pca)
    
    def _show_pca_justification_dialog(self, justificacion: str):
        """Muestra la justificación PCA en un diálogo."""
        from tkinter import messagebox
        messagebox.showinfo(
            "Justificación Metodológica PCA",
            justificacion,
            parent=self.parent
        )
    
    def _show_error(self, mensaje: str):
        """Muestra un mensaje de error."""
        from tkinter import messagebox
        messagebox.showerror("Error", mensaje)
    
    # ==========================================
    # MÉTODOS K-MEANS
    # ==========================================
    
    def execute_kmeans_analysis(self, tipo_caracteristicas: str, 
                               clusters_str: str, auto_eval: bool):
        """
        *** EJECUTAR ANÁLISIS K-MEANS COMPLETO ***
        
        Ejecuta análisis K-Means según requisitos de la pauta:
        - Aplica K-Means sobre conjunto de test
        - Evalúa combinaciones de características
        - Reporta mejor combinación encontrada
        
        Args:
            tipo_caracteristicas: Tipo de características a usar
            clusters_str: String con números de clusters separados por comas
            auto_eval: Si evaluar automáticamente todas las combinaciones
        """
        def kmeans_worker():
            try:
                self._update_status('working', 'Iniciando análisis K-Means...')
                
                # Cargar datos si no están disponibles
                if not self.test_data:
                    self._load_data()
                
                # Parsear clusters
                try:
                    clusters_list = [int(x.strip()) for x in clusters_str.split(',')]
                except ValueError:
                    clusters_list = [2, 3, 4, 5]  # Por defecto
                
                # Importar clasificador K-Means
                from src.clasificadores.kmeans import KMeansClasificador
                
                self.kmeans_clasificador = KMeansClasificador(
                    n_clusters_opciones=clusters_list,
                    random_state=42
                )
                
                self._update_status('working', 'Ejecutando K-Means sobre conjunto de test...')
                
                # Ejecutar análisis completo con entrenamiento y test separados
                self.kmeans_resultados = self.kmeans_clasificador.ejecutar_analisis_completo(
                    self.train_data, self.test_data
                )
                
                self.kmeans_analisis_completado = True
                
                # Mostrar resultados en área principal
                self.parent.after(0, lambda: self._show_kmeans_analysis_results())
                
                mejor_comb = self.kmeans_clasificador.mejor_combinacion
                if mejor_comb:
                    mensaje = f'K-Means completado - Mejor: {mejor_comb.nombre_combinacion} (Score: {mejor_comb.score_total:.3f})'
                else:
                    mensaje = 'K-Means completado'
                
                self._update_status('success', mensaje)
                
            except Exception as e:
                self._update_status('error', f'Error K-Means: {str(e)[:50]}...')
                print(f"Error en análisis K-Means: {e}")
                import traceback
                traceback.print_exc()
        
        # Ejecutar en hilo separado
        thread = threading.Thread(target=kmeans_worker, daemon=True)
        thread.start()
    
    def show_best_kmeans_combination(self):
        """Muestra información detallada de la mejor combinación."""
        if not self.kmeans_analisis_completado or not self.kmeans_clasificador:
            self._show_error("Debe ejecutar el análisis K-Means primero")
            return
        
        mejor_comb = self.kmeans_clasificador.obtener_resultado_mejor_combinacion()
        if not mejor_comb:
            self._show_error("No hay resultados de mejor combinación disponibles")
            return
        
        try:
            # Mostrar información detallada
            self._show_best_combination_dialog(mejor_comb)
            
        except Exception as e:
            self._show_error(f"Error mostrando mejor combinación: {e}")
    
    def _show_kmeans_analysis_results(self):
        """
        Muestra los resultados del análisis K-Means en el área principal.
        """
        try:
            # Importar función de visualización K-Means
            from ..graficos import mostrar_resultados_kmeans
            
            # Obtener reporte completo con todas las combinaciones
            reporte_completo = self.kmeans_clasificador._generar_reporte_completo()
            
            # Mostrar en área principal
            mostrar_resultados_kmeans(
                self.parent.graph_frame.inner_frame,
                reporte_completo,
                self.kmeans_clasificador.mejor_combinacion
            )
            
        except Exception as e:
            print(f"Error mostrando resultados K-Means: {e}")
            self._show_error(f"Error mostrando gráfico K-Means: {e}")
    
    def _show_best_combination_dialog(self, mejor_combinacion):
        """Muestra diálogo con información de la mejor combinación."""
        info_texto = f"""🏆 MEJOR COMBINACIÓN DE CARACTERÍSTICAS K-MEANS

📊 Combinación: {mejor_combinacion.nombre_combinacion}
🎯 Score Total: {mejor_combinacion.score_total:.3f}

📈 MÉTRICAS PROMEDIO:
"""
        
        for metrica, valor in mejor_combinacion.metricas_promedio.items():
            info_texto += f"   • {metrica}: {valor:.3f}\n"
        
        info_texto += f"""
✅ Mejor imagen: {mejor_combinacion.mejor_imagen}
⚠️ Imagen más difícil: {mejor_combinacion.peor_imagen}

📊 Imágenes procesadas: {len(mejor_combinacion.resultados_imagenes)}

💡 INTERPRETACIÓN:
• Score > 0.7: Excelente separación
• Score > 0.5: Buena separación
• Score > 0.3: Separación moderada

Esta combinación está lista para uso según requisitos de la pauta.
        """
        
        from tkinter import messagebox
        messagebox.showinfo(
            "Mejor Combinación K-Means",
            info_texto,
            parent=self.parent
        )
    
    def test_selected_kmeans(self):
        """
        Permite probar el K-Means elegido en una imagen específica del conjunto de test.
        """
        if not self.kmeans_analisis_completado or not self.kmeans_clasificador:
            self._show_error("Debe ejecutar el análisis K-Means primero")
            return
        
        mejor_combinacion = self.kmeans_clasificador.obtener_resultado_mejor_combinacion()
        if not mejor_combinacion:
            self._show_error("No hay mejor combinación disponible")
            return
        
        # Verificar que hay datos de test
        if not self.test_data:
            self._show_error("No hay datos de test cargados")
            return
        
        try:
            # Crear diálogo de selección de imagen
            self._show_image_selection_dialog(mejor_combinacion)
            
        except Exception as e:
            self._show_error(f"Error iniciando prueba K-Means: {e}")
    
    def _show_image_selection_dialog(self, mejor_combinacion):
        """Muestra diálogo para seleccionar imagen y probar K-Means."""
        import tkinter as tk
        from tkinter import ttk
        
        # Crear ventana de selección
        dialog = tk.Toplevel(self.parent)
        dialog.title("Probar K-Means - Seleccionar Imagen")
        dialog.geometry("500x400")
        dialog.configure(bg='#f0f0f0')
        dialog.transient(self.parent)
        dialog.grab_set()
        
        # Título
        title_label = tk.Label(
            dialog,
            text=f"Probar K-Means: {mejor_combinacion.nombre_combinacion}",
            font=('Segoe UI', 14, 'bold'),
            fg='#2c3e50',
            bg='#f0f0f0'
        )
        title_label.pack(pady=15)
        
        # Información de la mejor combinación
        info_text = f"""
Score: {mejor_combinacion.score_total:.3f}
Métricas: Silhouette {mejor_combinacion.metricas_promedio.get('silhouette', 0):.3f}
        """
        info_label = tk.Label(
            dialog,
            text=info_text,
            font=('Segoe UI', 10),
            fg='#34495e',
            bg='#f0f0f0'
        )
        info_label.pack(pady=5)
        
        # Lista de imágenes
        frame_list = tk.Frame(dialog, bg='#f0f0f0')
        frame_list.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        tk.Label(frame_list, text="Seleccionar imagen para probar:",
                font=('Segoe UI', 10, 'bold'), bg='#f0f0f0').pack(anchor='w')
        
        # Crear Listbox con scroll
        frame_scroll = tk.Frame(frame_list)
        frame_scroll.pack(fill=tk.BOTH, expand=True, pady=5)
        
        scrollbar = tk.Scrollbar(frame_scroll)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        listbox = tk.Listbox(frame_scroll, yscrollcommand=scrollbar.set,
                           font=('Courier New', 9), height=10)
        listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=listbox.yview)
        
        # Poblar con nombres de imágenes
        image_names = [img_data['nombre'] for img_data in self.test_data]
        for name in image_names:
            listbox.insert(tk.END, name)
        
        # Seleccionar primera imagen por defecto
        if image_names:
            listbox.selection_set(0)
        
        # Frame de botones
        button_frame = tk.Frame(dialog, bg='#f0f0f0')
        button_frame.pack(fill=tk.X, padx=20, pady=10)
        
        def on_test_click():
            selection = listbox.curselection()
            if not selection:
                self._show_error("Por favor seleccione una imagen")
                return
            
            selected_idx = selection[0]
            selected_image_data = self.test_data[selected_idx]
            
            dialog.destroy()
            self._execute_kmeans_test(selected_image_data, mejor_combinacion)
        
        def on_cancel_click():
            dialog.destroy()
        
        # Botones
        tk.Button(button_frame, text="Probar K-Means", 
                 command=on_test_click,
                 bg='#3498db', fg='white', font=('Segoe UI', 10, 'bold'),
                 padx=20, pady=5).pack(side=tk.LEFT, padx=5)
        
        tk.Button(button_frame, text="Cancelar", 
                 command=on_cancel_click,
                 bg='#95a5a6', fg='white', font=('Segoe UI', 10),
                 padx=20, pady=5).pack(side=tk.RIGHT, padx=5)
    
    def _execute_kmeans_test(self, image_data, mejor_combinacion):
        """Ejecuta K-Means en una imagen específica y muestra resultados."""
        def test_worker():
            try:
                self._update_status('working', f'Probando K-Means en {image_data["nombre"]}...')
                
                # Importar módulos necesarios
                from src.clasificadores.kmeans import SelectorCaracteristicas
                
                # Crear selector con la configuración de la mejor combinación
                selector = SelectorCaracteristicas(mejor_combinacion.configuracion)
                
                # Extraer características de la imagen
                caracteristicas = selector.extraer_caracteristicas(
                    image_data['imagen'], 
                    image_data.get('mascara', None)
                )
                
                # Aplicar K-Means con la mejor configuración encontrada
                from sklearn.cluster import KMeans
                from sklearn.metrics import silhouette_score
                import numpy as np
                
                # Usar el mejor número de clusters de la evaluación
                mejor_k = 2  # Por defecto
                if hasattr(mejor_combinacion, 'resultados_imagenes') and mejor_combinacion.resultados_imagenes:
                    # Encontrar el K más común en los resultados
                    clusters_counts = {}
                    for resultado in mejor_combinacion.resultados_imagenes:
                        k = resultado.n_clusters
                        clusters_counts[k] = clusters_counts.get(k, 0) + 1
                    mejor_k = max(clusters_counts.items(), key=lambda x: x[1])[0]
                
                # Expandir características para clustering
                caracteristicas_expandidas = self.kmeans_clasificador._expandir_caracteristicas_para_clustering(caracteristicas)
                
                # Aplicar K-Means
                kmeans = KMeans(n_clusters=mejor_k, random_state=42, n_init=10)
                etiquetas = kmeans.fit_predict(caracteristicas_expandidas)
                
                # Calcular métricas
                if len(np.unique(etiquetas)) > 1:
                    silhouette = silhouette_score(caracteristicas_expandidas, etiquetas)
                else:
                    silhouette = -1.0
                
                # Mostrar resultados en UI principal
                self.parent.after(0, lambda: self._show_single_kmeans_test_results(
                    image_data, mejor_combinacion, mejor_k, etiquetas, 
                    kmeans.cluster_centers_, silhouette, caracteristicas, caracteristicas_expandidas
                ))
                
                self._update_status('success', f'Prueba K-Means completada - Imagen: {image_data["nombre"]}')
                
            except Exception as e:
                self._update_status('error', f'Error en prueba K-Means: {str(e)[:50]}...')
                print(f"Error en prueba K-Means: {e}")
                import traceback
                traceback.print_exc()
        
        # Ejecutar en hilo separado
        import threading
        thread = threading.Thread(target=test_worker, daemon=True)
        thread.start()
    
    def _show_single_kmeans_test_results(self, image_data, mejor_combinacion, k_clusters, 
                                       etiquetas, centroids, silhouette_score, caracteristicas, caracteristicas_expandidas):
        """Muestra los resultados de la prueba K-Means en una imagen específica."""
        try:
            # Crear visualización personalizada para prueba individual
            self._show_individual_kmeans_visualization(
                image_data, mejor_combinacion, k_clusters, etiquetas, 
                centroids, silhouette_score, caracteristicas, caracteristicas_expandidas
            )
            
        except Exception as e:
            print(f"Error mostrando resultados de prueba: {e}")
            self._show_error(f"Error mostrando resultados: {e}")
    
    def _show_individual_kmeans_visualization(self, image_data, mejor_combinacion, 
                                            k_clusters, etiquetas, centroids, 
                                            silhouette_score, caracteristicas, caracteristicas_expandidas):
        """Crea visualización específica para prueba individual de K-Means."""
        # Importar matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        import numpy as np
        
        # Limpiar área de gráficos
        for widget in self.parent.graph_frame.inner_frame.winfo_children():
            widget.destroy()
        
        # Crear figura
        fig = Figure(figsize=(14, 10), facecolor='white')
        
        # Subplot 1: Imagen original
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.imshow(image_data['imagen'])
        ax1.set_title(f'Imagen Original: {image_data["nombre"]}', fontweight='bold')
        ax1.axis('off')
        
        # Subplot 2: Imagen segmentada K-Means
        ax2 = fig.add_subplot(2, 3, 2)
        try:
            # Generar imagen segmentada pixel por pixel usando K-Means
            imagen_segmentada = self._generar_imagen_segmentada_kmeans(
                image_data['imagen'], mejor_combinacion, k_clusters
            )
            ax2.imshow(imagen_segmentada, cmap='gray')
            ax2.set_title('Segmentación K-Means\n(Blanco=Lesión, Negro=Sano)', fontweight='bold')
            ax2.axis('off')
        except Exception as e:
            print(f"Error generando imagen segmentada: {e}")
            ax2.text(0.5, 0.5, f'Error generando\nsegmentación:\n{str(e)[:50]}...', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Error en Segmentación', fontweight='bold')
        
        # Subplot 3: Puntos expandidos coloreados por cluster
        ax3 = fig.add_subplot(2, 3, 3)
        if caracteristicas_expandidas.shape[1] >= 2:
            # Usar las primeras 2 dimensiones de los puntos expandidos
            scatter = ax3.scatter(caracteristicas_expandidas[:, 0], caracteristicas_expandidas[:, 1], 
                                c=etiquetas, cmap='viridis', alpha=0.7, s=50)
            ax3.set_title(f'Puntos K-Means (n={len(caracteristicas_expandidas)})', fontweight='bold')
            ax3.set_xlabel('Característica 1')
            ax3.set_ylabel('Característica 2')
            
            # Agregar centroides
            if centroids.shape[1] >= 2:
                ax3.scatter(centroids[:, 0], centroids[:, 1], 
                          c='red', marker='x', s=200, linewidths=3, label='Centroides')
                ax3.legend()
        else:
            ax3.text(0.5, 0.5, f'Puntos expandidos para clustering:\n{len(caracteristicas_expandidas)} puntos\n{caracteristicas_expandidas.shape[1]} dimensiones', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Puntos K-Means', fontweight='bold')
        
        # Subplot 4: Distribución de clusters
        ax4 = fig.add_subplot(2, 3, 4)
        unique_labels, counts = np.unique(etiquetas, return_counts=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        ax4.pie(counts, labels=[f'Cluster {i}' for i in unique_labels], 
               autopct='%1.1f%%', colors=colors)
        ax4.set_title(f'Distribución de Clusters (K={k_clusters})', fontweight='bold')
        
        # Subplot 5: Máscara real (si está disponible)
        ax5 = fig.add_subplot(2, 3, 5)
        if 'mascara' in image_data and image_data['mascara'] is not None:
            ax5.imshow(image_data['mascara'], cmap='gray')
            ax5.set_title('Máscara Real\n(Ground Truth)', fontweight='bold')
            ax5.axis('off')
        else:
            ax5.text(0.5, 0.5, 'Máscara real\nno disponible', 
                    ha='center', va='center', transform=ax5.transAxes)
            ax5.set_title('Sin Ground Truth', fontweight='bold')
        
        # Subplot 6: Información de resultados
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('off')
        
        result_text = f"""
RESULTADOS PRUEBA K-MEANS

Imagen: {image_data['nombre']}
Combinación: {mejor_combinacion.nombre_combinacion}
Score Original: {mejor_combinacion.score_total:.3f}

RESULTADOS ACTUALES:
• Clusters: {k_clusters}
• Silhouette Score: {silhouette_score:.3f}
• Puntos procesados: {len(etiquetas)}
• Características: {len(caracteristicas)}

INTERPRETACIÓN:
• Silhouette > 0.5: Excelente
• Silhouette > 0.2: Bueno  
• Silhouette > 0: Moderado
• Silhouette < 0: Pobre

Estado: {"Excelente" if silhouette_score > 0.5 else "Bueno" if silhouette_score > 0.2 else "Moderado" if silhouette_score > 0 else "Pobre"}

SEGMENTACIÓN:
• Píxel blanco: Cluster dominante (lesión)
• Píxel negro: Otros clusters (sano)
        """
        
        ax6.text(0.05, 0.95, result_text, transform=ax6.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3))
        
        fig.tight_layout(pad=2.0)
        
        # Mostrar en canvas
        canvas = FigureCanvasTkAgg(fig, master=self.parent.graph_frame.inner_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

    def _generar_imagen_segmentada_kmeans(self, imagen, mejor_combinacion, k_clusters: int):
        """
        Genera imagen segmentada pixel por pixel usando K-Means.
        
        Simula lo que hacen los clasificadores Bayesianos: clasifica cada píxel
        de la imagen original y crea una máscara binaria de segmentación.
        
        Args:
            imagen: Imagen RGB original (H,W,3)
            mejor_combinacion: Configuración de la mejor combinación K-Means
            k_clusters: Número de clusters a usar
            
        Returns:
            Imagen segmentada binaria (H,W) donde 1=lesión, 0=sano
        """
        import numpy as np
        from src.clasificadores.kmeans import SelectorCaracteristicas
        from sklearn.cluster import KMeans
        
        h, w, c = imagen.shape
        
        # Crear selector con la configuración de la mejor combinación
        selector = SelectorCaracteristicas(mejor_combinacion.configuracion)
        
        # Extraer características de cada píxel
        caracteristicas_pixeles = []
        
        # Procesar imagen píxel por píxel (muestreo cada N píxeles para eficiencia)
        step = max(1, min(h, w) // 50)  # Muestrear aprox 50x50 píxeles max
        
        pixel_positions = []
        for i in range(0, h, step):
            for j in range(0, w, step):
                # Crear ventana pequeña alrededor del píxel para extraer características
                ventana_size = 3
                i_start = max(0, i - ventana_size//2)
                i_end = min(h, i + ventana_size//2 + 1)
                j_start = max(0, j - ventana_size//2)
                j_end = min(w, j + ventana_size//2 + 1)
                
                ventana = imagen[i_start:i_end, j_start:j_end]
                
                # Extraer características de esta ventana
                try:
                    caract = selector.extraer_caracteristicas(ventana, None)
                    caracteristicas_pixeles.append(caract)
                    pixel_positions.append((i, j))
                except:
                    # Si falla la extracción, usar valores RGB promedio
                    rgb_promedio = np.mean(ventana.reshape(-1, 3), axis=0)
                    caracteristicas_pixeles.append(rgb_promedio)
                    pixel_positions.append((i, j))
        
        if not caracteristicas_pixeles:
            return np.zeros((h, w))
        
        # Convertir a array
        caracteristicas_array = np.array(caracteristicas_pixeles)
        
        # Aplicar K-Means
        kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
        etiquetas_pixeles = kmeans.fit_predict(caracteristicas_array)
        
        # Determinar qué cluster representa "lesión"
        # Usar el cluster con mayor promedio de intensidad en canal rojo
        # (asumiendo que las lesiones tienden a ser más rojizas)
        centroids = kmeans.cluster_centers_
        
        # Si tenemos características RGB en los primeros 3 componentes
        if centroids.shape[1] >= 3:
            # Usar intensidad roja como criterio
            cluster_lesion = np.argmax(centroids[:, 0])  # Canal rojo
        else:
            # Usar cluster con mayor valor promedio
            cluster_lesion = np.argmax(np.mean(centroids, axis=1))
        
        # Crear imagen segmentada
        imagen_segmentada = np.zeros((h, w))
        
        # Interpolar resultados a toda la imagen
        for idx, (i, j) in enumerate(pixel_positions):
            etiqueta = etiquetas_pixeles[idx]
            # Marcar como lesión si pertenece al cluster dominante
            valor = 1.0 if etiqueta == cluster_lesion else 0.0
            
            # Asignar valor a región alrededor del píxel
            for di in range(-step//2, step//2 + 1):
                for dj in range(-step//2, step//2 + 1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w:
                        imagen_segmentada[ni, nj] = valor
        
        return imagen_segmentada


__all__ = ['ClassifierController']