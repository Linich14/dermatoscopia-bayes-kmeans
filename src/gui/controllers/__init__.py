"""
Controladores para la lógica de aplicación.

Este módulo implementa el patrón MVC separando la lógica de negocio
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


__all__ = ['ClassifierController']