"""
Controladores para la lógica de aplicación.

Este módulo implementa el patrón MVC separando la lógica de negocio
de la presentación en la interfaz gráfica.
"""

import threading
from typing import Dict, List, Any, Optional, Callable
import tkinter as tk
from ..dialogs import EvaluationDialog, ComparisonDialog


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
    
    def train_classifier(self, criterio_umbral: str):
        """
        Entrena el clasificador en un hilo separado.
        
        Args:
            criterio_umbral: Criterio para selección de umbral
        """
        def train_worker():
            try:
                self._update_status('working', 'Cargando datos...')
                
                # Cargar datos si no están disponibles
                if not self.train_data:
                    self._load_data()
                
                self._update_status('working', 'Entrenando clasificador...')
                
                # Importar y crear clasificador
                from src.clasificadores.bayesiano import ClasificadorBayesianoRGB
                self.clasificador = ClasificadorBayesianoRGB(criterio_umbral=criterio_umbral)
                
                # Entrenar
                self.clasificador.entrenar(self.train_data)
                
                # Verificar entrenamiento
                if not self.clasificador.entrenado:
                    raise RuntimeError("El entrenamiento del clasificador falló")
                
                self.entrenado = True
                parametros = self.clasificador.obtener_parametros()
                
                # Actualizar UI con éxito
                self._update_status('success', 'Entrenamiento completado')
                self._update_progress(
                    f"✅ Entrenado\\nCriterio: {criterio_umbral}\\nUmbral: {parametros['umbral']:.4f}"
                )
                
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
    
    def _show_error(self, mensaje: str):
        """Muestra un mensaje de error."""
        from tkinter import messagebox
        messagebox.showerror("Error", mensaje)


__all__ = ['ClassifierController']