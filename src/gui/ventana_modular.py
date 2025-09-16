"""
Ventana principal refactorizada con arquitectura modular.

Esta implementación utiliza el patrón MVC y componentes modulares
para una mejor organización y mantenibilidad del código.
"""

import tkinter as tk
from tkinter import ttk

from .styles import COLORS, STYLES, DESIGN
from .components import RoundedContainer, RoundedButton, ScrollableFrame, StatusIndicator
from .controllers import ClassifierController
from .graficos import mostrar_histograma


class VentanaPrincipalModular(tk.Tk):
    """
    Ventana principal refactorizada con arquitectura modular.
    
    Utiliza el patrón MVC separando la lógica de presentación,
    control y datos en componentes especializados.
    """
    
    def __init__(self, stats_rgb):
        """
        Inicializa la ventana principal.
        
        Args:
            stats_rgb: Estadísticas RGB precalculadas
        """
        super().__init__()
        
        # Configuración básica
        self.title("Análisis Dermatoscópico")
        self.geometry("1280x900")
        self.configure(**STYLES['root'])
        
        # Datos
        self.stats_rgb = stats_rgb
        
        # Variables de control
        self.area_var = tk.StringVar(value='lesion')
        self.canal_var = tk.StringVar(value='R')
        self.criterio_var = tk.StringVar(value='youden')
        
        # Controlador
        self.classifier_controller = ClassifierController(self)
        self._setup_controller_callbacks()
        
        # Construir interfaz
        self._create_interface()
        
        # Actualizar histograma inicial
        self.after(100, self.update_histogram)
    
    def _setup_controller_callbacks(self):
        """Configura los callbacks del controlador para actualizar la UI."""
        self.classifier_controller.set_status_callback(self._update_classifier_status)
        self.classifier_controller.set_progress_callback(self._update_progress_display)
    
    def _create_interface(self):
        """Crea la estructura principal de la interfaz."""
        # Header
        self._create_header()
        
        # Contenedor principal
        main_container = tk.Frame(self, **STYLES['root'])
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Sidebar
        self._create_sidebar(main_container)
        
        # Área principal
        self._create_main_area(main_container)
    
    def _create_header(self):
        """Crea la barra superior."""
        header = tk.Frame(self, **STYLES['header'])
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        
        title = tk.Label(header,
                        text="🔬 Análisis de Imágenes Dermatoscópicas - Versión Modular",
                        fg='white',
                        bg=COLORS['primary'],
                        font=('Segoe UI', 16, 'bold'))
        title.place(relx=0.5, rely=0.5, anchor='center')
    
    def _create_sidebar(self, parent):
        """Crea el panel lateral con controles."""
        # Contenedor principal del sidebar
        sidebar_container = tk.Frame(parent, bg=COLORS['background'], width=220)
        sidebar_container.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        sidebar_container.pack_propagate(False)
        
        # Frame scrollable
        self.sidebar_scroll = ScrollableFrame(sidebar_container)
        self.sidebar_scroll.pack(fill=tk.BOTH, expand=True)
        
        content = self.sidebar_scroll.scrollable_frame
        content.configure(bg=COLORS['background'])
        
        # Secciones del sidebar
        self._create_analysis_section(content)
        self._create_channel_section(content)
        self._create_actions_section(content)
        self._create_classifier_section(content)
    
    def _create_analysis_section(self, parent):
        """Crea la sección de selección de área."""
        section_frame = tk.LabelFrame(parent,
                                     text="Área de análisis",
                                     bg=COLORS['background'],
                                     fg=COLORS['text'],
                                     font=('Segoe UI', 10, 'bold'))
        section_frame.pack(fill=tk.X, padx=5, pady=10)
        
        # Botones de área
        areas = [
            ('sana', 'Piel Sana ✅', COLORS['success']),
            ('lesion', 'Lesión ⚠️', COLORS['primary'])
        ]
        
        for value, text, color in areas:
            btn = RoundedButton(section_frame,
                              text=text,
                              command=lambda v=value: self._select_area(v),
                              background=color,
                              activebackground=COLORS['primary_dark'],
                              width=180,
                              height=35,
                              variable=self.area_var,
                              value=value)
            btn.pack(pady=5, padx=10)
    
    def _create_channel_section(self, parent):
        """Crea la sección de selección de canal."""
        section_frame = tk.LabelFrame(parent,
                                     text="Canal de color",
                                     bg=COLORS['background'],
                                     fg=COLORS['text'],
                                     font=('Segoe UI', 10, 'bold'))
        section_frame.pack(fill=tk.X, padx=5, pady=10)
        
        # Botones de canal
        channels = [
            ('R', '🔴 Rojo', COLORS['primary']),
            ('G', '🟢 Verde', COLORS['success']),
            ('B', '🔵 Azul', COLORS['info'])
        ]
        
        for value, text, color in channels:
            btn = RoundedButton(section_frame,
                              text=text,
                              command=lambda v=value: self._select_channel(v),
                              background=color,
                              activebackground=COLORS['primary_dark'],
                              width=180,
                              height=35,
                              variable=self.canal_var,
                              value=value)
            btn.pack(pady=5, padx=10)
    
    def _create_actions_section(self, parent):
        """Crea la sección de acciones generales."""
        section_frame = tk.LabelFrame(parent,
                                     text="Acciones",
                                     bg=COLORS['background'],
                                     fg=COLORS['text'],
                                     font=('Segoe UI', 10, 'bold'))
        section_frame.pack(fill=tk.X, padx=5, pady=10)
        
        reset_btn = RoundedButton(section_frame,
                                text="🔄 Reiniciar",
                                command=self._reset_values,
                                background=COLORS['secondary'],
                                width=180,
                                height=35)
        reset_btn.pack(pady=5, padx=10)
    
    def _create_classifier_section(self, parent):
        """Crea la sección del clasificador Bayesiano."""
        section_frame = tk.LabelFrame(parent,
                                     text="Clasificador Bayesiano RGB",
                                     bg=COLORS['background'],
                                     fg=COLORS['text'],
                                     font=('Segoe UI', 10, 'bold'))
        section_frame.pack(fill=tk.X, padx=5, pady=10)
        
        # Selector de criterio
        tk.Label(section_frame,
                text="Criterio de umbral:",
                font=('Segoe UI', 9),
                fg=COLORS['text'],
                bg=COLORS['background']).pack(fill=tk.X, padx=10, pady=(10, 5))
        
        criterio_combo = ttk.Combobox(section_frame,
                                     textvariable=self.criterio_var,
                                     values=['youden', 'equal_error', 'prior_balanced'],
                                     state='readonly',
                                     font=('Segoe UI', 9))
        criterio_combo.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Indicador de estado
        self.status_indicator = StatusIndicator(section_frame,
                                              bg=COLORS['background'])
        self.status_indicator.pack(fill=tk.X, padx=10, pady=5)
        
        # Texto de progreso
        self.progress_text = tk.Label(section_frame,
                                    text="Listo para entrenar",
                                    font=('Segoe UI', 9),
                                    fg=COLORS['text'],
                                    bg=COLORS['background'],
                                    wraplength=180)
        self.progress_text.pack(fill=tk.X, padx=10, pady=5)
        
        # Botones de acción
        classifier_actions = [
            ("🤖 Entrenar", self._train_classifier, COLORS['accent']),
            ("📊 Evaluar", self._evaluate_classifier, COLORS['info']),
            ("⚖️ Comparar", self._compare_criteria, COLORS['secondary']),
            ("🖼️ Clasificar", self._classify_image, COLORS['warning'])
        ]
        
        for text, command, color in classifier_actions:
            btn = RoundedButton(section_frame,
                              text=text,
                              command=command,
                              background=color,
                              activebackground=COLORS['primary_dark'],
                              width=180,
                              height=30)
            btn.pack(pady=3, padx=10)
    
    def _create_main_area(self, parent):
        """Crea el área principal para visualización."""
        main_container = RoundedContainer(parent, **STYLES['root'])
        main_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Tarjeta para el gráfico
        self.graph_card = RoundedContainer(main_container.inner_frame, **STYLES['card'])
        self.graph_card.pack(fill=tk.BOTH, expand=True)
        
        # Frame para el gráfico
        self.graph_frame = RoundedContainer(self.graph_card.inner_frame, 
                                          background=COLORS['background'])
        self.graph_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def _select_area(self, area):
        """Selecciona el área de análisis."""
        self.area_var.set(area)
        self.update_histogram()
    
    def _select_channel(self, channel):
        """Selecciona el canal de color."""
        self.canal_var.set(channel)
        self.update_histogram()
    
    def _reset_values(self):
        """Reinicia todos los valores."""
        self.area_var.set('lesion')
        self.canal_var.set('R')
        self.criterio_var.set('youden')
        
        # Reiniciar controlador
        self.classifier_controller.clasificador = None
        self.classifier_controller.entrenado = False
        
        # Actualizar UI
        self.status_indicator.set_state('idle', 'Listo')
        self.progress_text.configure(text="Listo para entrenar")
        self.update_histogram()
        
        print("🔄 Sistema reiniciado correctamente")
    
    def _train_classifier(self):
        """Inicia el entrenamiento del clasificador."""
        criterio = self.criterio_var.get()
        self.classifier_controller.train_classifier(criterio)
    
    def _evaluate_classifier(self):
        """Evalúa el clasificador."""
        self.classifier_controller.evaluate_classifier()
    
    def _compare_criteria(self):
        """Compara criterios de umbral."""
        self.classifier_controller.compare_criteria()
    
    def _classify_image(self):
        """Clasifica una imagen seleccionada."""
        self.classifier_controller.classify_image()
    
    def _update_classifier_status(self, state: str, message: str):
        """Actualiza el estado del clasificador en la UI."""
        self.status_indicator.set_state(state, message)
    
    def _update_progress_display(self, message: str):
        """Actualiza el texto de progreso."""
        self.progress_text.configure(text=message)
    
    def update_histogram(self):
        """Actualiza el histograma usando la función modular de graficos.py."""
        try:
            # Obtener datos actuales
            area = self.area_var.get()
            canal = self.canal_var.get()
            datos = self.stats_rgb[area][canal]
            
            # Usar la función modular existente
            mostrar_histograma(self.graph_frame.inner_frame, datos, area, canal)
            
        except Exception as e:
            print(f"Error actualizando histograma: {e}")
            # Limpiar el frame en caso de error
            for widget in self.graph_frame.inner_frame.winfo_children():
                widget.destroy()
            # Mostrar mensaje de error
            error_label = tk.Label(self.graph_frame.inner_frame,
                                 text="Error al generar histograma",
                                 font=('Segoe UI', 12),
                                 fg=COLORS['text'],
                                 bg=COLORS['background'])
            error_label.pack(pady=50)


# Alias para compatibilidad
VentanaPrincipal = VentanaPrincipalModular