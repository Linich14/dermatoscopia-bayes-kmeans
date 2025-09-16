"""
Ventana principal refactorizada con arquitectura modular.

Esta implementación utiliza el patrón MVC y componentes modulares
para una mejor organización y mantenibilidad del código.

COMPONENTES PRINCIPALES PARA LOCALIZAR:
- _create_classifier_section(): Línea ~184 - Crea controles del clasificador Bayesiano + PCA
- _train_classifier(): Línea ~326 - Maneja entrenamiento según tipo seleccionado
- _compare_rgb_vs_pca(): Línea ~338 - Inicia comparación RGB vs PCA
- _toggle_pca_controls(): Línea ~348 - Muestra/oculta controles específicos de PCA
- _show_pca_justification(): Línea ~345 - Muestra justificación metodológica

ARQUITECTURA:
- Ventana usa patrón MVC con ClassifierController
- Controles modulares y reutilizables
- Comunicación via callbacks para actualizar estado
- Hilos separados para operaciones largas

FLUJO DE USUARIO:
1. Seleccionar área y canal para histogramas
2. Elegir tipo de clasificador (RGB vs PCA)  
3. Configurar criterios (umbral, PCA)
4. Entrenar modelo
5. Evaluar y comparar resultados
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
        self.usar_pca_var = tk.BooleanVar(value=False)
        self.criterio_pca_var = tk.StringVar(value='varianza')
        
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
                        text="Análisis de Imágenes Dermatoscópicas",
                        fg='white',
                        bg=COLORS['primary'],
                        font=('Segoe UI', 16, 'bold'))
        title.place(relx=0.5, rely=0.5, anchor='center')
    
    def _create_sidebar(self, parent):
        """Crea el panel lateral con controles."""
        # Contenedor principal del sidebar
        sidebar_container = tk.Frame(parent, **STYLES['sidebar'])
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
                              width=DESIGN['content_width'] - 40,  # -40 para el padding
                              height=DESIGN['button_height'],
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
                              width=DESIGN['content_width'] - 40,  # -40 para el padding
                              height=DESIGN['button_height'],
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
        """
        *** SECCIÓN PRINCIPAL DEL CLASIFICADOR ***
        Localización: Línea ~184 del archivo ventana_modular.py
        
        PROPÓSITO: Crea todos los controles para configurar y usar el clasificador
        
        CONTROLES INCLUIDOS:
        1. Checkbox para activar/desactivar PCA
        2. Selector de criterio PCA (varianza, codo, discriminativo)
        3. Botón para ver justificación metodológica PCA
        4. Selector de criterio de umbral (Youden, EER, etc.)
        5. Botones de acción: entrenar, evaluar, comparar, clasificar
        
        FUNCIONALIDAD DINÁMICA:
        - Controles PCA se muestran/ocultan según checkbox
        - Botones se activan/desactivan según estado del entrenamiento
        - Indicadores visuales de progreso
        
        RESULTADO: Panel completo para manejo del clasificador Bayesiano + PCA
        
        Crea la sección del clasificador Bayesiano."""
        section_frame = tk.LabelFrame(parent,
                                     text="Clasificador Bayesiano RGB + PCA",
                                     bg=COLORS['background'],
                                     fg=COLORS['text'],
                                     font=('Segoe UI', 10, 'bold'))
        section_frame.pack(fill=tk.X, padx=5, pady=10)
        
        # Checkbox para usar PCA
        pca_frame = tk.Frame(section_frame, bg=COLORS['background'])
        pca_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        
        pca_check = tk.Checkbutton(pca_frame,
                                  text="🔬 Usar PCA (Reducción Dimensional)",
                                  variable=self.usar_pca_var,
                                  command=self._toggle_pca_controls,
                                  font=('Segoe UI', 9, 'bold'),
                                  fg=COLORS['primary'],
                                  bg=COLORS['background'],
                                  selectcolor=COLORS['background'])
        pca_check.pack(anchor='w')
        
        # Frame para controles PCA (inicialmente oculto)
        self.pca_controls_frame = tk.Frame(section_frame, bg=COLORS['accent_light'])
        
        # Criterio PCA
        tk.Label(self.pca_controls_frame,
                text="Criterio selección componentes:",
                font=('Segoe UI', 8),
                fg=COLORS['text'],
                bg=COLORS['accent_light']).pack(fill=tk.X, padx=10, pady=(5, 2))
        
        criterio_pca_combo = ttk.Combobox(self.pca_controls_frame,
                                         textvariable=self.criterio_pca_var,
                                         values=['varianza', 'codo', 'discriminativo'],
                                         state='readonly',
                                         font=('Segoe UI', 8))
        criterio_pca_combo.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        # Botón de justificación PCA
        justificar_btn = RoundedButton(self.pca_controls_frame,
                                     text="📄 Ver Justificación PCA",
                                     command=self._show_pca_justification,
                                     background=COLORS['info'],
                                     width=160,
                                     height=25)
        justificar_btn.pack(pady=2, padx=10)
        
        # Selector de criterio de umbral
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
            ("⚖️ Comparar Criterios", self._compare_criteria, COLORS['secondary']),
            ("🆚 RGB vs PCA", self._compare_rgb_vs_pca, COLORS['warning']),
            ("🖼️ Clasificar Imagen", self._classify_image, COLORS['success'])
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
        self.usar_pca_var.set(False)
        self.criterio_pca_var.set('varianza')
        
        # Ocultar controles PCA
        self._toggle_pca_controls()
        
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
        usar_pca = self.usar_pca_var.get()
        criterio_pca = self.criterio_pca_var.get()
        
        self.classifier_controller.train_classifier(
            criterio, 
            usar_pca=usar_pca, 
            criterio_pca=criterio_pca
        )
    
    def _compare_rgb_vs_pca(self):
        """Compara rendimiento RGB vs PCA."""
        criterio = self.criterio_var.get()
        criterio_pca = self.criterio_pca_var.get()
        self.classifier_controller.compare_rgb_vs_pca(criterio, criterio_pca)
    
    def _show_pca_justification(self):
        """Muestra la justificación metodológica del PCA."""
        self.classifier_controller.show_pca_justification()
    
    def _toggle_pca_controls(self):
        """Muestra/oculta controles específicos de PCA."""
        if self.usar_pca_var.get():
            self.pca_controls_frame.pack(fill=tk.X, padx=5, pady=5)
        else:
            self.pca_controls_frame.pack_forget()
    
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