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
import threading

from .styles import COLORS, STYLES, DESIGN
from .components import RoundedContainer, RoundedButton, ScrollableFrame, StatusIndicator
from .controllers import ClassifierController
from .graficos import mostrar_histograma
from .dialogs import mostrar_dialogo_comparacion
from ..comparadores.comparador_triple import ComparadorTriple, ejecutar_comparacion_rapida


class VentanaPrincipalModular(tk.Tk):
    """
    Ventana principal refactorizada con arquitectura modular.
    
    Utiliza el patrón MVC separando la lógica de presentación,
    control y datos en componentes especializados.
    """
    
    def __init__(self, stats_rgb, datos_train=None, datos_val=None, datos_test=None):
        """
        Inicializa la ventana principal.
        
        Args:
            stats_rgb: Estadísticas RGB precalculadas
            datos_train: Datos de entrenamiento (opcional, para comparador triple)
            datos_val: Datos de validación (opcional, para comparador triple)
            datos_test: Datos de test (opcional, para comparador triple)
        """
        super().__init__()
        
        # Configuración básica
        self.title("Centro Médico Ñielol - Sistema de Análisis Dermatoscópico")
        self.geometry("1280x900")
        self.configure(**STYLES['root'])
        
        # Datos
        self.stats_rgb = stats_rgb
        self.datos_train = datos_train
        self.datos_val = datos_val
        self.datos_test = datos_test
        
        # Variables de control
        self.area_var = tk.StringVar(value='lesion')
        self.canal_var = tk.StringVar(value='R')
        self.criterio_var = tk.StringVar(value='youden')
        self.usar_pca_var = tk.BooleanVar(value=False)
        self.criterio_pca_var = tk.StringVar(value='varianza')
        
        # Variables de control K-Means
        self.kmeans_caracteristicas_var = tk.StringVar(value='RGB_basico')
        self.kmeans_clusters_var = tk.StringVar(value='2,3,4,5')
        self.kmeans_auto_eval_var = tk.BooleanVar(value=True)
        
        # Variables de control Comparador Triple
        self.comparador_paralelo_var = tk.BooleanVar(value=True)
        self.comparador_metricas_detalladas_var = tk.BooleanVar(value=True)
        self.comparador_triple = None
        self.ultimo_reporte_triple = None
        
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
        
        # Título principal centrado
        title = tk.Label(header,
                        text="Centro Médico Ñielol - Diagnóstico Dermatológico Asistido",
                        fg='white',
                        bg=COLORS['primary'],
                        font=('Segoe UI', 16, 'bold'))
        title.place(relx=0.5, rely=0.5, anchor='center')
        
        # Subtítulo en la esquina derecha
        subtitle = tk.Label(header,
                          text="Unidad de Dermatología",
                          fg='white',
                          bg=COLORS['primary'],
                          font=('Segoe UI', 10))
        subtitle.place(relx=0.98, rely=0.7, anchor='e')
        
        # Información adicional en la esquina izquierda
        info_label = tk.Label(header,
                            text="🏥 Sistema de Apoyo Diagnóstico",
                            fg='white',
                            bg=COLORS['primary'],
                            font=('Segoe UI', 10))
        info_label.place(relx=0.02, rely=0.7, anchor='w')
    
    def _create_sidebar(self, parent):
        """Crea el panel lateral con controles."""
        # Contenedor principal del sidebar
        sidebar_container = tk.Frame(parent, **STYLES['sidebar'])
        sidebar_container.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        sidebar_container.configure(width=DESIGN['sidebar_width'])
        
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
        self._create_kmeans_section(content)
        self._create_comparador_triple_section(content)
    
    def _create_analysis_section(self, parent):
        """Crea la sección de selección de área."""
        section_frame = tk.LabelFrame(parent,
                                     text="Área de análisis",
                                     bg=COLORS['background'],
                                     fg=COLORS['text'],
                                     font=('Segoe UI', 10, 'bold'),
                                     relief='solid',
                                     borderwidth=1,
                                     highlightbackground=COLORS['border'])
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
                              width=320,  # Reducido para dar margen
                              height=DESIGN['button_height'],
                              variable=self.area_var,
                              value=value)
            btn.pack(pady=5, padx=20)
    
    def _create_channel_section(self, parent):
        """Crea la sección de selección de canal."""
        section_frame = tk.LabelFrame(parent,
                                     text="Canal de color",
                                     bg=COLORS['background'],
                                     fg=COLORS['text'],
                                     font=('Segoe UI', 10, 'bold'),
                                     relief='solid',
                                     borderwidth=1,
                                     highlightbackground=COLORS['border'])
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
                              width=320,  # Reducido para dar margen
                              height=DESIGN['button_height'],
                              variable=self.canal_var,
                              value=value)
            btn.pack(pady=5, padx=20)
    
    def _create_actions_section(self, parent):
        """Crea la sección de acciones generales."""
        section_frame = tk.LabelFrame(parent,
                                     text="Acciones",
                                     bg=COLORS['background'],
                                     fg=COLORS['text'],
                                     font=('Segoe UI', 10, 'bold'),
                                     relief='solid',
                                     borderwidth=1,
                                     highlightbackground=COLORS['border'])
        section_frame.pack(fill=tk.X, padx=5, pady=10)
        
        reset_btn = RoundedButton(section_frame,
                                text="🔄 Reiniciar",
                                command=self._reset_values,
                                background=COLORS['secondary'],
                                width=320,
                                height=35)
        reset_btn.pack(pady=5, padx=20)
    
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
                                     text="Sistema de Diagnóstico Bayesiano",
                                     bg=COLORS['background'],
                                     fg=COLORS['text'],
                                     font=('Segoe UI', 10, 'bold'),
                                     relief='solid',
                                     borderwidth=1,
                                     highlightbackground=COLORS['border'])
        section_frame.pack(fill=tk.X, padx=5, pady=10)
        
        # Checkbox para usar PCA
        pca_frame = tk.Frame(section_frame, bg=COLORS['background'])
        pca_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        
        pca_check = tk.Checkbutton(pca_frame,
                                  text="🔬 Análisis de Componentes Principales (PCA)",
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
                bg=COLORS['accent_light']).pack(fill=tk.X, padx=20, pady=(5, 2))
        
        criterio_pca_combo = ttk.Combobox(self.pca_controls_frame,
                                         textvariable=self.criterio_pca_var,
                                         values=['varianza', 'codo', 'discriminativo'],
                                         state='readonly',
                                         font=('Segoe UI', 8),
                                         width=20)
        criterio_pca_combo.pack(padx=20, pady=(0, 5))
        
        # Botón de justificación PCA
        justificar_btn = RoundedButton(self.pca_controls_frame,
                                     text="📄 Información Metodológica",
                                     command=self._show_pca_justification,
                                     background=COLORS['info'],
                                     width=300,
                                     height=25)
        justificar_btn.pack(pady=2, padx=20)
        
        # Selector de criterio de umbral
        tk.Label(section_frame,
                text="Criterio diagnóstico:",
                font=('Segoe UI', 9),
                fg=COLORS['text'],
                bg=COLORS['background']).pack(fill=tk.X, padx=20, pady=(10, 5))
        
        criterio_combo = ttk.Combobox(section_frame,
                                     textvariable=self.criterio_var,
                                     values=['youden', 'equal_error', 'prior_balanced'],
                                     state='readonly',
                                     font=('Segoe UI', 9),
                                     width=25)
        criterio_combo.pack(padx=20, pady=(0, 10))
        
        # Indicador de estado
        self.status_indicator = StatusIndicator(section_frame,
                                              bg=COLORS['background'])
        self.status_indicator.pack(fill=tk.X, padx=20, pady=5)
        
        # Texto de progreso
        self.progress_text = tk.Label(section_frame,
                                    text="Sistema listo para entrenamiento",
                                    font=('Segoe UI', 9),
                                    fg=COLORS['text'],
                                    bg=COLORS['background'],
                                    wraplength=180)
        self.progress_text.pack(fill=tk.X, padx=20, pady=5)
        
        # Botones de acción
        classifier_actions = [
            ("🤖 Entrenar", self._train_classifier, COLORS['accent']),
            ("📊 Evaluar", self._evaluate_classifier, COLORS['info']),
            ("📈 Análisis ROC", self._analyze_roc, COLORS['primary']),
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
                              width=320,
                              height=30)
            btn.pack(pady=3, padx=20)
    
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
        
        # Reiniciar variables K-Means
        self.kmeans_caracteristicas_var.set('RGB_basico')
        self.kmeans_clusters_var.set('2,3,4,5')
        self.kmeans_auto_eval_var.set(True)
        
        # Reiniciar variables Comparador Triple
        self.comparador_paralelo_var.set(True)
        self.comparador_metricas_detalladas_var.set(True)
        self.comparador_triple = None
        self.ultimo_reporte_triple = None
        
        # Ocultar controles PCA
        self._toggle_pca_controls()
        
        # Reiniciar controlador
        self.classifier_controller.clasificador = None
        self.classifier_controller.entrenado = False
        
        # Actualizar UI
        self.status_indicator.set_state('idle', 'Listo')
        self.progress_text.configure(text="Sistema listo para entrenamiento")
        
        # Actualizar estado del comparador
        if hasattr(self, 'comparador_status'):
            self.comparador_status.set_state('idle', 'Listo')
        if hasattr(self, 'comparador_progress_text'):
            self.comparador_progress_text.configure(text="Sistema listo para análisis comparativo")
        
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
    
    def _analyze_roc(self):
        """Ejecuta análisis ROC completo."""
        self.classifier_controller.analyze_roc()
    
    def _compare_criteria(self):
        """Compara criterios de umbral."""
        self.classifier_controller.compare_criteria()
    
    def _classify_image(self):
        """Clasifica una imagen seleccionada."""
        self.classifier_controller.classify_image()
    
    def _create_kmeans_section(self, parent):
        """
        *** SECCIÓN K-MEANS CLASIFICACIÓN NO SUPERVISADA ***
        
        Crea controles para configurar y ejecutar análisis K-Means según
        los requisitos del proyecto.
        
        FUNCIONALIDAD:
        - Selección de combinación de características
        - Configuración de número de clusters
        - Evaluación automática de combinaciones
        - Ejecución de análisis completo
        - Visualización de resultados
        """
        section_frame = tk.LabelFrame(parent,
                                     text="🎯 K-Means (Clasificación No Supervisada)",
                                     bg=COLORS['background'],
                                     fg=COLORS['text'],
                                     font=('Segoe UI', 10, 'bold'),
                                     relief='solid',
                                     borderwidth=1,
                                     highlightbackground=COLORS['border'])
        section_frame.pack(fill=tk.X, padx=5, pady=10)
        

        
        # Selector de tipo de características
        tk.Label(section_frame,
                text="Combinación de características:",
                font=('Segoe UI', 9),
                fg=COLORS['text'],
                bg=COLORS['background']).pack(fill=tk.X, padx=20, pady=(10, 2))
        
        caracteristicas_combo = ttk.Combobox(section_frame,
                                           textvariable=self.kmeans_caracteristicas_var,
                                           values=[
                                               'RGB_basico',
                                               'HSV_Textura', 
                                               'LAB_RGB',
                                               'Completo',
                                               'Textura_Avanzada',
                                               'Auto_Evaluar_Todas'
                                           ],
                                           state='readonly',
                                           font=('Segoe UI', 8),
                                           width=20)
        caracteristicas_combo.pack(padx=20, pady=(0, 5))
        
        # Configuración de clusters
        tk.Label(section_frame,
                text="Números de clusters a probar:",
                font=('Segoe UI', 9),
                fg=COLORS['text'],
                bg=COLORS['background']).pack(fill=tk.X, padx=20, pady=(5, 2))
        
        clusters_entry = tk.Entry(section_frame,
                                 textvariable=self.kmeans_clusters_var,
                                 font=('Segoe UI', 8),
                                 bg=COLORS['card_bg'],
                                 fg=COLORS['text'],
                                 width=15)
        clusters_entry.pack(padx=20, pady=(0, 5))
        
        # Checkbox para evaluación automática
        auto_check = tk.Checkbutton(section_frame,
                                   text="🔍 Evaluación automática (recomendado)",
                                   variable=self.kmeans_auto_eval_var,
                                   font=('Segoe UI', 8),
                                   fg=COLORS['text'],
                                   bg=COLORS['background'],
                                   selectcolor=COLORS['background'])
        auto_check.pack(fill=tk.X, padx=20, pady=5)
        
        # Botones de acción K-Means
        kmeans_actions = [
            ("Ejecutar K-Means", self._execute_kmeans, COLORS['warning']),
            ("Mejor Combinacion", self._show_best_combination, COLORS['success']),
            ("Probar K-Means Elegido", self._test_selected_kmeans, COLORS['primary'])
        ]
        
        for text, command, color in kmeans_actions:
            btn = RoundedButton(section_frame,
                              text=text,
                              command=command,
                              background=color,
                              activebackground=COLORS['primary_dark'],
                              width=320,
                              height=30)
            btn.pack(pady=3, padx=20)
    
    def _create_comparador_triple_section(self, parent):
        """
        *** SECCIÓN COMPARADOR TRIPLE ***
        
        Crea controles para ejecutar comparación simultánea de los tres
        clasificadores: RGB Bayesiano, PCA Bayesiano y K-Means.
        
        FUNCIONALIDAD:
        - Configuración de ejecución paralela
        - Opciones de métricas detalladas
        - Comparación completa y rápida
        - Visualización de resultados comparativos
        - Ranking automático de clasificadores
        """
        section_frame = tk.LabelFrame(parent,
                                     text="⚖️ Comparador Triple (RGB + PCA + K-Means)",
                                     bg=COLORS['background'],
                                     fg=COLORS['text'],
                                     font=('Segoe UI', 10, 'bold'),
                                     relief='solid',
                                     borderwidth=1,
                                     highlightbackground=COLORS['border'])
        section_frame.pack(fill=tk.X, padx=5, pady=10)
        
        # Descripción del comparador
        desc_label = tk.Label(section_frame,
                             text="Compare los tres clasificadores simultáneamente\npara evaluación integral de rendimiento",
                             font=('Segoe UI', 8),
                             fg=COLORS['text'],
                             bg=COLORS['background'],
                             justify='center',
                             wraplength=160)
        desc_label.pack(fill=tk.X, padx=20, pady=(10, 5))
        
        # Opciones de configuración
        config_frame = tk.Frame(section_frame, bg=COLORS['accent_light'])
        config_frame.pack(fill=tk.X, padx=15, pady=5)
        
        # Checkbox para ejecución paralela
        paralelo_check = tk.Checkbutton(config_frame,
                                       text="🚀 Ejecución en paralelo",
                                       variable=self.comparador_paralelo_var,
                                       font=('Segoe UI', 8),
                                       fg=COLORS['text'],
                                       bg=COLORS['accent_light'],
                                       selectcolor=COLORS['accent_light'])
        paralelo_check.pack(fill=tk.X, padx=15, pady=2)
        
        # Checkbox para métricas detalladas
        metricas_check = tk.Checkbutton(config_frame,
                                       text="📊 Métricas detalladas",
                                       variable=self.comparador_metricas_detalladas_var,
                                       font=('Segoe UI', 8),
                                       fg=COLORS['text'],
                                       bg=COLORS['accent_light'],
                                       selectcolor=COLORS['accent_light'])
        metricas_check.pack(fill=tk.X, padx=15, pady=2)
        
        # Indicador de progreso para comparador triple
        self.comparador_status = StatusIndicator(section_frame, bg=COLORS['background'])
        self.comparador_status.pack(fill=tk.X, padx=20, pady=5)
        
        # Texto de progreso específico para comparador
        self.comparador_progress_text = tk.Label(section_frame,
                                                text="Sistema listo para análisis comparativo",
                                                font=('Segoe UI', 8),
                                                fg=COLORS['text'],
                                                bg=COLORS['background'],
                                                wraplength=160)
        self.comparador_progress_text.pack(fill=tk.X, padx=20, pady=2)
        
        # Botones de acción del comparador
        comparador_actions = [
            ("🔬 Comparación Completa", self._ejecutar_comparacion_completa, COLORS['primary']),
            ("⚡ Comparación Rápida", self._ejecutar_comparacion_rapida, COLORS['warning']),
            ("📈 Ver Último Reporte", self._ver_ultimo_reporte, COLORS['info']),
            ("🏆 Mejor Clasificador", self._mostrar_mejor_clasificador, COLORS['success'])
        ]
        
        for text, command, color in comparador_actions:
            btn = RoundedButton(section_frame,
                              text=text,
                              command=command,
                              background=color,
                              activebackground=COLORS['primary_dark'],
                              width=320,
                              height=30)
            btn.pack(pady=3, padx=20)
    
    def _execute_kmeans(self):
        """Ejecuta el análisis K-Means completo."""
        caracteristicas = self.kmeans_caracteristicas_var.get()
        clusters_str = self.kmeans_clusters_var.get()
        auto_eval = self.kmeans_auto_eval_var.get()
        
        self.classifier_controller.execute_kmeans_analysis(
            caracteristicas, clusters_str, auto_eval
        )
    
    def _show_best_combination(self):
        """Muestra la mejor combinación de características encontrada."""
        self.classifier_controller.show_best_kmeans_combination()
    
    def _test_selected_kmeans(self):
        """Permite probar el K-Means con la mejor combinación en una imagen específica."""
        self.classifier_controller.test_selected_kmeans()
    
    # ===== MÉTODOS DEL COMPARADOR TRIPLE =====
    
    def _ejecutar_comparacion_completa(self):
        """Ejecuta comparación completa de los tres clasificadores."""
        # Verificar que hay datos disponibles
        if not hasattr(self, 'stats_rgb') or not self.stats_rgb:
            self._update_comparador_status('error', 'Error: No hay datos cargados')
            return
        
        # Verificar que tenemos los datos particionados
        if not self.datos_train or not self.datos_val or not self.datos_test:
            self._update_comparador_status('error', 'Error: No hay datos particionados disponibles')
            return
        
        # Actualizar estado
        self._update_comparador_status('working', 'Iniciando comparación completa...')
        self.comparador_progress_text.configure(text="Preparando clasificadores...")
        
        # Crear comparador si no existe
        if not self.comparador_triple:
            self.comparador_triple = ComparadorTriple(
                self.datos_train, 
                self.datos_val, 
                self.datos_test, 
                self.stats_rgb
            )
        
        # Obtener configuración
        paralelo = self.comparador_paralelo_var.get()
        metricas_detalladas = self.comparador_metricas_detalladas_var.get()
        
        # Ejecutar en hilo separado
        thread = threading.Thread(
            target=self._ejecutar_comparacion_worker,
            args=(False, paralelo, metricas_detalladas)  # False = no es rápida
        )
        thread.daemon = True
        thread.start()
    
    def _ejecutar_comparacion_rapida(self):
        """Ejecuta comparación rápida de los tres clasificadores."""
        # Verificar que hay datos disponibles
        if not hasattr(self, 'stats_rgb') or not self.stats_rgb:
            self._update_comparador_status('error', 'Error: No hay datos cargados')
            return
        
        # Verificar que tenemos los datos particionados
        if not self.datos_train or not self.datos_val or not self.datos_test:
            self._update_comparador_status('error', 'Error: No hay datos particionados disponibles')
            return
        
        # Actualizar estado
        self._update_comparador_status('working', 'Iniciando comparación rápida...')
        self.comparador_progress_text.configure(text="Ejecutando evaluación básica...")
        
        # Ejecutar en hilo separado
        thread = threading.Thread(
            target=self._ejecutar_comparacion_worker,
            args=(True, True, False)  # True = rápida, paralelo=True, metricas=False
        )
        thread.daemon = True
        thread.start()
    
    def _ejecutar_comparacion_worker(self, es_rapida, paralelo, metricas_detalladas):
        """Worker que ejecuta la comparación en hilo separado."""
        try:
            # Callback para actualizar progreso
            def callback_progreso(progreso, mensaje):
                self.after(0, lambda: self.comparador_progress_text.configure(text=f"{mensaje} ({progreso:.1f}%)"))
            
            print(f"🔄 Iniciando comparación - Rápida: {es_rapida}, Paralelo: {paralelo}")
            print(f"📊 Datos disponibles - Train: {len(self.datos_train) if self.datos_train else 0}, Val: {len(self.datos_val) if self.datos_val else 0}, Test: {len(self.datos_test) if self.datos_test else 0}")
            
            if es_rapida:
                # Comparación rápida usando función auxiliar
                print("⚡ Ejecutando comparación rápida...")
                reporte = ejecutar_comparacion_rapida(
                    self.datos_train,
                    self.datos_val,
                    self.datos_test,
                    self.stats_rgb
                )
            else:
                # Comparación completa - solo pasar usar_paralelismo
                print("🔬 Ejecutando comparación completa...")
                reporte = self.comparador_triple.ejecutar_comparacion_completa(
                    usar_paralelismo=paralelo
                )
            
            print(f"✅ Comparación completada. Resultados: {len(reporte.resultados) if reporte.resultados else 0}")
            
            # Guardar reporte
            self.ultimo_reporte_triple = reporte
            
            # Actualizar UI en hilo principal
            self.after(0, lambda: self._mostrar_resultados_comparacion(reporte))
            
        except Exception as e:
            import traceback
            error_msg = f"Error en comparación: {str(e)}"
            traceback_str = traceback.format_exc()
            print(f"❌ {error_msg}")
            print(f"📋 Traceback:\n{traceback_str}")
            self.after(0, lambda: self._mostrar_error_comparacion(error_msg))
    
    def _convertir_reporte_a_dict(self, reporte):
        """Convierte un ReporteTriple a formato dict para el diálogo."""
        resultados_dict = {}
        
        for resultado in reporte.resultados:
            nombre = resultado.nombre_clasificador
            resultados_dict[nombre] = {
                'metricas': {
                    'precision': resultado.precision,
                    'recall': resultado.recall,
                    'f1_score': resultado.f1_score,
                    'exactitud': resultado.accuracy,  # Mapear accuracy a exactitud
                    'sensibilidad': resultado.recall,  # Sensibilidad = Recall
                    'especificidad': resultado.precision,  # Aproximación usando precision
                    'jaccard': resultado.f1_score * 0.8,  # Aproximación del índice Jaccard
                    'youden': resultado.f1_score  # Usamos F1 como métrica principal
                },
                'umbral': 0.5,  # Valor por defecto para compatibilidad
                'tiempo_total': resultado.tiempo_entrenamiento + resultado.tiempo_evaluacion,
                'mejor_imagen': resultado.imagen_mejor,
                'peor_imagen': resultado.imagen_peor,
                'configuracion': resultado.mejor_configuracion,
                'notas': resultado.notas
            }
        
        return resultados_dict

    def _mostrar_resultados_comparacion(self, reporte):
        """Muestra los resultados de la comparación en la UI."""
        
        # Verificar que tenemos resultados válidos
        if not reporte.resultados:
            self._update_comparador_status('warning', 'Sin resultados válidos')
            self.comparador_progress_text.configure(text="No se obtuvieron resultados válidos")
            return
        
        # Encontrar el mejor F1-Score
        mejor_resultado = max(reporte.resultados, key=lambda x: x.f1_score) if reporte.resultados else None
        mejor_f1 = mejor_resultado.f1_score if mejor_resultado else 0.0
        
        # Debug: Mostrar métricas de todos los clasificadores
        for resultado in reporte.resultados:
            print(f"  {resultado.nombre_clasificador}: F1={resultado.f1_score:.3f}, Precision={resultado.precision:.3f}, Recall={resultado.recall:.3f}, Accuracy={resultado.accuracy:.3f}")
        
        # Actualizar estado
        self._update_comparador_status('success', 'Comparación completada')
        self.comparador_progress_text.configure(text=f"Mejor: {reporte.clasificador_ganador} (F1: {mejor_f1:.3f})")
        
        # Mostrar diálogo con resultados
        try:
            resultados_dict = self._convertir_reporte_a_dict(reporte)
            dialogo = mostrar_dialogo_comparacion(self, resultados_dict, "Resultados de Comparación Triple")
            
        except Exception as e:
            import traceback
            print(f"⚠️ Error mostrando diálogo: {e}")
            print(f"Traceback: {traceback.format_exc()}")
        
        print(f"🏆 Comparación completada. Mejor clasificador: {reporte.clasificador_ganador}")
        print(f"📊 F1-Score: {mejor_f1:.3f}")
        print(f"⏱️ Tiempo total: {reporte.tiempo_total:.2f}s")
    
    def _mostrar_error_comparacion(self, mensaje):
        """Muestra un error de comparación."""
        from tkinter import messagebox
        
        self._update_comparador_status('error', 'Error en comparación')
        self.comparador_progress_text.configure(text="Error - Ver detalles")
        
        messagebox.showerror("Error de Comparación", mensaje)
        print(f"❌ Error en comparación: {mensaje}")
    
    def _ver_ultimo_reporte(self):
        """Muestra el último reporte de comparación generado."""
        if self.ultimo_reporte_triple:
            resultados_dict = self._convertir_reporte_a_dict(self.ultimo_reporte_triple)
            mostrar_dialogo_comparacion(self, resultados_dict, "Último Reporte de Comparación")
        else:
            from tkinter import messagebox
            messagebox.showinfo("Sin Reportes", "No hay reportes de comparación disponibles.\nEjecute una comparación primero.")
    
    def _mostrar_mejor_clasificador(self):
        """Muestra información del mejor clasificador encontrado."""
        if self.ultimo_reporte_triple:
            from tkinter import messagebox
            
            reporte = self.ultimo_reporte_triple
            
            # Obtener información del mejor clasificador
            mejor_nombre = reporte.clasificador_ganador
            tiempo = reporte.tiempo_total
            
            # Encontrar el mejor F1-Score
            mejor_resultado = max(reporte.resultados, key=lambda x: x.f1_score) if reporte.resultados else None
            mejor_f1 = mejor_resultado.f1_score if mejor_resultado else 0.0
            
            mensaje = f"""🏆 MEJOR CLASIFICADOR ENCONTRADO:

📊 Clasificador: {mejor_nombre}
🎯 F1-Score: {mejor_f1:.3f}
⏱️ Tiempo total: {tiempo:.2f}s

🏅 RANKING COMPLETO:"""
            
            # Crear ranking con F1-Scores
            ranking_con_scores = []
            for nombre_clasificador in reporte.ranking:
                resultado = next((r for r in reporte.resultados if r.nombre_clasificador == nombre_clasificador), None)
                f1_score = resultado.f1_score if resultado else 0.0
                ranking_con_scores.append((nombre_clasificador, f1_score))
            
            for i, (nombre, f1_score) in enumerate(ranking_con_scores, 1):
                mensaje += f"\n{i}. {nombre}: {f1_score:.3f}"
            
            messagebox.showinfo("Mejor Clasificador", mensaje)
            
            messagebox.showinfo("Mejor Clasificador", mensaje)
        else:
            from tkinter import messagebox
            messagebox.showinfo("Sin Resultados", "No hay resultados de comparación disponibles.\nEjecute una comparación primero.")
    
    def _update_comparador_status(self, state: str, message: str):
        """Actualiza el estado del comparador en la UI."""
        self.comparador_status.set_state(state, message)
    
    def _mostrar_error_comparacion(self, error_msg: str):
        """Muestra un error de comparación en la UI."""
        print(f"❌ Error en comparación: {error_msg}")
        self._update_comparador_status('error', 'Error en comparación')
        self.comparador_progress_text.configure(text="Error - Ver consola para detalles")
        
        # También mostrar en un diálogo
        from tkinter import messagebox
        messagebox.showerror("Error en Comparación", f"Error ejecutando comparación:\n\n{error_msg}")
    
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