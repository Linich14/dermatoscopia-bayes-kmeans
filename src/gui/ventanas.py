"""
Ventana principal de la interfaz gráfica para visualización de resultados.
Moderna interfaz tipo dashboard para análisis de imágenes dermatoscópicas.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from .styles import COLORS, STYLES, DESIGN
import tkinter.font as tkfont

class RoundedContainer(tk.Canvas):
    def __init__(self, parent, **kwargs):
        # Extraer propiedades específicas del contenedor
        self.background = kwargs.pop('background', COLORS['card_bg'])
        self.border_color = kwargs.pop('highlightbackground', COLORS['border'])
        self.border_width = kwargs.pop('highlightthickness', 1)
        self.padx = kwargs.pop('padx', 10) if 'padx' in kwargs else 10
        self.pady = kwargs.pop('pady', 10) if 'pady' in kwargs else 10
        
        # Inicializar el Canvas
        super().__init__(parent, highlightthickness=0, **kwargs)
        
        # Configurar el fondo del canvas
        self.configure(bg=parent.cget('bg'))
        
        # Crear el frame interno para el contenido
        self.inner_frame = tk.Frame(self, bg=self.background)
        
        # Vinculamos el evento de redimensionamiento después de crear el frame
        self.bind('<Configure>', self._on_resize)
        
        # Dibujamos el contenedor inicial
        self._draw_container()
        
    def _draw_container(self):
        """Dibuja el contenedor redondeado y posiciona el frame interno"""
        width = self.winfo_width()
        height = self.winfo_height()
        
        if width <= 1 or height <= 1:
            return
        
        self.delete('container')
        
        # Dibujar el contenedor redondeado
        if self.border_width > 0:
            # Borde
            self.create_rounded_rect(0, 0, width, height,
                                   DESIGN['border_radius'],
                                   fill=self.border_color,
                                   tags='container')
            # Interior
            self.create_rounded_rect(self.border_width, self.border_width,
                                   width - self.border_width, height - self.border_width,
                                   DESIGN['border_radius'] - self.border_width,
                                   fill=self.background,
                                   tags='container')
        else:
            self.create_rounded_rect(0, 0, width, height,
                                   DESIGN['border_radius'],
                                   fill=self.background,
                                   tags='container')
    
    def _on_resize(self, event):
        """Maneja el evento de redimensionamiento"""
        self._draw_container()
        # Actualizar la posición del frame interno con padding
        try:
            self.inner_frame.place_forget()  # Primero removemos el frame
            width = self.winfo_width()
            height = self.winfo_height()
            if width > 1 and height > 1:  # Solo reposicionamos si hay espacio suficiente
                self.inner_frame.place(
                    x=self.border_width + self.padx,
                    y=self.border_width + self.pady,
                    width=max(0, width - (self.border_width * 2 + self.padx * 2)),
                    height=max(0, height - (self.border_width * 2 + self.pady * 2))
                )
        except tk.TclError:
            pass  # Ignoramos errores si el widget ya no existe
            
    def create_rounded_rect(self, x1, y1, x2, y2, radius, **kwargs):
        points = [
            x1+radius, y1,
            x2-radius, y1,
            x2, y1,
            x2, y1+radius,
            x2, y2-radius,
            x2, y2,
            x2-radius, y2,
            x1+radius, y2,
            x1, y2,
            x1, y2-radius,
            x1, y1+radius,
            x1, y1
        ]
        return self.create_polygon(points, smooth=True, **kwargs)

class RoundedButton(tk.Canvas):
    def __init__(self, parent, text, command=None, variable=None, value=None, **kwargs):
        height = kwargs.pop('height', 35)
        width = kwargs.pop('width', 120)
        self.background = kwargs.pop('background', COLORS['primary'])
        self.activebackground = kwargs.pop('activebackground', COLORS['primary_dark'])
        self.foreground = kwargs.pop('foreground', 'white')
        self.activeforeground = kwargs.pop('activeforeground', 'white')
        super().__init__(parent, width=width, height=height, highlightthickness=0, **kwargs)
        
        self.command = command
        self.text = text
        self.variable = variable
        self.value = value
        self.is_hover = False
        self.is_selected = False
        
        self.font = tkfont.Font(family='Segoe UI', size=10, weight='bold')
        
        # Configurar el fondo del canvas y eventos
        self.configure(bg=parent.cget('bg'))
        self.bind('<Button-1>', self._on_click)
        self.bind('<Enter>', self._on_enter)
        self.bind('<Leave>', self._on_leave)
        self.bind('<Configure>', self._on_configure)
        
        # Si es un botón vinculado a una variable
        if self.variable:
            self.variable.trace_add('write', self._update_state)
        
        # Dibujar el botón inicial
        self._draw_button()
    
    def _draw_button(self):
        width = self.winfo_width()
        height = self.winfo_height()
        if width <= 1 or height <= 1:
            return
            
        self.delete('all')  # Limpiar todo el canvas
        
        # Determinar los colores basados en el estado
        is_active = self.is_selected or self.is_hover
        bg_color = self.activebackground if is_active else self.background
        fg_color = self.activeforeground if is_active else self.foreground
        
        # Dibujar el fondo del botón
        self.create_rounded_rect(
            2, 2, width-2, height-2,
            DESIGN['border_radius'], fill=bg_color, tags='button'
        )
        
        # Dibujar el texto centrado
        self.create_text(
            width/2,
            height/2,
            text=self.text,
            fill=fg_color,
            font=self.font,
            tags='text'
        )
    
    def _on_click(self, event):
        if self.command is not None:
            if self.variable is not None and self.value is not None:
                old_value = self.variable.get()
                if old_value != self.value:
                    self.variable.set(self.value)
            self.command()
    
    def _on_enter(self, event):
        self.is_hover = True
        self._draw_button()
    
    def _on_leave(self, event):
        self.is_hover = False
        self._draw_button()
        
    def _on_configure(self, event):
        self._draw_button()
        
    def _update_state(self, *args):
        if self.variable is not None and self.value is not None:
            self.is_selected = self.variable.get() == self.value
        else:
            self.is_selected = False
        self._draw_button()
    
    def create_rounded_rect(self, x1, y1, x2, y2, radius, **kwargs):
        points = [
            x1+radius, y1,
            x2-radius, y1,
            x2, y1,
            x2, y1+radius,
            x2, y2-radius,
            x2, y2,
            x2-radius, y2,
            x1+radius, y2,
            x1, y2,
            x1, y2-radius,
            x1, y1+radius,
            x1, y1
        ]
        return self.create_polygon(points, smooth=True, **kwargs)

class VentanaPrincipal(tk.Tk):
    def __init__(self, stats_rgb, clasificador=None):
        super().__init__()
        self.title("Análisis Dermatoscópico")
        self.geometry("1280x900")  # Aumentar altura
        self.configure(**STYLES['root'])
        self.stats_rgb = stats_rgb
        self.clasificador = clasificador
        
        # Variables de control
        self.area_var = tk.StringVar(value='lesion')
        self.canal_var = tk.StringVar(value='R')
        
        self._crear_interfaz()
        # Actualizar el histograma inicial
        self.after(100, self.actualizar_histograma)
        
    def _crear_interfaz(self):
        """Crea la estructura principal de la interfaz"""
        # Header
        self._crear_header()
        
        # Contenedor principal
        main_container = tk.Frame(self, **STYLES['root'])
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Sidebar
        self._crear_sidebar(main_container)
        
        # Área principal
        self._crear_area_principal(main_container)
    
    def _crear_header(self):
        """Crea la barra superior con título"""
        header = tk.Frame(self, **STYLES['header'])
        header.pack(fill=tk.X)
        header.pack_propagate(False)  # Evita que el frame se ajuste al contenido
        
        # Título
        title = tk.Label(header,
                        text="Análisis de Imágenes Dermatoscópicas",
                        fg='white',
                        bg=COLORS['primary'],
                        font=('Segoe UI', 16, 'bold'))
        title.place(relx=0.5, rely=0.5, anchor='center')  # Centra el título vertical y horizontalmente
    
    def _crear_sidebar(self, parent):
        """Crea el panel lateral con controles y scroll"""
        # Contenedor principal del sidebar
        sidebar_main = tk.Frame(parent, bg=COLORS['background'], width=200)
        sidebar_main.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        sidebar_main.pack_propagate(False)
        
        # Canvas para scroll
        canvas = tk.Canvas(sidebar_main, bg=COLORS['background'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(sidebar_main, orient="vertical", command=canvas.yview)
        
        # Frame scrollable
        scrollable_frame = tk.Frame(canvas, bg=COLORS['background'])
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Empaquetar canvas y scrollbar
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Usar scrollable_frame como content_frame
        content_frame = scrollable_frame
        
        # Bind del scroll del mouse
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _bind_to_mousewheel(event):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        def _unbind_from_mousewheel(event):
            canvas.unbind_all("<MouseWheel>")
        
        canvas.bind('<Enter>', _bind_to_mousewheel)
        canvas.bind('<Leave>', _unbind_from_mousewheel)
        
        # Grupo: Área de análisis
        area_frame = tk.LabelFrame(content_frame, 
                                  text="Área de análisis",
                                  bg=COLORS['background'],
                                  fg=COLORS['text'],
                                  font=('Segoe UI', 9, 'bold'),
                                  relief='groove',
                                  bd=1)
        area_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Botones de área
        btn_sana = RoundedButton(area_frame,
                              text="Sana ✅",
                              command=lambda: self._select_area('sana'),
                              background=COLORS['secondary'],
                              activebackground=COLORS['primary'],
                              foreground='white',
                              activeforeground='white',
                              width=170,
                              height=DESIGN['button_height'],
                              variable=self.area_var,
                              value='sana')
        btn_sana.pack(pady=3, padx=5)
        
        btn_lesion = RoundedButton(area_frame,
                                text="Lesión ⚠️",
                                command=lambda: self._select_area('lesion'),
                                background=COLORS['primary'],
                                activebackground=COLORS['secondary'],
                                foreground='white',
                                activeforeground='white',
                                width=170,
                                height=DESIGN['button_height'],
                                variable=self.area_var,
                                value='lesion')
        btn_lesion.pack(pady=3, padx=5)
        
        # Grupo: Canal de color
        canal_frame = tk.LabelFrame(content_frame,
                                   text="Canal de color",
                                   bg=COLORS['background'],
                                   fg=COLORS['text'],
                                   font=('Segoe UI', 9, 'bold'),
                                   relief='groove',
                                   bd=1)
        canal_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Botones de canal
        canales = [
            ('R', '🔴 R', COLORS['accent'], COLORS['primary']),
            ('G', '🟢 G', '#2ECC71', '#1D8348'),
            ('B', '🔵 B', '#3498DB', '#2874A6')
        ]
        
        for canal, texto, color, color_hover in canales:
            btn = RoundedButton(canal_frame,
                            text=texto,
                            command=lambda c=canal: self._select_canal(c),
                            background=color,
                            activebackground=color_hover,
                            foreground='white',
                            activeforeground='white',
                            width=170,
                            height=DESIGN['button_height'],
                            variable=self.canal_var,
                            value=canal)
            btn.pack(pady=3, padx=5)
        
        # Grupo: Acciones
        actions_frame = tk.LabelFrame(content_frame,
                                     text="Acciones",
                                     bg=COLORS['background'],
                                     fg=COLORS['text'],
                                     font=('Segoe UI', 9, 'bold'),
                                     relief='groove',
                                     bd=1)
        actions_frame.pack(fill=tk.X, padx=5, pady=5)
        
        reset_btn = RoundedButton(actions_frame,
                               text="Reiniciar valores 🔄",
                               command=self._reiniciar_valores,
                               background=COLORS['primary'],
                               activebackground=COLORS['primary_dark'],
                               foreground='white',
                               activeforeground='white',
                               width=170,
                               height=DESIGN['button_height'])
        reset_btn.pack(pady=3, padx=5)
        
        # Grupo: Clasificador Bayesiano
        bayes_frame = tk.LabelFrame(content_frame,
                                   text="Clasificador Bayesiano RGB",
                                   bg=COLORS['background'],
                                   fg=COLORS['text'],
                                   font=('Segoe UI', 9, 'bold'),
                                   relief='groove',
                                   bd=1)
        bayes_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Variables de control para clasificador
        self.criterio_var = tk.StringVar(value='youden')
        self.clasificador_entrenado = False
        self.progreso_var = tk.StringVar(value="No entrenado")
        
        # Dropdown para criterio de umbral
        tk.Label(bayes_frame,
                text="Criterio de umbral:",
                font=('Segoe UI', 8),
                fg=COLORS['text'],
                bg=COLORS['background']).pack(fill=tk.X, padx=5, pady=(5,0))
        
        criterio_combo = ttk.Combobox(bayes_frame,
                                     textvariable=self.criterio_var,
                                     values=['youden', 'equal_error', 'prior_balanced'],
                                     state='readonly',
                                     font=('Segoe UI', 8))
        criterio_combo.pack(fill=tk.X, padx=5, pady=2)
        
        # Label de estado
        self.estado_label = tk.Label(bayes_frame,
                                    textvariable=self.progreso_var,
                                    font=('Segoe UI', 8),
                                    fg=COLORS['text'],
                                    bg=COLORS['background'],
                                    wraplength=160)
        self.estado_label.pack(fill=tk.X, padx=5, pady=5)
        
        # Botones de clasificador con más espacio
        entrenar_btn = RoundedButton(bayes_frame,
                                   text="Entrenar Clasificador 🤖",
                                   command=self._entrenar_clasificador,
                                   background=COLORS['accent'],
                                   activebackground='#E67E22',
                                   foreground='white',
                                   activeforeground='white',
                                   width=170,
                                   height=25)
        entrenar_btn.pack(pady=3, padx=5)
        
        evaluar_btn = RoundedButton(bayes_frame,
                                  text="Evaluar Modelo 📊",
                                  command=self._evaluar_clasificador,
                                  background=COLORS['secondary'],
                                  activebackground='#117A65',
                                  foreground='white',
                                  activeforeground='white',
                                  width=170,
                                  height=25)
        evaluar_btn.pack(pady=3, padx=5)
        
        comparar_btn = RoundedButton(bayes_frame,
                                   text="Comparar Criterios ⚖️",
                                   command=self._comparar_criterios,
                                   background='#8E44AD',
                                   activebackground='#7D3C98',
                                   foreground='white',
                                   activeforeground='white',
                                   width=170,
                                   height=25)
        comparar_btn.pack(pady=3, padx=5)
        
        clasificar_btn = RoundedButton(bayes_frame,
                                     text="Clasificar Imagen 🖼️",
                                     command=self._clasificar_imagen,
                                     background='#E74C3C',
                                     activebackground='#C0392B',
                                     foreground='white',
                                     activeforeground='white',
                                     width=170,
                                     height=25)
        clasificar_btn.pack(pady=3, padx=5)
    
    def _crear_area_principal(self, parent):
        """Crea el área principal con el gráfico"""
        # Contenedor principal
        main_area = RoundedContainer(parent, **STYLES['root'])
        main_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        main_content = main_area.inner_frame
        
        # Tarjeta para el gráfico
        graph_card = RoundedContainer(main_content, **STYLES['card'])
        graph_card.pack(fill=tk.BOTH, expand=True)
        
        # Frame para el gráfico
        self.frame_grafico = RoundedContainer(graph_card.inner_frame, background=COLORS['secondary'])
        self.frame_grafico.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def _select_area(self, area):
        self.area_var.set(area)
        self.actualizar_histograma()
    
    def _select_canal(self, canal):
        self.canal_var.set(canal)
        self.actualizar_histograma()
    
    def _reiniciar_valores(self):
        """Reinicia los valores a su estado inicial"""
        self.area_var.set('lesion')
        self.canal_var.set('R')
        self.actualizar_histograma()
    
    def actualizar_histograma(self):
        """Actualiza el histograma con los valores actuales"""
        # Limpiar el frame del gráfico
        for widget in self.frame_grafico.inner_frame.winfo_children():
            widget.destroy()
        
        # Crear nueva figura con estilo
        plt_colors = {
            'R': '#FF69B4',  # Rosa para el canal R
            'G': '#98FB98',  # Verde pastel para el canal G
            'B': '#87CEEB'   # Azul cielo para el canal B
        }
        
        fig = Figure(figsize=(8, 6), dpi=100)
        fig.patch.set_facecolor(COLORS['rosita'])
        
        # Configurar el subplot con grid y estilo
        ax = fig.add_subplot(111)
        ax.set_facecolor(COLORS['card_bg'])
        
        # Configurar el grid con estilo más sutil
        ax.grid(True, linestyle=':', alpha=0.2, color='gray')
        ax.set_axisbelow(True)  # Grid detrás de las barras
        
        # Obtener datos según el área y canal seleccionados
        area = self.area_var.get()
        canal = self.canal_var.get()
        stats = self.stats_rgb[area][canal]
        
        try:
            # Preparar datos del histograma
            hist_data = stats['histograma']
            bins = np.linspace(0, 255, len(hist_data))
            width = bins[1] - bins[0] * 0.8  # Reducir ancho para separación
            
            # Dibujar histograma con estilo mejorado
            bars = ax.bar(bins, hist_data, 
                         width=width * 0.8,  # Reducir el ancho para más separación
                         color=plt_colors[canal],
                         alpha=0.7,
                         edgecolor='white',  # Borde blanco para mejor separación visual
                         linewidth=0.5,
                         label=f'Canal {canal}')
            
            # Añadir valor medio y desviación estándar
            media = stats['media'] * 255 if stats['media'] is not None else 0
            std = stats['std'] * 255 if stats['std'] is not None else 0
            
            # Sombrear área de la desviación estándar
            x = np.linspace(media - std, media + std, 100)
            max_height = max(hist_data) if hist_data else 0
            ax.fill_between(x, 0, max_height * 0.3,
                          color=plt_colors[canal],
                          alpha=0.2,
                          label=f'σ = {std:.1f}')
            
            # Añadir línea vertical para la media
            ax.axvline(x=media, color='white', 
                      linestyle='--', linewidth=1,
                      label=f'µ = {media:.1f}')
            
            # Configurar aspecto
            ax.set_xlim(-width/2, 255+width/2)
            
            # Título y etiquetas con estilo
            title_text = f'Distribución de Intensidades - Canal {canal}'
            subtitle_text = f'Área {area.capitalize()}\nMedia (µ) = {media:.1f}, Desviación Estándar (σ) = {std:.1f}'
            
            ax.set_title(title_text + '\n' + subtitle_text,
                        color=COLORS['text'],
                        pad=20,
                        fontsize=12,
                        fontweight='bold')
            
            # Personalizar ejes
            ax.tick_params(colors=COLORS['text'], length=5)
            
            # Crear boxes de color rosa claro para las etiquetas
            ax.set_xlabel('Intensidad (Normalizada)', 
                         color=COLORS['primary'],
                         fontsize=10,
                         fontweight='bold',
                         bbox=dict(facecolor=COLORS['secondary'],
                                 alpha=0.2,
                                 edgecolor=COLORS['primary'],
                                 pad=5))
            ax.set_ylabel('Frecuencia Relativa', 
                         color=COLORS['primary'],
                         fontsize=10,
                         fontweight='bold',
                         bbox=dict(facecolor=COLORS['secondary'],
                                 alpha=0.2,
                                 edgecolor=COLORS['primary'],
                                 pad=5))
            
            # Añadir leyenda con estilo
            ax.legend(loc='upper right', 
                     facecolor=COLORS['card_bg'],
                     edgecolor=COLORS['primary'],
                     labelcolor=COLORS['text'],
                     framealpha=0.8)
            
            # Personalizar los bordes del gráfico
            for spine in ax.spines.values():
                spine.set_color(COLORS['border'])
                spine.set_linewidth(1.5)
            
            # Ajustar layout
            fig.tight_layout(pad=3.0)
            
            # Crear canvas y empaquetarlo
            canvas = FigureCanvasTkAgg(fig, self.frame_grafico.inner_frame)
            canvas.draw()
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
        except Exception as e:
            print(f"Error al dibujar el histograma: {e}")
            # Crear una etiqueta de error en lugar del gráfico
            tk.Label(self.frame_grafico.inner_frame,
                    text="Error al cargar el gráfico",
                    fg=COLORS['text'],
                    bg=COLORS['card_bg'],
                    font=('Segoe UI', 12)).pack(pady=20)
    
    def _entrenar_clasificador(self):
        """Entrena el clasificador bayesiano con el criterio seleccionado"""
        import threading
        from clasificadores.clasificador_bayesiano import ClasificadorBayesianoRGB
        from preprocesamiento.carga import cargar_imagenes_y_mascaras
        from preprocesamiento.particion import particionar_datos
        
        def entrenar_en_hilo():
            try:
                self.progreso_var.set("Cargando datos...")
                self.update()
                
                # Cargar y particionar datos
                imagenes = cargar_imagenes_y_mascaras()
                self.train, self.val, self.test = particionar_datos(imagenes)
                
                self.progreso_var.set("Entrenando...")
                self.update()
                
                # Crear y entrenar clasificador
                criterio = self.criterio_var.get()
                self.clasificador = ClasificadorBayesianoRGB(criterio_umbral=criterio)
                self.clasificador.entrenar(self.train)
                
                self.clasificador_entrenado = True
                parametros = self.clasificador.obtener_parametros()
                
                self.progreso_var.set(f"✅ Entrenado\nCriterio: {criterio}\nUmbral: {parametros['umbral']:.4f}")
                
            except Exception as e:
                self.progreso_var.set(f"❌ Error: {str(e)[:30]}...")
        
        # Ejecutar en hilo separado para no bloquear GUI
        thread = threading.Thread(target=entrenar_en_hilo)
        thread.daemon = True
        thread.start()
    
    def _evaluar_clasificador(self):
        """Evalúa el clasificador entrenado y muestra resultados"""
        if not hasattr(self, 'clasificador') or not self.clasificador_entrenado:
            self._mostrar_mensaje_error("Debe entrenar el clasificador primero")
            return
        
        import threading
        from clasificadores.evaluacion import evaluar_clasificador_en_conjunto
        
        def evaluar_en_hilo():
            try:
                self.progreso_var.set("Evaluando en test...")
                self.update()
                
                # Evaluar en conjunto de test
                metricas = evaluar_clasificador_en_conjunto(self.clasificador, self.test)
                
                # Mostrar ventana de resultados
                self._mostrar_resultados_evaluacion(metricas)
                
                self.progreso_var.set("✅ Evaluación completa")
                
            except Exception as e:
                self.progreso_var.set(f"❌ Error: {str(e)[:30]}...")
                self._mostrar_mensaje_error(f"Error en evaluación: {e}")
        
        thread = threading.Thread(target=evaluar_en_hilo)
        thread.daemon = True
        thread.start()
    
    def _comparar_criterios(self):
        """Compara diferentes criterios de umbral"""
        import threading
        from clasificadores.evaluacion import comparar_criterios_umbral
        
        def comparar_en_hilo():
            try:
                self.progreso_var.set("Comparando criterios...")
                self.update()
                
                # Cargar datos si no están cargados
                if not hasattr(self, 'train'):
                    from preprocesamiento.carga import cargar_imagenes_y_mascaras
                    from preprocesamiento.particion import particionar_datos
                    imagenes = cargar_imagenes_y_mascaras()
                    self.train, self.val, self.test = particionar_datos(imagenes)
                
                # Comparar criterios
                resultados = comparar_criterios_umbral(self.train, self.val)
                
                # Mostrar ventana de comparación
                self._mostrar_comparacion_criterios(resultados)
                
                self.progreso_var.set("✅ Comparación completa")
                
            except Exception as e:
                self.progreso_var.set(f"❌ Error: {str(e)[:30]}...")
                self._mostrar_mensaje_error(f"Error en comparación: {e}")
        
        thread = threading.Thread(target=comparar_en_hilo)
        thread.daemon = True
        thread.start()
    
    def _clasificar_imagen(self):
        """Permite al usuario seleccionar una imagen y la clasifica"""
        if not hasattr(self, 'clasificador') or not self.clasificador_entrenado:
            self._mostrar_mensaje_error("Debe entrenar el clasificador primero")
            return
        
        from tkinter import filedialog
        import cv2
        from PIL import Image
        
        # Seleccionar archivo de imagen
        archivo = filedialog.askopenfilename(
            title="Seleccionar imagen para clasificar",
            filetypes=[
                ("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("Todos los archivos", "*.*")
            ]
        )
        
        if not archivo:
            return
        
        try:
            # Cargar y normalizar imagen
            imagen = cv2.imread(archivo)
            if imagen is None:
                self._mostrar_mensaje_error("No se pudo cargar la imagen seleccionada")
                return
            
            imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            imagen_norm = imagen_rgb.astype(np.float32) / 255.0
            
            # Clasificar imagen
            self.progreso_var.set("Clasificando imagen...")
            self.update()
            
            mascara_pred = self.clasificador.clasificar(imagen_norm)
            
            # Mostrar resultado
            self._mostrar_resultado_clasificacion(imagen_rgb, mascara_pred, archivo)
            
            self.progreso_var.set("✅ Clasificación completa")
            
        except Exception as e:
            self._mostrar_mensaje_error(f"Error al clasificar imagen: {e}")
            self.progreso_var.set("❌ Error en clasificación")
    
    def _mostrar_resultados_evaluacion(self, metricas):
        """Muestra ventana con resultados de evaluación"""
        ventana = tk.Toplevel(self)
        ventana.title("Resultados de Evaluación")
        ventana.geometry("500x400")
        ventana.configure(bg=COLORS['background'])
        
        # Frame principal con scroll
        main_frame = tk.Frame(ventana, bg=COLORS['background'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Título
        tk.Label(main_frame,
                text="📊 Resultados de Evaluación",
                font=('Segoe UI', 14, 'bold'),
                fg=COLORS['text'],
                bg=COLORS['background']).pack(pady=(0, 10))
        
        # Métricas principales
        metricas_frame = RoundedContainer(main_frame, background=COLORS['card_bg'])
        metricas_frame.pack(fill=tk.X, pady=5)
        
        contenido = metricas_frame.inner_frame
        
        metricas_texto = f"""
Exactitud: {metricas['exactitud']:.4f} ({metricas['exactitud']*100:.1f}%)
Precisión: {metricas['precision']:.4f} ({metricas['precision']*100:.1f}%)
Sensibilidad: {metricas['sensibilidad']:.4f} ({metricas['sensibilidad']*100:.1f}%)
Especificidad: {metricas['especificidad']:.4f} ({metricas['especificidad']*100:.1f}%)
F1-Score: {metricas['f1_score']:.4f}
Índice Jaccard: {metricas['jaccard']:.4f}
Índice Youden: {metricas['youden']:.4f}
        """
        
        tk.Label(contenido,
                text=metricas_texto.strip(),
                font=('Consolas', 10),
                fg=COLORS['text'],
                bg=COLORS['card_bg'],
                justify=tk.LEFT).pack(padx=10, pady=10)
        
        # Matriz de confusión
        matriz_frame = RoundedContainer(main_frame, background=COLORS['card_bg'])
        matriz_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(matriz_frame.inner_frame,
                text="Matriz de Confusión",
                font=('Segoe UI', 11, 'bold'),
                fg=COLORS['text'],
                bg=COLORS['card_bg']).pack(pady=(10, 5))
        
        mc = metricas['matriz_confusion']
        matriz_texto = f"""
        Predicción
        Lesión    Sana
Real Lesión  {mc['TP']:6d}  {mc['FN']:6d}
     Sana    {mc['FP']:6d}  {mc['TN']:6d}
        """
        
        tk.Label(matriz_frame.inner_frame,
                text=matriz_texto.strip(),
                font=('Consolas', 9),
                fg=COLORS['text'],
                bg=COLORS['card_bg'],
                justify=tk.CENTER).pack(padx=10, pady=(0, 10))
    
    def _mostrar_comparacion_criterios(self, resultados):
        """Muestra ventana con comparación de criterios"""
        ventana = tk.Toplevel(self)
        ventana.title("Comparación de Criterios")
        ventana.geometry("700x500")
        ventana.configure(bg=COLORS['background'])
        
        # Frame con scroll
        canvas = tk.Canvas(ventana, bg=COLORS['background'])
        scrollbar = ttk.Scrollbar(ventana, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=COLORS['background'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Título
        tk.Label(scrollable_frame,
                text="⚖️ Comparación de Criterios de Umbral",
                font=('Segoe UI', 14, 'bold'),
                fg=COLORS['text'],
                bg=COLORS['background']).pack(pady=(10, 20))
        
        # Crear tarjeta para cada criterio
        for criterio, resultado in resultados.items():
            # Frame del criterio
            criterio_frame = RoundedContainer(scrollable_frame, background=COLORS['card_bg'])
            criterio_frame.pack(fill=tk.X, padx=20, pady=10)
            
            contenido = criterio_frame.inner_frame
            
            # Título del criterio
            tk.Label(contenido,
                    text=f"🎯 {criterio.upper()}",
                    font=('Segoe UI', 12, 'bold'),
                    fg=COLORS['primary'],
                    bg=COLORS['card_bg']).pack(pady=(10, 5))
            
            # Umbral
            tk.Label(contenido,
                    text=f"Umbral: {resultado['umbral']:.6f}",
                    font=('Segoe UI', 10),
                    fg=COLORS['text'],
                    bg=COLORS['card_bg']).pack()
            
            # Métricas
            metricas = resultado['metricas']
            metricas_texto = f"""Exactitud: {metricas['exactitud']:.4f} | Youden: {metricas['youden']:.4f}
Sensibilidad: {metricas['sensibilidad']:.4f} | Especificidad: {metricas['especificidad']:.4f}
Jaccard: {metricas['jaccard']:.4f}"""
            
            tk.Label(contenido,
                    text=metricas_texto,
                    font=('Consolas', 9),
                    fg=COLORS['text'],
                    bg=COLORS['card_bg'],
                    justify=tk.CENTER).pack(pady=(5, 10))
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Encontrar mejor criterio
        mejor_criterio = max(resultados.keys(), 
                           key=lambda x: resultados[x]['metricas']['youden'])
        
        # Mostrar recomendación
        recom_frame = tk.Frame(ventana, bg=COLORS['background'])
        recom_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(recom_frame,
                text=f"🏆 Recomendado: {mejor_criterio.upper()}",
                font=('Segoe UI', 11, 'bold'),
                fg=COLORS['accent'],
                bg=COLORS['background']).pack()
    
    def _mostrar_mensaje_error(self, mensaje):
        """Muestra un mensaje de error en una ventana emergente"""
        ventana = tk.Toplevel(self)
        ventana.title("Error")
        ventana.geometry("400x150")
        ventana.configure(bg=COLORS['background'])
        
        tk.Label(ventana,
                text="❌ Error",
                font=('Segoe UI', 12, 'bold'),
                fg='red',
                bg=COLORS['background']).pack(pady=10)
        
        tk.Label(ventana,
                text=mensaje,
                font=('Segoe UI', 10),
                fg=COLORS['text'],
                bg=COLORS['background'],
                wraplength=350).pack(pady=10)
        
        tk.Button(ventana,
                 text="Cerrar",
                 command=ventana.destroy,
                 bg=COLORS['primary'],
                 fg='white',
                 font=('Segoe UI', 9)).pack(pady=10)
    
    def _mostrar_resultado_clasificacion(self, imagen_original, mascara_pred, nombre_archivo):
        """Muestra ventana con resultado de clasificación de imagen"""
        import os
        
        ventana = tk.Toplevel(self)
        ventana.title(f"Resultado Clasificación - {os.path.basename(nombre_archivo)}")
        ventana.geometry("900x600")
        ventana.configure(bg=COLORS['background'])
        
        # Frame principal
        main_frame = tk.Frame(ventana, bg=COLORS['background'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Título
        tk.Label(main_frame,
                text=f"🎯 Resultado de Clasificación",
                font=('Segoe UI', 14, 'bold'),
                fg=COLORS['text'],
                bg=COLORS['background']).pack(pady=(0, 10))
        
        # Frame para las imágenes
        images_frame = tk.Frame(main_frame, bg=COLORS['background'])
        images_frame.pack(fill=tk.BOTH, expand=True)
        
        # Crear figura con subplots
        fig = Figure(figsize=(12, 8), dpi=80)
        fig.patch.set_facecolor(COLORS['background'])
        
        # Imagen original
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.imshow(imagen_original)
        ax1.set_title("Imagen Original", fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # Máscara predicha
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(mascara_pred, cmap='hot', alpha=0.8)
        ax2.set_title("Máscara Predicha\n(Rojo = Lesión)", fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # Imagen con superposición
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(imagen_original)
        ax3.imshow(mascara_pred, cmap='Reds', alpha=0.5)
        ax3.set_title("Superposición", fontsize=12, fontweight='bold')
        ax3.axis('off')
        
        # Estadísticas
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.axis('off')
        
        # Calcular estadísticas de la clasificación
        total_pixels = mascara_pred.size
        lesion_pixels = np.sum(mascara_pred)
        sana_pixels = total_pixels - lesion_pixels
        
        porcentaje_lesion = (lesion_pixels / total_pixels) * 100
        porcentaje_sana = (sana_pixels / total_pixels) * 100
        
        stats_text = f"""
ESTADÍSTICAS DE CLASIFICACIÓN

Total de píxeles: {total_pixels:,}

Píxeles de lesión: {lesion_pixels:,}
Porcentaje lesión: {porcentaje_lesion:.2f}%

Píxeles sanos: {sana_pixels:,}
Porcentaje sano: {porcentaje_sana:.2f}%

Criterio usado: {self.criterio_var.get().upper()}
Umbral: {self.clasificador.umbral:.6f}
        """
        
        ax4.text(0.1, 0.9, stats_text.strip(), transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS['card_bg'], alpha=0.8))
        
        fig.tight_layout(pad=2.0)
        
        # Crear canvas y empaquetarlo
        canvas = FigureCanvasTkAgg(fig, images_frame)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # Frame de botones
        buttons_frame = tk.Frame(main_frame, bg=COLORS['background'])
        buttons_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Botón para guardar resultado
        tk.Button(buttons_frame,
                 text="💾 Guardar Resultado",
                 command=lambda: self._guardar_resultado_clasificacion(fig, nombre_archivo),
                 bg=COLORS['secondary'],
                 fg='white',
                 font=('Segoe UI', 9)).pack(side=tk.LEFT, padx=(0, 10))
        
        # Botón cerrar
        tk.Button(buttons_frame,
                 text="Cerrar",
                 command=ventana.destroy,
                 bg=COLORS['primary'],
                 fg='white',
                 font=('Segoe UI', 9)).pack(side=tk.RIGHT)
    
    def _guardar_resultado_clasificacion(self, figura, nombre_archivo):
        """Guarda el resultado de clasificación como imagen"""
        from tkinter import filedialog
        import os
        
        # Obtener nombre base sin extensión
        nombre_base = os.path.splitext(os.path.basename(nombre_archivo))[0]
        
        # Seleccionar donde guardar
        archivo_salida = filedialog.asksaveasfilename(
            title="Guardar resultado de clasificación",
            defaultextension=".png",
            initialvalue=f"{nombre_base}_clasificacion.png",
            filetypes=[
                ("PNG", "*.png"),
                ("JPG", "*.jpg"),
                ("PDF", "*.pdf"),
                ("Todos los archivos", "*.*")
            ]
        )
        
        if archivo_salida:
            try:
                figura.savefig(archivo_salida, dpi=300, bbox_inches='tight', 
                             facecolor=COLORS['background'])
                
                # Mostrar confirmación
                tk.messagebox.showinfo("Guardado", f"Resultado guardado en:\n{archivo_salida}")
                
            except Exception as e:
                self._mostrar_mensaje_error(f"Error al guardar: {e}")