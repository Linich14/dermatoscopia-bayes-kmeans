"""
Ventana principal de la interfaz gráfica para visualización de resultados.
Moderna interfaz tipo dashboard para análisis de imágenes dermatoscópicas.
"""

import tkinter as tk
from tkinter import ttk, filedialog
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
    def __init__(self, stats_rgb):
        super().__init__()
        self.title("Análisis Dermatoscópico")
        self.geometry("1280x800")
        self.configure(**STYLES['root'])
        self.stats_rgb = stats_rgb
        
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
        """Crea el panel lateral con controles"""
        sidebar = RoundedContainer(parent, **STYLES['sidebar'])
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        content_frame = sidebar.inner_frame
        
        # Grupo: Área de análisis
        area_container = RoundedContainer(content_frame, background=COLORS['rosita'], height=DESIGN['container_height'])
        area_container.pack(fill=tk.X, padx=1, pady=1)
        area_container.pack_propagate(False)  # Evita que el contenedor se ajuste al contenido
        
        area_frame = area_container.inner_frame
        
        tk.Label(area_frame,
                text="Área de análisis",
                font=('Segoe UI', 9, 'bold'),
                fg=COLORS['text'],
                bg=COLORS['background']).pack(fill=tk.X, padx=2, pady=(2,1))
        
        # Botones de área con menos espacio
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
        btn_sana.pack(pady=1)
        
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
        btn_lesion.pack(pady=1)
        
        # Grupo: Canal de color
        canal_container = RoundedContainer(content_frame, background=COLORS['rosita'], height=DESIGN['container_height'] + 30)  # Más alto para 3 botones
        canal_container.pack(fill=tk.X, padx=1, pady=1)
        canal_container.pack_propagate(False)  # Evita que el contenedor se ajuste al contenido
        
        canal_frame = canal_container.inner_frame
        
        tk.Label(canal_frame,
                text="Canal de color",
                font=('Segoe UI', 9, 'bold'),
                fg=COLORS['text'],
                bg=COLORS['background']).pack(fill=tk.X, padx=2, pady=(2,1))
        
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
            btn.pack(pady=1)
        
        # Botones de acción en un contenedor redondeado
        actions_container = RoundedContainer(content_frame, background=COLORS['rosita'], height=DESIGN['container_height'])
        actions_container.pack(fill=tk.X, padx=1, pady=1)
        actions_container.pack_propagate(False)  # Evita que el contenedor se ajuste al contenido
        
        actions_frame = actions_container.inner_frame
        
        tk.Label(actions_frame,
                text="Acciones",
                font=('Segoe UI', 9, 'bold'),
                fg=COLORS['text'],
                bg=COLORS['background']).pack(fill=tk.X, padx=2, pady=(2,1))
        
        reset_btn = RoundedButton(actions_frame,
                               text="Reiniciar valores 🔄",
                               command=self._reiniciar_valores,
                               background=COLORS['primary'],
                               activebackground=COLORS['primary_dark'],
                               foreground='white',
                               activeforeground='white',
                               width=170,
                               height=DESIGN['button_height'])
        reset_btn.pack(pady=1)
    
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