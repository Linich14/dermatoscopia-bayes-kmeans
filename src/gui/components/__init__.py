"""
Componentes UI reutilizables.

Este módulo contiene componentes de interfaz gráfica reutilizables
que pueden ser utilizados en toda la aplicación.
"""

import tkinter as tk
import tkinter.font as tkfont
from ..styles import COLORS, STYLES, DESIGN


class RoundedContainer(tk.Canvas):
    """
    Contenedor personalizado con bordes redondeados para elementos de la interfaz.
    
    Este componente crea un contenedor visual con bordes redondeados que mejora
    la apariencia moderna de la interfaz. Proporciona un frame interno donde
    se pueden colocar otros widgets de Tkinter.
    
    Attributes:
        background (str): Color de fondo del contenedor
        border_color (str): Color del borde del contenedor
        border_width (int): Grosor del borde en píxeles
        inner_frame (tk.Frame): Frame interno para colocar widgets
    """
    
    def __init__(self, parent, **kwargs):
        """
        Inicializa el contenedor redondeado.
        
        Args:
            parent (tk.Widget): Widget padre que contendrá este contenedor
            **kwargs: Argumentos adicionales para configuración
        """
        # Extraer propiedades específicas del contenedor
        self.background = kwargs.pop('background', COLORS['card_bg'])
        self.border_color = kwargs.pop('highlightbackground', COLORS['border'])
        self.border_width = kwargs.pop('highlightthickness', 1)
        self.padx = kwargs.pop('padx', 10) if 'padx' in kwargs else 10
        self.pady = kwargs.pop('pady', 10) if 'pady' in kwargs else 10
        
        # Inicializar el Canvas base
        super().__init__(parent, highlightthickness=0, **kwargs)
        
        # Configurar el fondo del canvas para que coincida con el padre
        self.configure(bg=parent.cget('bg'))
        
        # Crear el frame interno para el contenido real
        self.inner_frame = tk.Frame(self, bg=self.background)
        
        # Vincular evento de redimensionamiento para redibujar el contenedor
        self.bind('<Configure>', self._on_resize)
        
        # Dibujar el contenedor inicial
        self._draw_container()
        
    def _draw_container(self):
        """Dibuja el contenedor redondeado y posiciona el frame interno."""
        width = self.winfo_width()
        height = self.winfo_height()
        
        # Evitar dibujar si las dimensiones no son válidas
        if width <= 1 or height <= 1:
            return
        
        # Limpiar dibujos previos
        self.delete('container')
        
        # Dibujar el contenedor redondeado
        if self.border_width > 0:
            # Dibujar borde si está especificado
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
        """Maneja el evento de redimensionamiento."""
        self._draw_container()
        # Actualizar la posición del frame interno con padding
        try:
            self.inner_frame.place_forget()
            width = self.winfo_width()
            height = self.winfo_height()
            if width > 1 and height > 1:
                self.inner_frame.place(
                    x=self.border_width + self.padx,
                    y=self.border_width + self.pady,
                    width=max(0, width - (self.border_width * 2 + self.padx * 2)),
                    height=max(0, height - (self.border_width * 2 + self.pady * 2))
                )
        except tk.TclError:
            pass  # Ignoramos errores si el widget ya no existe
            
    def create_rounded_rect(self, x1, y1, x2, y2, radius, **kwargs):
        """Crea un rectángulo con bordes redondeados."""
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
    """
    Botón personalizado con bordes redondeados y efectos hover.
    
    Proporciona un botón con diseño moderno que puede actuar como
    botón regular o como botón de selección vinculado a una variable.
    """
    
    def __init__(self, parent, text, command=None, variable=None, value=None, **kwargs):
        """
        Inicializa el botón redondeado.
        
        Args:
            parent: Widget padre
            text: Texto del botón
            command: Función a ejecutar al hacer clic
            variable: Variable tkinter para vincular (opcional)
            value: Valor a asignar a la variable (opcional)
            **kwargs: Argumentos adicionales de estilo
        """
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
        """Dibuja el botón con el estado actual."""
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
        """Maneja el evento de clic."""
        if self.command is not None:
            if self.variable is not None and self.value is not None:
                old_value = self.variable.get()
                if old_value != self.value:
                    self.variable.set(self.value)
            self.command()
    
    def _on_enter(self, event):
        """Maneja el evento de entrada del mouse."""
        self.is_hover = True
        self._draw_button()
    
    def _on_leave(self, event):
        """Maneja el evento de salida del mouse."""
        self.is_hover = False
        self._draw_button()
        
    def _on_configure(self, event):
        """Maneja el evento de configuración."""
        self._draw_button()
        
    def _update_state(self, *args):
        """Actualiza el estado de selección del botón."""
        if self.variable is not None and self.value is not None:
            self.is_selected = self.variable.get() == self.value
        else:
            self.is_selected = False
        self._draw_button()
    
    def create_rounded_rect(self, x1, y1, x2, y2, radius, **kwargs):
        """Crea un rectángulo con bordes redondeados."""
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


class ScrollableFrame(tk.Frame):
    """
    Frame con scroll automático.
    
    Proporciona un contenedor que añade scroll automáticamente
    cuando el contenido excede el tamaño del frame.
    """
    
    def __init__(self, parent, **kwargs):
        """
        Inicializa el frame scrollable.
        
        Args:
            parent: Widget padre
            **kwargs: Argumentos adicionales
        """
        super().__init__(parent, **kwargs)
        self.configure(bg=COLORS['content_bg'])
        
        # Canvas para scroll
        self.canvas = tk.Canvas(self, highlightthickness=0, bg=COLORS['content_bg'])
        self.scrollbar = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        
        # Frame scrollable
        self.scrollable_frame = tk.Frame(self.canvas, bg=COLORS['content_bg'])
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # Empaquetar componentes
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # Configurar scroll con mouse
        self._configure_mousewheel()
    
    def _configure_mousewheel(self):
        """Configura el scroll con la rueda del mouse."""
        def _on_mousewheel(event):
            self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        def _bind_to_mousewheel(event):
            self.canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        def _unbind_from_mousewheel(event):
            self.canvas.unbind_all("<MouseWheel>")
        
        self.canvas.bind('<Enter>', _bind_to_mousewheel)
        self.canvas.bind('<Leave>', _unbind_from_mousewheel)


class StatusIndicator(tk.Label):
    """
    Indicador de estado con colores y iconos.
    
    Proporciona un widget para mostrar el estado actual
    de operaciones con retroalimentación visual.
    """
    
    STATES = {
        'idle': {'color': COLORS['text'], 'icon': '⚪'},
        'working': {'color': COLORS['warning'], 'icon': '🔄'},
        'success': {'color': COLORS['success'], 'icon': '✅'},
        'error': {'color': COLORS['primary'], 'icon': '❌'},
        'warning': {'color': COLORS['warning'], 'icon': '⚠️'}
    }
    
    def __init__(self, parent, **kwargs):
        """
        Inicializa el indicador de estado.
        
        Args:
            parent: Widget padre
            **kwargs: Argumentos adicionales
        """
        super().__init__(parent, **kwargs)
        self.configure(
            font=('Segoe UI', 9),
            bg=kwargs.get('bg', COLORS['background']),
            wraplength=200
        )
        self.set_state('idle', "Listo")
    
    def set_state(self, state: str, message: str = ""):
        """
        Establece el estado del indicador.
        
        Args:
            state: Estado a mostrar ('idle', 'working', 'success', 'error', 'warning')
            message: Mensaje a mostrar
        """
        if state in self.STATES:
            config = self.STATES[state]
            icon = config['icon']
            color = config['color']
            
            self.configure(
                text=f"{icon} {message}",
                fg=color
            )
        else:
            self.configure(text=message, fg=COLORS['text'])


__all__ = [
    'RoundedContainer',
    'RoundedButton', 
    'ScrollableFrame',
    'StatusIndicator'
]