"""
Definición de estilos para la interfaz gráfica.
"""

# Paleta de colores moderna
COLORS = {
    'primary': '#FF4081',      # Rosa fuerte
    'secondary': '#FF80AB',    # Rosa medio
    'accent': '#FF94C2',       # Rosa claro
    'accent_light': '#FFE0F0', # Rosa muy claro para fondos
    'primary_dark': '#D81B60', # Rosa fuerte oscuro (para hover)
    'background': '#FCE4EC',   # Rosita como fondo general
    'card_bg': '#FCE4EC',     # Rosa muy claro para tarjetas
    'sidebar_bg': '#FCE4EC',   # Rosita para sidebar
    'text': '#212529',        # Negro suave
    'text_light': '#6C757D',  # Gris medio
    'success': '#4CAF50',     # Verde
    'warning': '#FFC107',     # Amarillo
    'info': '#2196F3',        # Azul
    'border': '#FF80AB',      # Borde rosa medio
    'dialog_bg': '#FCE4EC',   # Rosita para diálogos
    'content_bg': '#FCE4EC'    # Fondo para contenido
}

# Constantes de diseño
DESIGN = {
    'border_radius': 6,        # Radio de bordes redondeados
    'card_padding': 15,        # Padding mínimo de las tarjetas
    'sidebar_width': 400,      # Ancho mínimo del sidebar
    'sidebar_padding': 15,     # Padding interno del sidebar
    'content_width': 370,      # Ancho efectivo para contenido
    'header_height': 50,       # Altura del header
    'button_height': 35,       # Altura mínima de los botones
    'section_height': 30,      # Altura para títulos de sección
    'section_margin': 8,       # Margen entre secciones
    'container_height': 160,   # Altura para contenedores de botones
    'dialog_width': 700,       # Ancho para ventanas de diálogo
    'dialog_height': 700,      # Altura para ventanas de diálogo 
    'dialog_margin': 30,       # Margen interno para diálogos
    'dialog_content_width': 650,  # Ancho efectivo para contenido del diálogo
    'info_card_width': 620,    # Ancho para tarjetas de información
    'metrics_width': 580,      # Ancho para métricas
    'matrix_width': 400,       # Ancho para matriz de confusión
    'table_row_height': 28,    # Altura de filas en tablas
    'button_spacing': 6,       # Espacio entre botones
    'group_margin': 12         # Margen entre grupos de elementos
}

# Estilos para widgets
STYLES = {
    'root': {
        'background': COLORS['background']
    },
    'header': {
        'background': COLORS['primary'],
        'height': DESIGN['header_height']
    },
    'header_button': {
        'background': COLORS['primary'],
        'foreground': 'white',
        'activebackground': COLORS['primary_dark'],
        'activeforeground': 'white',
        'font': ('Segoe UI', 11),
        'border': 0,
        'padx': 15,
        'pady': 7,
        'cursor': 'hand2',
        'borderwidth': 0,
        'highlightthickness': 0,
        'relief': 'ridge',
        'bd': 0
    },
    'sidebar': {
        'background': COLORS['sidebar_bg'],
        'padx': DESIGN['sidebar_padding'],
        'pady': DESIGN['sidebar_padding']
    },
    'card': {
        'background': COLORS['card_bg'],
        'relief': 'solid',
        'borderwidth': 1,
        'highlightbackground': COLORS['border'],
        'highlightthickness': 1,
        'bd': 0,
        'padx': DESIGN['card_padding'],
        'pady': DESIGN['card_padding'],
        'width': DESIGN['content_width']
    },
    'section_title': {
        'background': COLORS['sidebar_bg'],
        'foreground': COLORS['text'],
        'font': ('Segoe UI', 10, 'bold'),
        'pady': 8,
        'padx': DESIGN['card_padding'],
        'anchor': 'w',
        'height': DESIGN['section_height'],
        'width': DESIGN['content_width']
    },
    'segmented_button': {
        'background': COLORS['background'],
        'foreground': COLORS['text'],
        'activebackground': COLORS['primary'],
        'activeforeground': 'white',
        'font': ('Segoe UI', 10),
        'relief': 'flat',
        'border': 0,
        'borderwidth': 1,
        'highlightthickness': 0,
        'highlightbackground': COLORS['border'],
        'padx': DESIGN['card_padding'],
        'pady': 5,
        'width': DESIGN['content_width'] - (2 * DESIGN['card_padding']),
        'cursor': 'hand2',
        'bd': 0,
        'anchor': 'center'
    },
    'action_button': {
        'background': COLORS['primary'],
        'foreground': 'white',
        'activebackground': COLORS['primary_dark'],
        'activeforeground': 'white',
        'font': ('Segoe UI', 10, 'bold'),
        'relief': 'flat',
        'borderwidth': 1,
        'highlightthickness': 1,
        'highlightbackground': COLORS['border'],
        'padx': 10,
        'pady': 8,
        'cursor': 'hand2',
        'bd': 0,
        'width': 25  # Ancho fijo para mejor consistencia
    },
    'secondary_button': {
        'background': COLORS['card_bg'],
        'foreground': COLORS['text'],
        'activebackground': COLORS['border'],
        'activeforeground': COLORS['text'],
        'font': ('Segoe UI', 10),
        'relief': 'ridge',
        'borderwidth': 0,
        'highlightthickness': 1,
        'highlightbackground': COLORS['border'],
        'padx': 15,
        'pady': 8,
        'cursor': 'hand2',
        'bd': 0
    },
    'dialog': {
        'background': COLORS['dialog_bg'],
        'relief': 'flat',
        'borderwidth': 1,
        'highlightbackground': COLORS['border'],
        'highlightthickness': 1,
        'padx': DESIGN['dialog_margin'],
        'pady': DESIGN['dialog_margin'],
        'width': DESIGN['dialog_width'],
        'height': DESIGN['dialog_height'],
        'minwidth': 650,
        'minheight': 500
    },
    'dialog_title': {
        'background': COLORS['primary'],
        'foreground': 'white',
        'font': ('Segoe UI', 14, 'bold'),
        'pady': 12,
        'padx': 20,
        'anchor': 'center',
        'width': 40
    },
    'dialog_text': {
        'background': COLORS['dialog_bg'],
        'foreground': COLORS['text'],
        'font': ('Segoe UI', 12),
        'pady': 12,
        'padx': 25,
        'justify': 'left',
        'anchor': 'w',
        'wraplength': DESIGN['dialog_content_width'] - 60
    },
    'dialog_label': {
        'background': COLORS['dialog_bg'],
        'foreground': COLORS['text'],
        'font': ('Segoe UI', 11),
        'pady': 5,
        'padx': 10,
        'anchor': 'w'
    },
    'dialog_section': {
        'background': COLORS['dialog_bg'],
        'foreground': COLORS['primary'],
        'font': ('Segoe UI', 12, 'bold'),
        'pady': 15,
        'padx': 20,
        'anchor': 'center'
    },
    'info_card': {
        'background': COLORS['card_bg'],
        'relief': 'solid',
        'borderwidth': 1,
        'highlightbackground': COLORS['border'],
        'highlightthickness': 1,
        'padx': 20,
        'pady': 15,
        'width': DESIGN['info_card_width']
    },
    'metrics_card': {
        'background': COLORS['card_bg'],
        'relief': 'solid',
        'borderwidth': 1,
        'highlightbackground': COLORS['border'],
        'highlightthickness': 1,
        'padx': 25,
        'pady': 15,
        'width': DESIGN['metrics_width']
    },
    'matrix_card': {
        'background': COLORS['card_bg'],
        'relief': 'solid',
        'borderwidth': 1,
        'highlightbackground': COLORS['border'],
        'highlightthickness': 1,
        'padx': 20,
        'pady': 15,
        'width': DESIGN['matrix_width']
    },
    'card_title': {
        'background': COLORS['card_bg'],
        'foreground': COLORS['primary'],
        'font': ('Segoe UI', 13, 'bold'),
        'pady': 10,
        'padx': 15,
        'anchor': 'center'
    },
    'metrics_text': {
        'background': COLORS['card_bg'],
        'foreground': COLORS['text'],
        'font': ('Consolas', 11),
        'justify': 'left',
        'pady': 8,
        'padx': 15,
        'anchor': 'w'
    },
    'matrix_text': {
        'background': COLORS['card_bg'],
        'foreground': COLORS['text'],
        'font': ('Consolas', 11),
        'justify': 'center',
        'pady': 10,
        'padx': 20,
        'anchor': 'center'
    },
    'interpretation_container': {
        'background': COLORS['card_bg'],
        'relief': 'solid',
        'borderwidth': 1,
        'highlightbackground': COLORS['border'],
        'highlightthickness': 1,
        'padx': 25,
        'pady': 15
    },
    'interpretation_title': {
        'background': COLORS['card_bg'],
        'foreground': COLORS['primary'],
        'font': ('Segoe UI', 13, 'bold'),
        'pady': 10,
        'anchor': 'center'
    },
    'interpretation_text': {
        'background': COLORS['card_bg'],
        'foreground': COLORS['text'],
        'font': ('Segoe UI', 11),
        'justify': 'left',
        'pady': 8,
        'padx': 10,
        'anchor': 'w',
        'wraplength': 500
    },
    'metric_highlight': {
        'background': COLORS['card_bg'],
        'foreground': COLORS['primary'],
        'font': ('Segoe UI', 11, 'bold'),
        'pady': 5,
        'padx': 10
    },
    'confusion_matrix': {
        'background': COLORS['card_bg'],
        'foreground': COLORS['text'],
        'font': ('Consolas', 11),
        'pady': 12,
        'padx': 20,
        'justify': 'center',
        'anchor': 'center',
        'relief': 'solid',
        'borderwidth': 1,
        'highlightthickness': 1,
        'highlightbackground': COLORS['border']
    },
    'matrix_header': {
        'background': COLORS['primary'],
        'foreground': 'white',
        'font': ('Segoe UI', 12, 'bold'),
        'pady': 8,
        'padx': 15,
        'anchor': 'center'
    },
    'matrix_cell': {
        'background': COLORS['card_bg'],
        'foreground': COLORS['text'],
        'font': ('Consolas', 11, 'bold'),
        'width': 10,
        'pady': 5,
        'padx': 10,
        'anchor': 'center',
        'relief': 'groove',
        'borderwidth': 1
    },
    'dialog_button': {
        'background': COLORS['primary'],
        'foreground': 'white',
        'activebackground': COLORS['primary_dark'],
        'activeforeground': 'white',
        'font': ('Segoe UI', 10),
        'relief': 'flat',
        'borderwidth': 1,
        'highlightthickness': 1,
        'highlightbackground': COLORS['border'],
        'padx': 15,
        'pady': 5,
        'cursor': 'hand2',
        'width': 15
    }
}

# Configuración de gráficos
PLOT_STYLE = {
    'figure': {
        'facecolor': COLORS['background'],
        'figsize': (10, 6)
    },
    'axes': {
        'facecolor': 'white',
        'grid': True,
        'grid_color': '#E0E0E0',
        'spines_color': '#CCCCCC'
    },
    'bars': {
        'color': COLORS['secondary'],
        'alpha': 0.7,
        'edgecolor': COLORS['primary']
    },
    'title': {
        'color': COLORS['text'],
        'fontsize': 12,
        'fontweight': 'bold'
    },
    'labels': {
        'color': COLORS['text'],
        'fontsize': 10
    }
}

# Fuentes del sistema
FONTS = {
    'default': ('Segoe UI', 9),
    'heading': ('Segoe UI', 12, 'bold'),
    'button': ('Segoe UI', 9),
    'small': ('Segoe UI', 8),
    'monospace': ('Consolas', 9)
}
