"""
Definición de estilos visuales para la interfaz gráfica.
"""

# Paleta de colores moderna
COLORS = {
    'primary': '#FF4081',      # Rosa fuerte
    'secondary': '#FF80AB',    # Rosa medio
    'accent': '#FF94C2',       # Rosa claro
    'accent_light': '#FFE0F0', # Rosa muy claro para fondos
    'primary_dark': '#D81B60', # Rosa fuerte oscuro (para hover)
    'background': '#FFFFFF',   # Blanco
    'card_bg': '#F8F9FA',     # Gris muy claro
    'sidebar_bg': '#F0F2F5',  # Gris claro
    'text': '#212529',        # Negro suave
    'text_light': '#6C757D',  # Gris medio
    'success': '#4CAF50',     # Verde
    'warning': '#FFC107',     # Amarillo
    'info': '#2196F3',        # Azul
    'border': '#DEE2E6',       # Gris borde
    'rosita': '#FCE4EC'        #exotic
}

# Constantes de diseño
DESIGN = {
    'border_radius': 14,      # Radio de bordes redondeados
    'card_padding': 3,        # Padding mínimo de las tarjetas
    'sidebar_width': 200,     # Ancho mínimo del sidebar
    'header_height': 70,      # Altura del header
    'button_height': 30,      # Altura mínima de los botones
    'section_height': 25,     # Altura para títulos de sección
    'container_height': 130,   # Altura para contenedores de botones
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
        'width': DESIGN['sidebar_width']
    },
    'card': {
        'background': COLORS['card_bg'],
        'relief': 'flat',
        'borderwidth': 0,
        'highlightbackground': COLORS['border'],
        'highlightthickness': 1,
        'bd': 0
    },
    'section_title': {
        'background': COLORS['sidebar_bg'],
        'foreground': COLORS['text'],
        'font': ('Segoe UI', 9, 'bold'),
        'pady': 2,
        'height': DESIGN['section_height']
    },
    'segmented_button': {
        'background': COLORS['background'],
        'foreground': COLORS['text'],
        'activebackground': COLORS['primary'],
        'activeforeground': 'white',
        'font': ('Segoe UI', 10),
        'relief': 'ridge',
        'border': 0,
        'borderwidth': 0,
        'highlightthickness': 1,
        'highlightbackground': COLORS['border'],
        'padx': 15,
        'pady': 8,
        'width': 15,
        'cursor': 'hand2',
        'bd': 0
    },
    'action_button': {
        'background': COLORS['primary'],
        'foreground': 'white',
        'activebackground': COLORS['primary_dark'],
        'activeforeground': 'white',
        'font': ('Segoe UI', 10, 'bold'),
        'relief': 'ridge',
        'borderwidth': 0,
        'highlightthickness': 0,
        'padx': 20,
        'pady': 10,
        'cursor': 'hand2',
        'bd': 0
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
