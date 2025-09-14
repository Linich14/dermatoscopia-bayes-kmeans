
"""
Funciones para crear y actualizar gráficos (histogramas, barras, pastel, etc.).
"""

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from .styles import COLORS, PLOT_STYLE
import numpy as np

def mostrar_histograma(frame_grafico, datos, area, canal):
    """
    Dibuja el gráfico de histograma y muestra estadísticas para el área y canal seleccionados en el frame dado.
    """
    # Limpiar frame
    for widget in frame_grafico.winfo_children():
        widget.destroy()

    # Crear figura con estilo
    fig = Figure(**PLOT_STYLE['figure'])
    ax = fig.add_subplot(111)
    
    # Configurar el estilo de los ejes
    ax.set_facecolor(PLOT_STYLE['axes']['facecolor'])
    ax.grid(PLOT_STYLE['axes']['grid'], color=PLOT_STYLE['axes']['grid_color'], linestyle='--', alpha=0.7)
    
    # Personalizar el aspecto de los bordes
    for spine in ax.spines.values():
        spine.set_color(PLOT_STYLE['axes']['spines_color'])
        spine.set_linewidth(0.5)

    # Dibujar histograma con colores rosa
    hist = datos['histograma']
    # Diferentes tonos de rosa para cada canal
    canal_colors = {
        'R': '#FF1493',  # Rosa brillante
        'G': '#FF69B4',  # Rosa medio
        'B': '#FFB6C1'   # Rosa claro
    }
    bar_color = canal_colors[canal]
    bars = ax.bar(range(len(hist)), 
                 hist,
                 color=bar_color,
                 alpha=0.7,
                 edgecolor=COLORS['primary'])

    # Configurar títulos y etiquetas
    ax.set_title(f"Distribución de intensidades - Canal {canal}\nÁrea: {area.capitalize()}",
                 **PLOT_STYLE['title'])
    ax.set_xlabel("Intensidad (normalizada)", **PLOT_STYLE['labels'])
    ax.set_ylabel("Frecuencia", **PLOT_STYLE['labels'])

    # Mostrar estadísticos en un cuadro elegante
    media = datos['media']
    std = datos['std']
    stats_text = (f"Estadísticas\n"
                 f"---------------\n"
                 f"Media: {media:.4f}\n"
                 f"Desv. Est.: {std:.4f}")
    
    ax.text(0.95, 0.95, 
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            family='Segoe UI',
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(
                facecolor='white',
                edgecolor=COLORS['text_light'],
                boxstyle='round,pad=0.5',
                alpha=0.9
            ))

    # Ajustar los márgenes
    fig.tight_layout(pad=2.0)

    # Crear y mostrar el canvas
    canvas = FigureCanvasTkAgg(fig, master=frame_grafico)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)
