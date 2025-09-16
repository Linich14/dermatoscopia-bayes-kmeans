
"""
Funciones para crear y actualizar gráficos (histogramas, barras, pastel, etc.).
"""

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from .styles import COLORS, PLOT_STYLE
import numpy as np
import tkinter as tk

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

    # Dibujar histograma con colores de la paleta original
    hist = datos['histograma']
    # Usar colores de la paleta original para cada canal
    canal_colors = {
        'R': COLORS['primary'],    # Rosa fuerte principal
        'G': COLORS['success'],    # Verde
        'B': COLORS['info']        # Azul
    }
    bar_color = canal_colors[canal]
    bars = ax.bar(range(len(hist)), 
                 hist,
                 color=bar_color,
                 alpha=0.7,
                 edgecolor=COLORS['primary'])

    # Configurar títulos y etiquetas
    ax.set_title(f"Distribución de intensidades - Canal {canal} Área: {area.capitalize()}",
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


def mostrar_resultado_clasificacion(parent_window, imagen_original, mascara_pred, nombre_archivo):
    """
    Muestra el resultado de clasificación en una ventana separada.
    
    Args:
        parent_window: Ventana padre
        imagen_original: Imagen original como array numpy
        mascara_pred: Máscara predicha como array numpy
        nombre_archivo: Nombre del archivo procesado
    """
    # Crear ventana de resultados
    resultado_window = tk.Toplevel(parent_window)
    resultado_window.title(f"Resultado de Clasificación - {nombre_archivo}")
    resultado_window.geometry("1000x600")
    resultado_window.configure(bg=COLORS['background'])
    
    # Crear figura con subplots usando el estilo configurado
    fig = Figure(figsize=(12, 5), dpi=100)
    fig.patch.set_facecolor(COLORS['background'])
    
    # Imagen original
    ax1 = fig.add_subplot(131)
    ax1.imshow(imagen_original)
    ax1.set_title('Imagen Original', fontsize=12, fontweight='bold', color=COLORS['text'])
    ax1.axis('off')
    
    # Máscara predicha
    ax2 = fig.add_subplot(132)
    ax2.imshow(mascara_pred, cmap='hot', alpha=0.8)
    ax2.set_title('Segmentación Predicha', fontsize=12, fontweight='bold', color=COLORS['text'])
    ax2.axis('off')
    
    # Superposición
    ax3 = fig.add_subplot(133)
    ax3.imshow(imagen_original)
    ax3.imshow(mascara_pred, cmap='hot', alpha=0.4)
    ax3.set_title('Superposición', fontsize=12, fontweight='bold', color=COLORS['text'])
    ax3.axis('off')
    
    fig.tight_layout()
    
    # Mostrar en la ventana
    canvas = FigureCanvasTkAgg(fig, resultado_window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Frame de botones
    button_frame = tk.Frame(resultado_window, bg=COLORS['background'])
    button_frame.pack(fill=tk.X, pady=10)
    
    # Botón para guardar
    save_button = tk.Button(button_frame,
                           text="💾 Guardar Resultado",
                           command=lambda: _guardar_resultado(fig, nombre_archivo),
                           bg=COLORS['primary'],
                           fg='white',
                           font=('Segoe UI', 10, 'bold'),
                           padx=20, pady=8)
    save_button.pack(side=tk.LEFT, padx=10)
    
    # Botón para cerrar
    close_button = tk.Button(button_frame,
                            text="❌ Cerrar",
                            command=resultado_window.destroy,
                            bg=COLORS['card_bg'],
                            fg=COLORS['text'],
                            font=('Segoe UI', 10),
                            padx=20, pady=8)
    close_button.pack(side=tk.RIGHT, padx=10)


def _guardar_resultado(fig, nombre_archivo):
    """Helper para guardar el resultado de clasificación."""
    from tkinter import filedialog, messagebox
    import os
    
    # Obtener directorio para guardar
    directorio = filedialog.askdirectory(title="Seleccionar directorio para guardar")
    if directorio:
        try:
            # Crear nombre de archivo
            nombre_base = os.path.splitext(nombre_archivo)[0]
            ruta_completa = os.path.join(directorio, f"{nombre_base}_resultado.png")
            
            # Guardar figura
            fig.savefig(ruta_completa, dpi=300, bbox_inches='tight', 
                       facecolor=COLORS['background'])
            
            messagebox.showinfo("Éxito", f"Resultado guardado en:\n{ruta_completa}")
        except Exception as e:
            messagebox.showerror("Error", f"Error al guardar: {str(e)}")
