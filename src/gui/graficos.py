
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


def mostrar_analisis_roc(frame_grafico, resultados_rgb, resultados_pca, criterio):
    """
    *** VISUALIZACIÓN ROC INTEGRADA EN GUI ***
    
    Muestra análisis ROC comparativo directamente en el área principal
    de la interfaz gráfica, siguiendo los requisitos de la pauta.
    
    REQUISITOS IMPLEMENTADOS:
    1. ✅ Curvas ROC para ambos clasificadores (RGB y PCA)
    2. ✅ Valores AUC mostrados en leyenda
    3. ✅ Puntos de operación marcados según criterio
    4. ✅ Línea diagonal de referencia (clasificador aleatorio)
    5. ✅ Interpretación visual clara con colores distintivos
    
    Args:
        frame_grafico: Frame de tkinter donde mostrar el gráfico
        resultados_rgb: Diccionario con resultados ROC del clasificador RGB
        resultados_pca: Diccionario con resultados ROC del clasificador PCA  
        criterio: Criterio usado para selección de punto de operación
    """
    # Limpiar frame anterior
    for widget in frame_grafico.winfo_children():
        widget.destroy()
    
    # Crear figura con estilo de la interfaz
    fig = Figure(**PLOT_STYLE['figure'])
    ax = fig.add_subplot(111)
    
    # Configurar estilo
    ax.set_facecolor(PLOT_STYLE['axes']['facecolor'])
    ax.grid(True, color=PLOT_STYLE['axes']['grid_color'], linestyle='--', alpha=0.3)
    
    for spine in ax.spines.values():
        spine.set_color(PLOT_STYLE['axes']['spines_color'])
        spine.set_linewidth(0.5)
    
    # *** EXTRAER DATOS ROC ***
    roc_rgb = resultados_rgb['resultados_roc']
    roc_pca = resultados_pca['resultados_roc']
    
    punto_rgb = resultados_rgb['punto_operacion']
    punto_pca = resultados_pca['punto_operacion']
    
    auc_rgb = resultados_rgb['auc']
    auc_pca = resultados_pca['auc']
    
    # *** GRAFICAR CURVAS ROC ***
    # Curva RGB
    ax.plot(roc_rgb['fpr'], roc_rgb['tpr'], 
            color=COLORS['primary'], linewidth=3, alpha=0.8,
            label=f'RGB (AUC = {auc_rgb:.3f})')
    
    # Curva PCA
    ax.plot(roc_pca['fpr'], roc_pca['tpr'], 
            color=COLORS['success'], linewidth=3, alpha=0.8,
            label=f'PCA (AUC = {auc_pca:.3f})')
    
    # *** MARCAR PUNTOS DE OPERACIÓN ***
    # Punto RGB
    ax.plot(punto_rgb.fpr, punto_rgb.tpr, 
            marker='o', markersize=12, color=COLORS['primary'],
            markerfacecolor='white', markeredgewidth=3, markeredgecolor=COLORS['primary'],
            label=f'Punto RGB ({criterio.title()})')
    
    # Punto PCA  
    ax.plot(punto_pca.fpr, punto_pca.tpr,
            marker='s', markersize=12, color=COLORS['success'], 
            markerfacecolor='white', markeredgewidth=3, markeredgecolor=COLORS['success'],
            label=f'Punto PCA ({criterio.title()})')
    
    # *** LÍNEA DE REFERENCIA ***
    ax.plot([0, 1], [0, 1], 
            color=COLORS['secondary'], linestyle='--', linewidth=2, alpha=0.6,
            label='Clasificador Aleatorio')
    
    # *** CONFIGURACIÓN DE EJES ***
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel('Tasa de Falsos Positivos (FPR)', **PLOT_STYLE['labels'])
    ax.set_ylabel('Tasa de Verdaderos Positivos (TPR)', **PLOT_STYLE['labels'])
    
    # *** TÍTULO INFORMATIVO ***
    mejor_metodo = "PCA" if auc_pca > auc_rgb else "RGB"
    diferencia_auc = abs(auc_pca - auc_rgb)
    
    titulo = f"Análisis ROC - Detección de Melanoma\\n"
    titulo += f"Criterio: {criterio.title()} | Mejor método: {mejor_metodo} (+{diferencia_auc:.3f} AUC)"
    
    ax.set_title(titulo, **PLOT_STYLE['title'], pad=20)
    
    # *** LEYENDA PERSONALIZADA ***
    legend = ax.legend(loc='lower right', 
                      fancybox=True, shadow=True, 
                      framealpha=0.9, facecolor=COLORS['card_bg'],
                      edgecolor=COLORS['border'])
    legend.get_frame().set_facecolor(COLORS['card_bg'])
    for text in legend.get_texts():
        text.set_color(COLORS['text'])
    
    # *** AGREGAR ANOTACIONES INFORMATIVAS ***
    # Anotación para punto RGB
    ax.annotate(f'Sens: {punto_rgb.tpr:.2f}\\nSpec: {punto_rgb.tnr:.2f}',
                xy=(punto_rgb.fpr, punto_rgb.tpr),
                xytext=(punto_rgb.fpr + 0.15, punto_rgb.tpr - 0.15),
                bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS['primary'], alpha=0.7),
                fontsize=8, color='white', weight='bold',
                arrowprops=dict(arrowstyle='->', color=COLORS['primary'], lw=1.5))
    
    # Anotación para punto PCA
    ax.annotate(f'Sens: {punto_pca.tpr:.2f}\\nSpec: {punto_pca.tnr:.2f}',
                xy=(punto_pca.fpr, punto_pca.tpr),
                xytext=(punto_pca.fpr + 0.15, punto_pca.tpr + 0.15),
                bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS['success'], alpha=0.7),
                fontsize=8, color='white', weight='bold',
                arrowprops=dict(arrowstyle='->', color=COLORS['success'], lw=1.5))
    
    fig.tight_layout()
    
    # *** MOSTRAR EN INTERFAZ ***
    canvas = FigureCanvasTkAgg(fig, frame_grafico)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    # *** AGREGAR PANEL DE INFORMACIÓN TEXTUAL ***
    info_frame = tk.Frame(frame_grafico, bg=COLORS['card_bg'], relief='ridge', bd=1)
    info_frame.pack(fill=tk.X, padx=5, pady=5)
    
    # Título del panel
    titulo_info = tk.Label(info_frame, 
                          text="📊 RESUMEN ANÁLISIS ROC",
                          font=('Segoe UI', 10, 'bold'),
                          fg=COLORS['primary'],
                          bg=COLORS['card_bg'])
    titulo_info.pack(pady=5)
    
    # Información comparativa
    info_text = f"""
🔴 RGB: AUC = {auc_rgb:.3f} | Sens = {punto_rgb.tpr:.3f} | Spec = {punto_rgb.tnr:.3f} | Youden = {punto_rgb.youden_index:.3f}
🟢 PCA: AUC = {auc_pca:.3f} | Sens = {punto_pca.tpr:.3f} | Spec = {punto_pca.tnr:.3f} | Youden = {punto_pca.youden_index:.3f}

💡 INTERPRETACIÓN: El método {mejor_metodo} muestra {"mejor" if diferencia_auc > 0.01 else "similar"} capacidad discriminativa
    """.strip()
    
    info_label = tk.Label(info_frame,
                         text=info_text,
                         font=('Segoe UI', 9),
                         fg=COLORS['text'],
                         bg=COLORS['card_bg'],
                         justify='left')
    info_label.pack(padx=10, pady=5)
