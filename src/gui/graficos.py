
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
    de la interfaz gráfica, siguiendo los requisitos del proyecto.
    
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


def mostrar_resultados_kmeans(frame_grafico, resultados_kmeans, mejor_combinacion):
    """
    Muestra los resultados del análisis K-Means con visualización de clusters.
    
    Args:
        frame_grafico: Frame donde mostrar el gráfico
        resultados_kmeans: Dict con todos los resultados del análisis
        mejor_combinacion: Objeto con la mejor combinación encontrada
    """
    # Limpiar frame
    for widget in frame_grafico.winfo_children():
        widget.destroy()
    
    if not resultados_kmeans or not mejor_combinacion:
        # Mostrar mensaje de error
        error_label = tk.Label(
            frame_grafico,
            text="ERROR: No hay resultados K-Means disponibles",
            font=('Segoe UI', 12, 'bold'),
            fg=COLORS['error'],
            bg=COLORS['background']
        )
        error_label.pack(expand=True)
        return
    
    # *** CREAR SCROLL CONTAINER ***
    from .components import ScrollableFrame
    scroll_container = ScrollableFrame(frame_grafico)
    scroll_container.pack(fill='both', expand=True, padx=10, pady=10)
    content_frame = scroll_container.scrollable_frame
    content_frame.configure(bg=COLORS['background'])
    
    # *** TÍTULO PRINCIPAL ***
    titulo_principal = tk.Label(content_frame,
                               text="📊 RESULTADOS ANÁLISIS K-MEANS",
                               font=('Segoe UI', 14, 'bold'),
                               fg=COLORS['primary'],
                               bg=COLORS['background'])
    titulo_principal.pack(pady=(0, 20))
    
    # *** CREAR FIGURA CON SUBPLOTS MÁS COMPACTA ***
    fig = Figure(figsize=(12, 8), facecolor=COLORS['background'])
    
    # Panel principal con comparación de métricas
    ax_main = fig.add_subplot(2, 2, (1, 2))
    
    # Obtener datos de todas las combinaciones desde mejor_combinacion y otros resultados
    combinaciones = []
    scores = []
    silhouette_scores = []
    
    # Si hay resultados disponibles y mejor_combinacion está definida
    if mejor_combinacion and hasattr(mejor_combinacion, 'nombre_combinacion'):
        # Intentar obtener datos del reporte completo
        try:
            # Si resultados_kmeans tiene la estructura de reporte completo
            if resultados_kmeans and isinstance(resultados_kmeans, dict):
                if 'todas_las_combinaciones' in resultados_kmeans:
                    # Extraer de estructura de reporte completo
                    todas_combinaciones = resultados_kmeans['todas_las_combinaciones']
                    
                    for comb in todas_combinaciones:
                        combinaciones.append(comb['nombre'])
                        scores.append(comb['score_total'])
                        # Buscar silhouette en metricas_promedio
                        silhouette_val = 0
                        if 'metricas_promedio' in comb:
                            # Buscar diferentes variantes del nombre
                            metricas = comb['metricas_promedio']
                            silhouette_val = (metricas.get('silhouette', 0) or 
                                            metricas.get('silhouette_score', 0))
                        silhouette_scores.append(silhouette_val)
                else:
                    # Fallback: mostrar solo la mejor combinación
                    combinaciones = [mejor_combinacion.nombre_combinacion]
                    scores = [mejor_combinacion.score_total]
                    silhouette_scores = [mejor_combinacion.metricas_promedio.get('silhouette', 0)]
            else:
                # Fallback: mostrar solo la mejor combinación
                combinaciones = [mejor_combinacion.nombre_combinacion]
                scores = [mejor_combinacion.score_total]
                silhouette_scores = [mejor_combinacion.metricas_promedio.get('silhouette', 0)]
                
        except Exception as e:
            print(f"Error extrayendo datos para gráfico: {e}")
            # Fallback: mostrar solo la mejor combinación
            combinaciones = [mejor_combinacion.nombre_combinacion]
            scores = [mejor_combinacion.score_total]
            silhouette_scores = [mejor_combinacion.metricas_promedio.get('silhouette', 0)]
    
    if scores and len(scores) > 0:
        x_pos = np.arange(len(combinaciones))
        width = 0.35
        
        bars1 = ax_main.bar(x_pos - width/2, scores, width, 
                           label='Score Total', color=COLORS['primary'], alpha=0.8)
        bars2 = ax_main.bar(x_pos + width/2, silhouette_scores, width,
                           label='Silhouette Score', color=COLORS['success'], alpha=0.8)
        
        ax_main.set_xlabel('Combinaciones de Características', **PLOT_STYLE['labels'])
        ax_main.set_ylabel('Score', **PLOT_STYLE['labels'])
        ax_main.set_title('Comparacion de Combinaciones K-Means', **PLOT_STYLE['title'])
        ax_main.set_xticks(x_pos)
        ax_main.set_xticklabels([c.replace('_', '\n') for c in combinaciones], rotation=45, ha='right')
        ax_main.legend()
        ax_main.grid(True, alpha=0.3)
        
        # Marcar la mejor combinación si hay múltiples
        if len(scores) > 1:
            mejor_idx = scores.index(max(scores))
            ax_main.annotate('>>> MEJOR <<<', 
                            xy=(mejor_idx, scores[mejor_idx]), 
                            xytext=(mejor_idx, scores[mejor_idx] + 0.1),
                            ha='center', fontweight='bold', color=COLORS['primary'],
                            arrowprops=dict(arrowstyle='->', color=COLORS['primary']))
    else:
        # Si no hay datos, mostrar mensaje informativo
        ax_main.text(0.5, 0.5, 'No hay datos suficientes\npara mostrar comparación', 
                    transform=ax_main.transAxes, ha='center', va='center',
                    fontsize=14, color=COLORS['text'])
        ax_main.set_title('Comparacion de Combinaciones K-Means', **PLOT_STYLE['title'])
    
    # *** GRÁFICO DE DISTRIBUCIÓN DE CLUSTERS ***
    ax_clusters = fig.add_subplot(2, 2, 3)
    
    # Obtener datos de distribución de K del ENTRENAMIENTO (no del test)
    mejor_resultado = mejor_combinacion
    if hasattr(mejor_resultado, 'k_distribution_entrenamiento') and mejor_resultado.k_distribution_entrenamiento:
        clusters_counts = mejor_resultado.k_distribution_entrenamiento
        
        if clusters_counts:
            clusters_list = list(clusters_counts.keys())
            counts_list = list(clusters_counts.values())
            
            colors_clusters = [COLORS['primary'], COLORS['success'], COLORS['info'], COLORS['warning']]
            ax_clusters.pie(counts_list, labels=[f'{k} clusters' for k in clusters_list], 
                          autopct='%1.1f%%', colors=colors_clusters[:len(clusters_list)])
            ax_clusters.set_title('Distribucion Optima de Clusters\n(Durante Entrenamiento)', **PLOT_STYLE['title'])
    else:
        # Fallback a datos de test si no hay datos de entrenamiento (retrocompatibilidad)
        if hasattr(mejor_resultado, 'resultados_imagenes') and mejor_resultado.resultados_imagenes:
            clusters_counts = {}
            for img_resultado in mejor_resultado.resultados_imagenes:
                n_clusters = img_resultado.n_clusters
                if n_clusters not in clusters_counts:
                    clusters_counts[n_clusters] = 0
                clusters_counts[n_clusters] += 1
            
            if clusters_counts:
                clusters_list = list(clusters_counts.keys())
                counts_list = list(clusters_counts.values())
                
                colors_clusters = [COLORS['primary'], COLORS['success'], COLORS['info'], COLORS['warning']]
                ax_clusters.pie(counts_list, labels=[f'{k} clusters' for k in clusters_list], 
                              autopct='%1.1f%%', colors=colors_clusters[:len(clusters_list)])
                ax_clusters.set_title('Distribucion Optima de Clusters\n(Test)', **PLOT_STYLE['title'])
    
    # *** MÉTRICAS DETALLADAS ***
    ax_metricas = fig.add_subplot(2, 2, 4)
    ax_metricas.axis('off')
    
    if mejor_combinacion:
        # Texto más compacto y organizado
        metricas_text = f"""MEJOR COMBINACION K-MEANS

{mejor_combinacion.nombre_combinacion}
Score: {mejor_combinacion.score_total:.3f}

METRICAS PROMEDIO:"""
        
        for metrica, valor in mejor_combinacion.metricas_promedio.items():
            emoji = {'silhouette_score': '>', 'calinski_harabasz': '+', 'davies_bouldin': '-'}.get(metrica, '*')
            # Acortar nombres de métricas para evitar corte
            metrica_names = {
                'silhouette_score': 'Silhouette',
                'calinski_harabasz': 'Calinski H.',
                'davies_bouldin': 'Davies B.',
                'inercia': 'Inercia'
            }
            metrica_corta = metrica_names.get(metrica, metrica.replace('_', ' ').title())
            metricas_text += f"\n{emoji} {metrica_corta}: {valor:.3f}"
        
        # Información adicional más compacta
        mejor_img = getattr(mejor_combinacion, 'mejor_imagen', 'N/A')
        peor_img = getattr(mejor_combinacion, 'peor_imagen', 'N/A')
        total_imgs = len(getattr(mejor_combinacion, 'resultados_imagenes', []))
        
        # Acortar nombres de imágenes si son muy largos
        if len(mejor_img) > 15:
            mejor_img = mejor_img[:12] + "..."
        if len(peor_img) > 15:
            peor_img = peor_img[:12] + "..."
        
        metricas_text += f"""

Mejor imagen: {mejor_img}
Imagen mas dificil: {peor_img}

Total procesadas: {total_imgs}"""
        
        # Usar fuente más pequeña y mejor posicionamiento
        ax_metricas.text(0.02, 0.98, metricas_text, 
                        transform=ax_metricas.transAxes,
                        fontsize=8, family='monospace',
                        verticalalignment='top', horizontalalignment='left',
                        bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor=COLORS['card_bg'], 
                                edgecolor=COLORS['primary'],
                                alpha=0.9))
    
    # Ajustar espaciado más compacto
    fig.tight_layout(pad=2.0)
    
    # *** CREAR CANVAS Y MOSTRAR EN SCROLL ***
    canvas_frame = tk.Frame(content_frame, bg=COLORS['background'])
    canvas_frame.pack(fill='both', expand=True, pady=(0, 20))
    
    canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill='both', expand=True)
    
    # *** SEPARADOR ***
    separador = tk.Frame(content_frame, height=2, bg=COLORS['border'])
    separador.pack(fill='x', pady=20)
    
    # *** PANEL DE INFORMACIÓN TEXTUAL ADICIONAL ***
    info_frame = tk.Frame(content_frame, bg=COLORS['card_bg'], relief='solid', borderwidth=1)
    info_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
    
    titulo_info = tk.Label(info_frame, 
                          text="📋 RESUMEN DEL ANÁLISIS",
                          font=('Segoe UI', 12, 'bold'),
                          fg=COLORS['primary'],
                          bg=COLORS['card_bg'])
    titulo_info.pack(pady=10)
    
    if mejor_combinacion:
        # *** INFORMACIÓN PRINCIPAL ***
        info_principal_frame = tk.Frame(info_frame, bg=COLORS['card_bg'])
        info_principal_frame.pack(fill='x', padx=20, pady=(0, 15))
        
        # Mejor combinación info
        mejor_info = tk.Label(info_principal_frame,
                             text=f"🏆 MEJOR COMBINACIÓN: {mejor_combinacion.nombre_combinacion}",
                             font=('Segoe UI', 11, 'bold'),
                             fg=COLORS['success'],
                             bg=COLORS['card_bg'])
        mejor_info.pack(anchor='w', pady=2)
        
        score_info = tk.Label(info_principal_frame,
                             text=f"📊 Score Total: {mejor_combinacion.score_total:.3f}",
                             font=('Segoe UI', 10),
                             fg=COLORS['text'],
                             bg=COLORS['card_bg'])
        score_info.pack(anchor='w', pady=2)
        
        interpretacion = "Excelente" if mejor_combinacion.score_total > 0.7 else "Buena" if mejor_combinacion.score_total > 0.5 else "Moderada"
        interp_info = tk.Label(info_principal_frame,
                              text=f"🎯 Calidad: {interpretacion} separación de clusters",
                              font=('Segoe UI', 10),
                              fg=COLORS['primary'],
                              bg=COLORS['card_bg'])
        interp_info.pack(anchor='w', pady=2)
        
        # *** MÉTRICAS DETALLADAS ***
        metricas_frame = tk.Frame(info_frame, bg=COLORS['accent_light'], relief='ridge', borderwidth=1)
        metricas_frame.pack(fill='x', padx=20, pady=(0, 15))
        
        metricas_titulo = tk.Label(metricas_frame,
                                  text="📈 MÉTRICAS PROMEDIO",
                                  font=('Segoe UI', 10, 'bold'),
                                  fg=COLORS['primary'],
                                  bg=COLORS['accent_light'])
        metricas_titulo.pack(pady=5)
        
        metricas_text = ""
        for metrica, valor in mejor_combinacion.metricas_promedio.items():
            emoji = {'silhouette_score': '🔸', 'calinski_harabasz': '🔹', 'davies_bouldin': '🔺'}.get(metrica, '•')
            metricas_text += f"{emoji} {metrica.replace('_', ' ').title()}: {valor:.3f}\n"
        
        if hasattr(mejor_combinacion, 'mejor_imagen') and mejor_combinacion.mejor_imagen:
            metricas_text += f"\n🎯 Mejor imagen: {mejor_combinacion.mejor_imagen}"
        if hasattr(mejor_combinacion, 'peor_imagen') and mejor_combinacion.peor_imagen:
            metricas_text += f"\n⚠️ Imagen más difícil: {mejor_combinacion.peor_imagen}"
        if hasattr(mejor_combinacion, 'resultados_imagenes') and mejor_combinacion.resultados_imagenes:
            metricas_text += f"\n📊 Total procesadas: {len(mejor_combinacion.resultados_imagenes)}"
        
        metricas_label = tk.Label(metricas_frame,
                                 text=metricas_text,
                                 font=('Segoe UI', 9),
                                 fg=COLORS['text'],
                                 bg=COLORS['accent_light'],
                                 justify='left')
        metricas_label.pack(padx=10, pady=(0, 10))
