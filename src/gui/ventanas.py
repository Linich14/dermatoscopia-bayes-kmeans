"""
Ventana principal de la interfaz gráfica para visualización de resultados.
Incluye gráfico de barras/pastel, galería de imágenes clasificadas y botón de exportar.
"""
import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class VentanaPrincipal(tk.Tk):
    """
    Ventana principal de la interfaz gráfica.
    Permite visualizar gráficos de distribución y galería de imágenes clasificadas.
    Incluye botón para exportar resultados.
    """
    def __init__(self, stats_rgb):
        super().__init__()
        self.title("Exploración Estadística de Imágenes Dermatoscópicas")
        self.geometry("900x700")
        self.stats_rgb = stats_rgb
        # Selección de área y canal
        self.area_var = tk.StringVar(value='lesion')
        self.canal_var = tk.StringVar(value='R')
        frame_sel = ttk.Frame(self)
        frame_sel.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        ttk.Label(frame_sel, text="Área:").pack(side=tk.LEFT)
        area_menu = ttk.OptionMenu(frame_sel, self.area_var, 'lesion', 'lesion', 'sana', command=self.actualizar_histograma)
        area_menu.pack(side=tk.LEFT, padx=5)
        ttk.Label(frame_sel, text="Canal:").pack(side=tk.LEFT)
        canal_menu = ttk.OptionMenu(frame_sel, self.canal_var, 'R', 'R', 'G', 'B', command=self.actualizar_histograma)
        canal_menu.pack(side=tk.LEFT, padx=5)
        # Frame para el gráfico
        self.frame_grafico = ttk.Frame(self)
        self.frame_grafico.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        # Botón para exportar resultados
        # Inicializar gráfico
        self.actualizar_histograma()

    def actualizar_histograma(self, *args):
        """
        Actualiza el gráfico de histograma y muestra estadísticas para el área y canal seleccionados.
        """
        area = self.area_var.get()
        canal = self.canal_var.get()
        datos = self.stats_rgb[area][canal]
        from gui.graficos import mostrar_histograma
        mostrar_histograma(self.frame_grafico, datos, area, canal)


