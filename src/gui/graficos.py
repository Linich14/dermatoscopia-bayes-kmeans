
"""
Funciones para crear y actualizar gráficos (histogramas, barras, pastel, etc.).
"""

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def mostrar_histograma(frame_grafico, datos, area, canal):
	"""
	Dibuja el gráfico de histograma y muestra estadísticas para el área y canal seleccionados en el frame dado.
	"""
	# Limpiar frame
	for widget in frame_grafico.winfo_children():
		widget.destroy()
	fig = Figure(figsize=(5,4))
	ax = fig.add_subplot(111)
	hist = datos['histograma']
	ax.bar(range(len(hist)), hist, color=canal.lower())
	ax.set_title(f"Histograma {canal} - Área: {area}")
	ax.set_xlabel(f"Intensidad {canal} (normalizada)")
	ax.set_ylabel("Frecuencia")
	# Mostrar estadísticos
	media = datos['media']
	std = datos['std']
	ax.text(0.95, 0.95, f"Media: {media:.4f}\nStd: {std:.4f}", transform=ax.transAxes,
			fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.7))
	canvas = FigureCanvasTkAgg(fig, master=frame_grafico)
	canvas.draw()
	canvas.get_tk_widget().pack()
