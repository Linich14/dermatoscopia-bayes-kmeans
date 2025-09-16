"""
Punto de entrada principal de la aplicación.
Inicializa la interfaz gráfica, ejecuta el preprocesamiento y la clasificación automática,
y muestra los resultados agregados en la GUI.
"""

# Importaciones necesarias
import sys
sys.path.append('src')
from preprocesamiento.carga import cargar_imagenes_y_mascaras
from preprocesamiento.particion import particionar_datos
from estadisticas.estadisticas_rgb import estadisticas_rgb
from gui.ventanas import VentanaPrincipal

def main():
	"""
	Flujo principal del sistema:
	1. Carga y preprocesa todas las imágenes y máscaras
	2. Particiona los datos en entrenamiento, validación y test
	3. Calcula estadísticas RGB
	4. Muestra la interfaz gráfica interactiva con clasificador integrado
	"""
	print("Cargando y particionando datos...")
	imagenes = cargar_imagenes_y_mascaras()
	train, val, test = particionar_datos(imagenes)
	
	print("Calculando estadísticas RGB...")
	stats = estadisticas_rgb(train)
	
	print("Iniciando interfaz gráfica...")
	ventana = VentanaPrincipal(stats)
	ventana.mainloop()

if __name__ == "__main__":
	main()

