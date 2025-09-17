"""
Punto de entrada principal de la aplicación.
Inicializa la interfaz gráfica, ejecuta el preprocesamiento y la clasificación automática,
y muestra los resultados agregados en la GUI.
"""

# Importaciones necesarias
import sys
import os

# Agregar el directorio padre al path para poder importar módulos src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocesamiento.carga import cargar_imagenes_y_mascaras
from src.preprocesamiento.particion import particionar_datos
from src.estadisticas.estadisticas_rgb import estadisticas_rgb

# Importar la ventana modular (arquitectura MVC)
from src.gui.ventana_modular import VentanaPrincipalModular

ventana_clase = VentanaPrincipalModular
ventana_tipo = "modular (MVC)"
print("✅ Usando VentanaPrincipalModular - Arquitectura MVC")

def main():
	"""
	Flujo principal del sistema:
	1. Carga y preprocesa todas las imágenes y máscaras
	2. Particiona los datos en entrenamiento, validación y test
	3. Calcula estadísticas RGB
	4. Muestra la interfaz gráfica interactiva con clasificador integrado
	"""
	print("🔄 Cargando y particionando datos...")
	imagenes = cargar_imagenes_y_mascaras()
	train, val, test = particionar_datos(imagenes)
	
	print("📊 Calculando estadísticas RGB...")
	stats = estadisticas_rgb(train)
	
	print(f"🚀 Iniciando interfaz gráfica {ventana_tipo}...")
	ventana = ventana_clase(stats, train, val, test)
	ventana.mainloop()

if __name__ == "__main__":
	main()

