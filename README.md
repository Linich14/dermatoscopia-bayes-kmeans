# Dermatoscopia Bayes-KMeans

Sistema modular para la exploración estadística y clasificación automática de imágenes dermatoscópicas, utilizando métodos Bayesianos y K-Means. Incluye una interfaz gráfica interactiva para visualizar histogramas y estadísticas RGB de áreas seleccionadas.

## Estructura del proyecto

```
src/
├── clasificadores/      # Módulos para clasificadores Bayes, K-Means, ROC (pendientes)
├── config/              # Configuración general
├── estadisticas/        # Cálculo de estadísticas RGB
├── events/              # (Pendiente) Gestión de eventos
├── gui/                 # Interfaz gráfica y visualización
│   ├── ventanas.py      # Ventana principal de la GUI
│   ├── graficos.py      # Funciones de visualización (histogramas, etc.)
│   ├── styles.py        # (Pendiente) Estilos de la GUI
│   ├── widgets.py       # (Pendiente) Widgets personalizados
├── main.py              # Punto de entrada principal
├── muestreo/            # Muestreo equilibrado
├── preprocesamiento/    # Carga y partición de datos
├── utils/               # (Pendiente) Utilidades varias
data/                    # Imágenes y máscaras de entrada
requirements.txt         # Dependencias Python
README.md                # Este archivo
```

## Instalación

1. Clona el repositorio y entra al directorio.
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

## Uso

Ejecuta el sistema desde la raíz del proyecto:
```bash
python src/main.py
```

## Flujo principal

1. **Carga y preprocesamiento:** Se cargan todas las imágenes y máscaras desde `data/`.
2. **Partición:** Los datos se dividen en conjuntos de entrenamiento, validación y test.
3. **Cálculo de estadísticas:** Se calculan estadísticas RGB por área y canal.
4. **Visualización:** Se muestra la interfaz gráfica interactiva, permitiendo explorar histogramas y estadísticas.

## Visualización

- La lógica de visualización (histogramas, estadísticas) está modularizada en `gui/graficos.py`.
- La ventana principal (`gui/ventanas.py`) utiliza estas funciones para mostrar los resultados.

## Clasificadores

- Los módulos de clasificadores (`clasificadores/`) están preparados para implementar Bayes, K-Means y ROC, pero actualmente contienen solo docstrings y comentarios de "pendiente".

## Créditos

- Proyecto académico para la exploración y clasificación de imágenes dermatoscópicas.
