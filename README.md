# Dermatoscopia Bayes-KMeans

Este proyecto implementa una aplicación para segmentar lesiones en imágenes de dermatoscopía usando inteligencia artificial. Permite comparar distintos métodos de clasificación de píxeles: Bayesiano RGB, Bayesiano con PCA y K-Means.

## Objetivo

Desarrollar una herramienta que ayude a diferenciar entre regiones de lesión y no-lesión en imágenes de piel, facilitando el apoyo al diagnóstico médico.

## ¿Qué hace la aplicación?
- Carga imágenes y máscaras de referencia.
- Extrae características de color y/o textura de cada píxel.
- Permite entrenar y comparar:
  - Clasificador Bayesiano RGB
  - Clasificador Bayesiano con reducción de dimensionalidad (PCA)
  - Algoritmo no supervisado K-Means
- Muestra resultados y métricas de segmentación (exactitud, precisión, sensibilidad, etc.)
- Visualiza las segmentaciones obtenidas y las compara con la referencia.

## Uso rápido

1. Instala dependencias:
   ```bash
   pip install -r requirements.txt
   ```
2. Ejecuta la aplicación:
   ```bash
   python src/main.py
   ```

## Estructura básica

- `src/` Código fuente principal
- `data/` Imágenes y máscaras
- `requirements.txt` Dependencias


## Autores
- [Jorge Soto](https://github.com/Linich14)
- [Daniel Peña](https://github.com/DPBascur)

---
Proyecto académico. Uso educativo y de experimentación.


