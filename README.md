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



## 👥 Autores

<div align="center">
   <table>
      <tr>
         <td align="center">
            <img src="images/autores/linich.jpg" width="150" style="border-radius:50%"><br>
            <b>Jorge Soto</b><br>
            <a href="https://github.com/Linich14">@Linich14</a>
         </td>
         <td align="center">
            <img src="images/autores/dpbascur.png" width="150" style="border-radius:50%"><br>
            <b>Daniel Peña</b><br>
            <a href="https://github.com/DPBascur">@DPBascur</a>
         </td>
      </tr>
   </table>
</div>

---
Proyecto académico. Uso educativo y de experimentación.


