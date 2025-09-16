# Dermatoscopia Bayes-KMeans

Sistema avanzado para clasificación de imágenes dermatoscópicas a nivel de píxel con **clasificadores Bayesianos RGB y PCA**

## 🏗️ Estructura del proyecto

```
src/
├── clasificadores/           # Sistema de clasificación modular
│   ├── bayesiano/           # Clasificadores Bayesianos RGB y PCA
│   │   ├── clasificador.py  # Clasificador Bayesiano RGB base
│   │   ├── clasificador_pca.py # Clasificador Bayesiano + PCA (NUEVO)
│   │   ├── modelo.py        # Modelo gaussiano multivariado
│   │   ├── umbrales.py      # Estrategias de selección de umbral
│   │   ├── evaluacion.py    # Métricas y evaluación de rendimiento
│   │   └── base.py          # Interfaces y clases base
│   ├── evaluacion.py        # Evaluación comparativa de criterios
│   ├── clasificador_roc.py  # (Pendiente) Análisis ROC
│   └── clasificador_kmeans.py # (Pendiente) K-Means
├── reduccion_dimensionalidad/ # Reducción de dimensionalidad (NUEVO)
│   ├── pca_especializado.py  # PCA con selección automática de componentes
│   └── __init__.py          # Exportaciones del módulo PCA
├── config/                  # Configuración centralizada
│   └── configuracion.py     # Parámetros del sistema
├── estadisticas/            # Análisis estadístico RGB
│   └── estadisticas_rgb.py  # Cálculo de histogramas y estadísticos
├── gui/                     # Interfaz gráfica modular (MVC)
│   ├── ventana_modular.py   # Ventana principal con controles RGB/PCA
│   ├── components/          # Componentes UI reutilizables
│   ├── controllers/         # Controladores con soporte RGB/PCA
│   ├── dialogs/             # Diálogos especializados + comparación PCA
│   ├── graficos.py          # Funciones de visualización
│   └── styles.py            # Sistema de estilos centralizado
├── muestreo/                # Muestreo equilibrado de píxeles
│   └── muestreo_equilibrado.py # Balanceo automático de clases
├── preprocesamiento/        # Carga y partición de datos
│   ├── carga.py             # Carga de imágenes y máscaras
│   └── particion.py         # Partición estratificada (60/20/20)
├── events/                  # Gestión de eventos del sistema
└── main.py                  # Punto de entrada principal
```

## 📋 Requisitos del proyecto

El sistema implementa **TODOS** los requisitos especificados, incluyendo el **NUEVO requisito de PCA**:

### ✅ Requisitos Base Implementados
- **✅ Partición de datos:** 60% entrenamiento / 20% validación / 20% test por imagen
- **✅ Muestreo equilibrado:** Balanceo automático de píxeles lesión/no-lesión para entrenamiento
- **✅ Exploración visual:** Histogramas y estadísticos RGB por área y canal
- **✅ Clasificador Bayesiano RGB:** Implementación completa con razón de verosimilitud
- **✅ Criterios de umbral:** Youden, Equal Error Rate, Prior Balanced con justificación
- **✅ Reproducibilidad:** Semilla fija (SEED=42) para resultados consistentes
- **✅ Evaluación robusta:** Métricas completas en conjunto de test

### 🆕 NUEVO: Clasificador Bayesiano + PCA
- **✅ Reducción dimensional:** PCA aplicado al espacio RGB (3D → 1-3D)
- **✅ Selección automática de componentes:** 3 criterios implementados
  - **Varianza:** Umbral de varianza explicada acumulada (95%, 99%)
  - **Codo (Elbow):** Análisis de curvatura para punto óptimo
  - **Discriminativo:** Razón de Fisher para máxima separabilidad entre clases
- **✅ Justificación metodológica:** Análisis automático y detallado de la selección
- **✅ Comparación RGB vs PCA:** Evaluación completa de rendimiento
- **✅ Interfaz integrada:** Controles UI para configurar y comparar ambos métodos

## 🚀 Instalación y uso

### Requisitos previos
```bash
Python 3.8+
pip install -r requirements.txt
```

### Ejecución
```bash
python src/main.py
```

### Flujo de trabajo
1. **🔄 Carga automática:** Imágenes y máscaras desde `data/`
2. **📊 Partición estratificada:** División automática 60/20/20
3. **📈 Estadísticas RGB:** Cálculo de histogramas por área/canal  
4. **🎨 Interfaz interactiva:** Exploración visual y entrenamiento
5. **🤖 Entrenamiento:** Clasificador Bayesiano con muestreo equilibrado
6. **📊 Evaluación:** Métricas de rendimiento y comparación de criterios

## 🎯 Funcionalidades implementadas

### 🔬 Clasificador Bayesiano RGB (Base)
- **Modelo gaussiano multivariado** para píxeles RGB
- **Muestreo equilibrado automático** (50% lesión, 50% no-lesión)
- **Selección de umbral óptimo** con 3 criterios disponibles:
  - **Youden:** Maximiza sensibilidad + especificidad
  - **Equal Error Rate:** Equilibra falsos positivos y negativos  
  - **Prior Balanced:** Considera probabilidades a priori
- **Evaluación completa:** Métricas detalladas y matriz de confusión

### 🆕 Clasificador Bayesiano + PCA (Nuevo Requisito)
- **Reducción dimensional automática:** De RGB 3D a espacio PCA optimizado
- **Selección inteligente de componentes** con 3 estrategias:
  - **Criterio Varianza:** Preserva 95%/99% de varianza explicada
  - **Método del Codo:** Detecta punto de inflexión óptimo automáticamente
  - **Capacidad Discriminativa:** Maximiza separabilidad entre clases (Fisher)
- **Justificación metodológica automática** con análisis detallado
- **Comparación RGB vs PCA** con recomendaciones fundamentadas
- **Preservación de información discriminativa** para clasificación médica



## 📊 Estado de desarrollo

### ✅ Completado
- **🧠 Clasificador Bayesiano RGB** - Implementación completa y funcional
- **⚖️ Muestreo equilibrado** - Integrado en el entrenamiento del clasificador
- **🎯 Criterios de umbral** - Youden, EER, Prior Balanced implementados
- **📊 Visualización avanzada** - Histogramas y estadísticas interactivos
- **📈 Evaluación completa** - Métricas y comparación de criterios

### 🔄 En desarrollo
- **📈 Análisis ROC** - Curvas ROC y métricas AUC
- **🎯 K-Means** - Clasificación no supervisada
- **📈 Métricas avanzadas** - Índice de Jaccard a nivel de imagen

### 🎯 Próximas mejoras
- **📋 Exportación de resultados** - Informes en PDF/HTML
- **🎛️ Configuración avanzada** - Parámetros de muestreo ajustables
- **📊 Análisis estadístico** - Pruebas de significancia
- **🔧 Optimizaciones** - Procesamiento en paralelo


## 📚 Documentación técnica

### Clasificador Bayesiano RGB (Tradicional)
```python
# Ejemplo de uso del clasificador RGB base
from src.clasificadores.bayesiano import ClasificadorBayesianoRGB

clasificador = ClasificadorBayesianoRGB(criterio_umbral='youden')
clasificador.entrenar(datos_entrenamiento)
resultado = clasificador.clasificar(imagen_test)
metricas = clasificador.evaluar(datos_test)
```

### 🆕 Clasificador Bayesiano + PCA
```python
# Ejemplo de uso del nuevo clasificador PCA
from src.clasificadores.bayesiano import ClasificadorBayesianoPCA

# Configuración con criterio de varianza
clasificador_pca = ClasificadorBayesianoPCA(
    criterio_umbral='youden',
    criterio_pca='varianza',
    umbral_varianza=0.95  # Preservar 95% de varianza
)

# Entrenar con reducción dimensional automática
clasificador_pca.entrenar(datos_entrenamiento)
print(f"Componentes seleccionados: {clasificador_pca.num_componentes_pca}")

# Obtener justificación metodológica
justificacion = clasificador_pca.obtener_justificacion_pca()
print(justificacion)

# Comparar con clasificador RGB
comparacion = clasificador_pca.comparar_con_rgb(datos_validacion)
reporte = clasificador_pca.generar_reporte_comparativo(comparacion)
print(reporte)
```

### Criterios de Selección PCA
```python
# Criterio basado en varianza explicada
pca_varianza = ClasificadorBayesianoPCA(
    criterio_pca='varianza',
    umbral_varianza=0.99  # 99% de varianza
)

# Criterio del codo (elbow method)
pca_codo = ClasificadorBayesianoPCA(
    criterio_pca='codo',
    sensibilidad=0.1  # Sensibilidad para detectar codo
)

# Criterio discriminativo (Fisher)
pca_discriminativo = ClasificadorBayesianoPCA(
    criterio_pca='discriminativo',
    umbral_fisher=0.15  # Umbral mínimo Fisher ratio
)
```

### Análisis de Reducción Dimensional
```python
# Obtener análisis detallado del PCA
analisis = clasificador_pca.obtener_analisis_pca()
print(f"Varianza primera componente: {analisis['varianza_primera_componente']:.1%}")
print(f"Componentes para 90% varianza: {analisis['num_componentes_90_pct']}")

# Parámetros completos del modelo
parametros = clasificador_pca.obtener_parametros()
print(f"Reducción: {parametros['reduccion_dimensional']}")
print(f"Varianza preservada: {parametros['varianza_preservada']:.1%}")
```
metricas = clasificador.evaluar(datos_test)
```

### Muestreo Equilibrado
```python
# Muestreo automático 50/50
from src.muestreo.muestreo_equilibrado import muestreo_equilibrado

X_equilibrado, y_equilibrado = muestreo_equilibrado(imagenes)
# Resultado: 50% píxeles lesión, 50% píxeles no-lesión
```


## 📄 Licencia

Proyecto académico para análisis y clasificación de imágenes dermatoscópicas.


