# Dermatoscopia Bayes-KMeans

Sistema para clasificación de imágenes dermatoscópicas a nivel de píxel

## 🏗️ Estructura del proyecto

```
src/
├── clasificadores/           # Sistema de clasificación modular
│   ├── bayesiano/           # Clasificador Bayesiano RGB completo
│   │   ├── clasificador.py  # Clasificador principal con muestreo equilibrado
│   │   ├── modelo.py        # Modelo gaussiano multivariado
│   │   ├── umbrales.py      # Estrategias de selección de umbral
│   │   ├── evaluacion.py    # Métricas y evaluación de rendimiento
│   │   └── base.py          # Interfaces y clases base
│   ├── evaluacion.py        # Evaluación comparativa de criterios
│   ├── clasificador_roc.py  # (Pendiente) Análisis ROC
│   └── clasificador_kmeans.py # (Pendiente) K-Means
├── config/                  # Configuración centralizada
│   └── configuracion.py     # Parámetros del sistema
├── estadisticas/            # Análisis estadístico RGB
│   └── estadisticas_rgb.py  # Cálculo de histogramas y estadísticos
├── gui/                     # Interfaz gráfica modular (MVC)
│   ├── ventana_modular.py   # Ventana principal (arquitectura MVC)
│   ├── components/          # Componentes UI reutilizables
│   ├── controllers/         # Controladores de lógica de negocio
│   ├── dialogs/             # Diálogos especializados
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

El sistema implementa todos los requisitos especificados en las condiciones del proyecto:

- **✅ Partición de datos:** 60% entrenamiento / 20% validación / 20% test por imagen
- **✅ Muestreo equilibrado:** Balanceo automático de píxeles lesión/no-lesión para entrenamiento
- **✅ Exploración visual:** Histogramas y estadísticos RGB por área y canal
- **✅ Clasificador Bayesiano:** Implementación completa con razón de verosimilitud
- **✅ Criterios de umbral:** Youden, Equal Error Rate, Prior Balanced con justificación
- **✅ Reproducibilidad:** Semilla fija (SEED=42) para resultados consistentes
- **✅ Evaluación robusta:** Métricas completas en conjunto de test

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

### Clasificador Bayesiano RGB
- **Modelo gaussiano multivariado** para píxeles RGB
- **Muestreo equilibrado automático** (50% lesión, 50% no-lesión)
- **Selección de umbral óptimo** con 3 criterios disponibles:
  - **Youden:** Maximiza sensibilidad + especificidad
  - **Equal Error Rate:** Equilibra falsos positivos y negativos  
  - **Prior Balanced:** Considera probabilidades a priori


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
- **📊 PCA + Bayesiano** - Reducción de dimensionalidad
- **📈 Métricas avanzadas** - Índice de Jaccard a nivel de imagen

### 🎯 Próximas mejoras
- **📋 Exportación de resultados** - Informes en PDF/HTML
- **🎛️ Configuración avanzada** - Parámetros de muestreo ajustables
- **📊 Análisis estadístico** - Pruebas de significancia
- **🔧 Optimizaciones** - Procesamiento en paralelo


## 📚 Documentación técnica

### Clasificador Bayesiano
```python
# Ejemplo de uso
from src.clasificadores.bayesiano import ClasificadorBayesianoRGB

clasificador = ClasificadorBayesianoRGB(criterio_umbral='youden')
clasificador.entrenar(datos_entrenamiento)
resultado = clasificador.clasificar(imagen_test)
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


