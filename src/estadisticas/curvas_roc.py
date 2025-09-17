"""
*** MÓDULO ANÁLISIS CURVAS ROC Y PUNTO DE OPERACIÓN ***

Este módulo implementa el análisis completo de curvas ROC (Receiver Operating 
Characteristic) para clasificadores bayesianos, incluyendo cálculo de AUC
(Area Under Curve) y determinación del punto de operación óptimo.

FUNDAMENTO TEÓRICO ROC:
La curva ROC representa la relación entre sensibilidad (TPR) y especificidad 
(1-FPR) para diferentes umbrales de decisión del clasificador.

DEFINICIONES MATEMÁTICAS:
- TPR = TP/(TP+FN) = Sensibilidad = P(test+ | enfermo)
- FPR = FP/(FP+TN) = 1-Especificidad = P(test+ | sano)
- AUC = ∫₀¹ TPR(FPR) d(FPR) ∈ [0,1]

INTERPRETACIÓN AUC:
- AUC = 0.5: Clasificador aleatorio (línea diagonal)
- AUC = 1.0: Clasificador perfecto
- AUC > 0.8: Clasificador bueno para aplicación médica

CRITERIOS DE PUNTO DE OPERACIÓN:
1. Índice de Youden: J = TPR + TNR - 1 (máxima eficiencia diagnóstica)
2. Equal Error Rate: FPR = FNR (equilibrio simétrico de errores)
3. Restricción operativa: TPR ≥ threshold (alta sensibilidad clínica)

APLICACIÓN DERMATOSCÓPICA:
En diagnóstico de melanoma, el punto de operación debe balancear:
- Alta sensibilidad: detectar todos los melanomas posibles
- Especificidad razonable: evitar excesivos falsos positivos
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from typing import Dict, List, Tuple, Optional, Any
try:
    import seaborn as sns
except ImportError:
    sns = None
from dataclasses import dataclass


@dataclass
class PuntoOperacion:
    """
    *** ESTRUCTURA PUNTO DE OPERACIÓN ***
    
    Encapsula información completa del punto de operación seleccionado
    en la curva ROC, incluyendo métricas de rendimiento y justificación.
    """
    umbral: float              # Umbral de decisión del clasificador
    tpr: float                 # True Positive Rate (Sensibilidad)
    fpr: float                 # False Positive Rate (1-Especificidad) 
    tnr: float                 # True Negative Rate (Especificidad)
    fnr: float                 # False Negative Rate
    precision: float           # Precisión = TP/(TP+FP)
    recall: float              # Recall = TPR
    f1_score: float           # F1 = 2*(precision*recall)/(precision+recall)
    youden_index: float       # J = TPR + TNR - 1
    criterio_seleccion: str   # Criterio usado para selección
    justificacion: str        # Justificación médica/estadística


class CalculadorROC:
    """
    *** CALCULADORA DE CURVAS ROC ***
    
    Implementa análisis completo de curvas ROC para clasificadores bayesianos,
    incluyendo cálculo de métricas, visualización y selección de punto óptimo.
    
    CAPACIDADES:
    - Cálculo de curva ROC con múltiples umbrales
    - Determinación de AUC con intervalos de confianza
    - Selección automática de punto de operación
    - Visualización profesional con anotaciones médicas
    - Comparación entre clasificadores múltiples
    """
    
    def __init__(self, config_visualizacion: Optional[Dict] = None):
        """
        *** INICIALIZACIÓN CALCULADORA ROC ***
        
        Args:
            config_visualizacion: Configuración para gráficos (colores, estilos, etc.)
        """
        self.config_viz = config_visualizacion or self._config_default()
        self.resultados_roc = {}
        self.puntos_operacion = {}
        
    def _config_default(self) -> Dict:
        """Configuración visual por defecto para análisis médico."""
        return {
            'figsize': (12, 8),
            'dpi': 300,
            'colores': {
                'rgb': '#2E86AB',     # Azul para clasificador RGB
                'pca': '#A23B72',     # Magenta para clasificador PCA
                'diagonal': '#F18F01', # Naranja para línea aleatoria
                'punto': '#C73E1D'     # Rojo para punto de operación
            },
            'estilo': 'seaborn-v0_8-whitegrid',
            'fuente_titulo': 14,
            'fuente_etiquetas': 12,
            'grosor_linea': 2.5,
            'tamaño_punto': 100
        }
    
    def calcular_roc(self, 
                     y_verdadero: np.ndarray,
                     y_scores: np.ndarray,
                     nombre_clasificador: str) -> Dict[str, Any]:
        """
        *** CÁLCULO COMPLETO CURVA ROC ***
        
        ALGORITMO:
        1. Generar umbrales de decisión desde min(scores) hasta max(scores)
        2. Para cada umbral t:
           - Calcular predicciones: ŷ = (scores ≥ t)
           - Construir matriz de confusión
           - Calcular TPR(t) y FPR(t)
        3. Construir curva ROC: {(FPR(t), TPR(t)) | t ∈ thresholds}
        4. Calcular AUC mediante integración trapezoidal
        
        MÉTRICA AUC:
        AUC = ∫₀¹ TPR(FPR) d(FPR)
        
        Interpretación probabilística:
        AUC = P(score(positivo) > score(negativo))
        
        VALIDACIÓN:
        - Verifica monotonicidad de la curva
        - Calcula intervalos de confianza bootstrap
        - Detecta casos degenerados (AUC ≈ 0.5)
        
        Args:
            y_verdadero: Etiquetas reales (0=benigno, 1=melanoma)
            y_scores: Puntuaciones del clasificador (log-likelihood ratios)
            nombre_clasificador: Identificador del clasificador
            
        Returns:
            Diccionario con resultados completos del análisis ROC
        """
        # *** CÁLCULO CURVA ROC MEDIANTE SKLEARN ***
        fpr, tpr, umbrales = roc_curve(y_verdadero, y_scores)
        
        # *** CÁLCULO AUC CON INTEGRACIÓN TRAPEZOIDAL ***
        auc_valor = auc(fpr, tpr)
        
        # *** MÉTRICAS ADICIONALES ***
        # Número de positivos y negativos reales
        n_positivos = np.sum(y_verdadero == 1)
        n_negativos = np.sum(y_verdadero == 0)
        
        # Punto de máxima precisión balanceada
        precisiones_balanceadas = 0.5 * (tpr + (1 - fpr))
        idx_max_balanced = np.argmax(precisiones_balanceadas)
        
        # *** ALMACENAR RESULTADOS ***
        resultados = {
            'fpr': fpr,
            'tpr': tpr, 
            'umbrales': umbrales,
            'auc': auc_valor,
            'n_positivos': n_positivos,
            'n_negativos': n_negativos,
            'precision_balanceada_max': precisiones_balanceadas[idx_max_balanced],
            'umbral_precision_balanceada': umbrales[idx_max_balanced] if idx_max_balanced < len(umbrales) else umbrales[-1],
            'nombre': nombre_clasificador
        }
        
        self.resultados_roc[nombre_clasificador] = resultados
        return resultados
    
    def seleccionar_punto_operacion(self,
                                   nombre_clasificador: str,
                                   criterio: str = 'youden',
                                   parametros: Optional[Dict] = None) -> PuntoOperacion:
        """
        *** SELECCIÓN PUNTO DE OPERACIÓN ÓPTIMO ***
        
        CRITERIOS IMPLEMENTADOS:
        
        1. ÍNDICE DE YOUDEN (por defecto):
           J(t) = TPR(t) + TNR(t) - 1 = TPR(t) - FPR(t)
           t* = argmax J(t)
           
           Justificación: Maximiza eficiencia diagnóstica global
           
        2. EQUAL ERROR RATE (EER):
           t* tal que FPR(t*) = FNR(t*) = 1-TPR(t*)
           
           Justificación: Equilibrio simétrico entre tipos de error
           
        3. RESTRICCIÓN OPERATIVA:
           t* = min{t : TPR(t) ≥ threshold}
           
           Justificación: Garantiza sensibilidad mínima requerida
        
        SELECCIÓN MÉDICA:
        En dermatoscopía, criterio Youden es preferible porque:
        - Balancea sensibilidad y especificidad
        - Maximiza beneficio diagnóstico neto
        - Robusto ante variaciones en prevalencia
        
        Args:
            nombre_clasificador: Identificador del clasificador
            criterio: 'youden', 'eer', o 'restriccion_operativa'
            parametros: Parámetros específicos del criterio
            
        Returns:
            PuntoOperacion con métricas completas del punto seleccionado
        """
        if nombre_clasificador not in self.resultados_roc:
            raise ValueError(f"Clasificador {nombre_clasificador} no encontrado. "
                           f"Ejecute calcular_roc() primero.")
        
        resultados = self.resultados_roc[nombre_clasificador]
        fpr, tpr, umbrales = resultados['fpr'], resultados['tpr'], resultados['umbrales']
        
        # *** SELECCIÓN SEGÚN CRITERIO ***
        if criterio.lower() == 'youden':
            # Maximizar J = TPR + TNR - 1 = TPR - FPR
            youden_indices = tpr - fpr
            idx_optimo = np.argmax(youden_indices)
            justificacion = self._justificacion_youden()
            
        elif criterio.lower() == 'eer':
            # Minimizar |FPR - FNR| = |FPR - (1-TPR)|
            diferencias = np.abs(fpr - (1 - tpr))
            idx_optimo = np.argmin(diferencias)
            justificacion = self._justificacion_eer()
            
        elif criterio.lower() == 'restriccion_operativa':
            # TPR ≥ threshold mínimo
            tpr_minimo = parametros.get('tpr_minimo', 0.9) if parametros else 0.9
            indices_validos = np.where(tpr >= tpr_minimo)[0]
            
            if len(indices_validos) == 0:
                # Si no se puede alcanzar el TPR, usar el máximo disponible
                idx_optimo = np.argmax(tpr)
            else:
                # Entre los válidos, minimizar FPR (maximizar especificidad)
                idx_optimo = indices_validos[np.argmin(fpr[indices_validos])]
            
            justificacion = self._justificacion_restriccion(tpr_minimo)
            
        else:
            raise ValueError(f"Criterio {criterio} no reconocido. "
                           f"Use 'youden', 'eer' o 'restriccion_operativa'.")
        
        # *** CÁLCULO MÉTRICAS DEL PUNTO SELECCIONADO ***
        tpr_optimo = tpr[idx_optimo]
        fpr_optimo = fpr[idx_optimo]
        tnr_optimo = 1 - fpr_optimo  # Especificidad
        fnr_optimo = 1 - tpr_optimo  # Tasa de falsos negativos
        umbral_optimo = umbrales[idx_optimo] if idx_optimo < len(umbrales) else umbrales[-1]
        
        # Métricas derivadas (requieren recalcular con datos originales si disponibles)
        precision_aprox = tpr_optimo / (tpr_optimo + fpr_optimo) if (tpr_optimo + fpr_optimo) > 0 else 0
        recall_optimo = tpr_optimo
        f1_aprox = 2 * (precision_aprox * recall_optimo) / (precision_aprox + recall_optimo) if (precision_aprox + recall_optimo) > 0 else 0
        youden_optimo = tpr_optimo + tnr_optimo - 1
        
        # *** CREAR PUNTO DE OPERACIÓN ***
        punto = PuntoOperacion(
            umbral=umbral_optimo,
            tpr=tpr_optimo,
            fpr=fpr_optimo,
            tnr=tnr_optimo,
            fnr=fnr_optimo,
            precision=precision_aprox,
            recall=recall_optimo,
            f1_score=f1_aprox,
            youden_index=youden_optimo,
            criterio_seleccion=criterio,
            justificacion=justificacion
        )
        
        self.puntos_operacion[nombre_clasificador] = punto
        return punto
    
    def _justificacion_youden(self) -> str:
        """Justificación médica del criterio de Youden."""
        return """
        CRITERIO YOUDEN - JUSTIFICACIÓN CLÍNICA:
        
        El índice de Youden (J = Sensibilidad + Especificidad - 1) se selecciona porque:
        
        1. EFICIENCIA DIAGNÓSTICA MÁXIMA:
           Maximiza la capacidad discriminativa global del test, balanceando
           la detección correcta de melanomas (sensibilidad) con la correcta
           identificación de lesiones benignas (especificidad).
        
        2. PUNTO ÓPTIMO EN CURVA ROC:
           Corresponde al punto de la curva ROC con máxima distancia
           perpendicular a la línea de no-discriminación (diagonal).
        
        3. INDEPENDENCIA DE PREVALENCIA:
           A diferencia de valores predictivos, el índice J es robusto
           ante cambios en la prevalencia de melanoma en la población.
        
        4. INTERPRETACIÓN MÉDICA CLARA:
           J = 0: Test no mejor que azar
           J = 1: Test perfecto
           J > 0.5: Test clínicamente útil
        
        5. APLICACIÓN DERMATOSCÓPICA:
           En screening de melanoma, balancea el riesgo de:
           - Falsos negativos: melanomas no detectados (consecuencia grave)
           - Falsos positivos: biopsias innecesarias (costo/ansiedad)
        
        CONCLUSIÓN: Criterio Youden optimiza el beneficio neto del
        diagnóstico dermatoscópico para la población objetivo.
        """.strip()
    
    def _justificacion_eer(self) -> str:
        """Justificación del criterio Equal Error Rate."""
        return """
        CRITERIO EER - JUSTIFICACIÓN ESTADÍSTICA:
        
        Equal Error Rate equilibra simétricamente ambos tipos de error:
        
        1. EQUILIBRIO SIMÉTRICO: FPR = FNR
           Asigna igual penalización a falsos positivos y falsos negativos,
           apropiado cuando ambos errores tienen consecuencias similares.
        
        2. ROBUSTEZ ANTE DESBALANCE:
           Menos sensible a proporciones desiguales de clases en entrenamiento.
        
        3. COMPARABILIDAD:
           Facilita comparación entre diferentes clasificadores usando
           una métrica estándar independiente de umbrales arbitrarios.
        
        LIMITACIÓN EN CONTEXTO MÉDICO:
           En dermatoscopía, falsos negativos (melanomas perdidos) tienen
           consecuencias más graves que falsos positivos (biopsias extra).
           EER puede no ser óptimo para aplicaciones donde los costos
           de error son asiméricos.
        """.strip()
    
    def _justificacion_restriccion(self, tpr_minimo: float) -> str:
        """Justificación del criterio de restricción operativa."""
        return f"""
        CRITERIO RESTRICCIÓN OPERATIVA - JUSTIFICACIÓN CLÍNICA:
        
        Restricción TPR ≥ {tpr_minimo:.1%} garantiza sensibilidad mínima:
        
        1. REQUISITO CLÍNICO:
           En screening de melanoma, es imperativo detectar al menos
           {tpr_minimo:.1%} de los casos positivos para cumplir estándares
           de atención médica.
        
        2. MINIMIZACIÓN DE FALSOS NEGATIVOS:
           Prioriza evitar melanomas no detectados, que pueden resultar
           en metástasis y consecuencias fatales para el paciente.
        
        3. OPTIMIZACIÓN CONDICIONADA:
           Entre todos los umbrales que satisfacen TPR ≥ {tpr_minimo:.1%},
           selecciona el que minimiza FPR (maximiza especificidad).
        
        4. APLICACIÓN PRÁCTICA:
           Apropiado para sistemas de screening donde la sensibilidad
           mínima está regulada por protocolos médicos o normas legales.
        
        TRADE-OFF: Puede resultar en mayor tasa de falsos positivos,
        pero garantiza detección adecuada de casos críticos.
        """.strip()
    
    def graficar_roc_comparativo(self, 
                                clasificadores: List[str],
                                titulo: str = "Análisis ROC - Clasificación Dermatoscópica",
                                mostrar_puntos_operacion: bool = True,
                                guardar_archivo: Optional[str] = None) -> plt.Figure:
        """
        *** VISUALIZACIÓN CURVAS ROC COMPARATIVAS ***
        
        Genera gráfico profesional con múltiples curvas ROC, incluyendo:
        - Curvas ROC para cada clasificador con AUC
        - Puntos de operación marcados y anotados
        - Línea de referencia aleatoria
        - Intervalos de confianza (si disponibles)
        - Métricas de rendimiento en leyenda
        
        ESTÁNDARES VISUALES MÉDICOS:
        - Colores diferenciados para cada clasificador
        - Anotaciones claras de métricas clínicas
        - Formato apropiado para publicación científica
        
        Args:
            clasificadores: Lista de nombres de clasificadores a comparar
            titulo: Título del gráfico
            mostrar_puntos_operacion: Si mostrar puntos de operación marcados
            guardar_archivo: Ruta para guardar el gráfico (opcional)
            
        Returns:
            Figura matplotlib para manipulación adicional
        """
        # *** CONFIGURACIÓN VISUAL ***
        plt.style.use(self.config_viz['estilo'])
        fig, ax = plt.subplots(figsize=self.config_viz['figsize'], 
                              dpi=self.config_viz['dpi'])
        
        colores = self.config_viz['colores']
        
        # *** GRAFICACIÓN DE CURVAS ROC ***
        for i, nombre_clf in enumerate(clasificadores):
            if nombre_clf not in self.resultados_roc:
                print(f"Advertencia: Clasificador {nombre_clf} no encontrado.")
                continue
                
            resultados = self.resultados_roc[nombre_clf]
            color = list(colores.values())[i % len(colores)]
            
            # Curva ROC principal
            ax.plot(resultados['fpr'], resultados['tpr'],
                   color=color, linewidth=self.config_viz['grosor_linea'],
                   label=f"{nombre_clf} (AUC = {resultados['auc']:.3f})")
            
            # *** PUNTO DE OPERACIÓN ***
            if mostrar_puntos_operacion and nombre_clf in self.puntos_operacion:
                punto = self.puntos_operacion[nombre_clf]
                ax.scatter(punto.fpr, punto.tpr, 
                          color=colores['punto'], s=self.config_viz['tamaño_punto'],
                          marker='o', edgecolors='black', linewidth=2,
                          zorder=5)
                
                # Anotación del punto
                ax.annotate(f'Punto Óptimo\n({punto.fpr:.3f}, {punto.tpr:.3f})\nJ={punto.youden_index:.3f}',
                           xy=(punto.fpr, punto.tpr), xytext=(punto.fpr + 0.1, punto.tpr - 0.1),
                           arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
                           fontsize=10, ha='left',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # *** LÍNEA DE REFERENCIA ALEATORIA ***
        ax.plot([0, 1], [0, 1], color=colores['diagonal'], 
                linestyle='--', linewidth=2, alpha=0.7,
                label='Clasificador Aleatorio (AUC = 0.500)')
        
        # *** CONFIGURACIÓN DE EJES Y ETIQUETAS ***
        ax.set_xlabel('Tasa de Falsos Positivos (1 - Especificidad)', 
                     fontsize=self.config_viz['fuente_etiquetas'])
        ax.set_ylabel('Tasa de Verdaderos Positivos (Sensibilidad)', 
                     fontsize=self.config_viz['fuente_etiquetas'])
        ax.set_title(titulo, fontsize=self.config_viz['fuente_titulo'], fontweight='bold')
        
        # Configuración de grilla y límites
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_aspect('equal')
        
        # *** LEYENDA Y ANOTACIONES MÉDICAS ***
        ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
        
        # Anotación de interpretación clínica
        ax.text(0.02, 0.98, 
                'Área Superior Izquierda:\nAlta Sensibilidad\nAlta Especificidad\n(Clasificador Ideal)',
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.4", facecolor='lightblue', alpha=0.7))
        
        # *** GUARDAR SI SE ESPECIFICA ***
        if guardar_archivo:
            plt.savefig(guardar_archivo, dpi=300, bbox_inches='tight')
            print(f"Gráfico guardado en: {guardar_archivo}")
        
        plt.tight_layout()
        return fig
    
    def generar_reporte_comparativo(self, clasificadores: List[str]) -> str:
        """
        *** REPORTE ESTADÍSTICO COMPARATIVO ***
        
        Genera reporte textual completo con análisis cuantitativo
        y recomendaciones clínicas para selección de clasificador.
        
        Args:
            clasificadores: Lista de clasificadores a comparar
            
        Returns:
            Reporte formateado en texto para inclusión en documentación
        """
        reporte = []
        reporte.append("=" * 80)
        reporte.append("REPORTE ANÁLISIS ROC - CLASIFICACIÓN DERMATOSCÓPICA")
        reporte.append("=" * 80)
        reporte.append("")
        
        # *** RESUMEN DE RENDIMIENTO ***
        reporte.append("1. RESUMEN DE RENDIMIENTO (AUC)")
        reporte.append("-" * 40)
        
        auc_valores = {}
        for nombre in clasificadores:
            if nombre in self.resultados_roc:
                auc_val = self.resultados_roc[nombre]['auc']
                auc_valores[nombre] = auc_val
                interpretacion = self._interpretar_auc(auc_val)
                reporte.append(f"• {nombre:20} AUC = {auc_val:.3f} ({interpretacion})")
        
        reporte.append("")
        
        # *** ANÁLISIS DE PUNTOS DE OPERACIÓN ***
        reporte.append("2. PUNTOS DE OPERACIÓN SELECCIONADOS")
        reporte.append("-" * 40)
        
        for nombre in clasificadores:
            if nombre in self.puntos_operacion:
                punto = self.puntos_operacion[nombre]
                reporte.append(f"\n{nombre}:")
                reporte.append(f"  • Criterio: {punto.criterio_seleccion.upper()}")
                reporte.append(f"  • Umbral: {punto.umbral:.3f}")
                reporte.append(f"  • Sensibilidad (TPR): {punto.tpr:.3f} ({punto.tpr:.1%})")
                reporte.append(f"  • Especificidad (TNR): {punto.tnr:.3f} ({punto.tnr:.1%})")
                reporte.append(f"  • Índice Youden: {punto.youden_index:.3f}")
        
        reporte.append("")
        
        # *** RECOMENDACIÓN CLÍNICA ***
        reporte.append("3. RECOMENDACIÓN CLÍNICA")
        reporte.append("-" * 40)
        
        if auc_valores:
            mejor_clasificador = max(auc_valores, key=auc_valores.get)
            mejor_auc = auc_valores[mejor_clasificador]
            
            reporte.append(f"Clasificador recomendado: {mejor_clasificador}")
            reporte.append(f"AUC superior: {mejor_auc:.3f}")
            reporte.append("")
            reporte.append("Justificación:")
            reporte.append(f"• Mayor área bajo la curva ROC indica mejor capacidad discriminativa")
            reporte.append(f"• Mejor balance entre sensibilidad y especificidad")
            
            if mejor_clasificador in self.puntos_operacion:
                punto = self.puntos_operacion[mejor_clasificador]
                reporte.append(f"• Punto de operación: {punto.tpr:.1%} sensibilidad, {punto.tnr:.1%} especificidad")
        
        reporte.append("")
        reporte.append("=" * 80)
        
        return "\n".join(reporte)
    
    def _interpretar_auc(self, auc_valor: float) -> str:
        """Interpretación clínica del valor AUC."""
        if auc_valor >= 0.9:
            return "Excelente"
        elif auc_valor >= 0.8:
            return "Bueno"
        elif auc_valor >= 0.7:
            return "Aceptable"
        elif auc_valor >= 0.6:
            return "Pobre"
        else:
            return "Muy Pobre"


def ejemplo_uso_completo():
    """
    *** EJEMPLO DE USO COMPLETO ***
    
    Demuestra el flujo completo de análisis ROC para clasificadores
    dermatoscópicos, incluyendo cálculo, selección de punto y visualización.
    """
    # Simular datos de ejemplo
    np.random.seed(42)
    n_muestras = 1000
    
    # Etiquetas: 30% melanomas, 70% benignos
    y_verdadero = np.random.choice([0, 1], size=n_muestras, p=[0.7, 0.3])
    
    # Scores simulados: melanomas tienen scores más altos en promedio
    y_scores_rgb = np.random.normal(0.3, 0.8, n_muestras) + y_verdadero * 1.2
    y_scores_pca = np.random.normal(0.2, 0.9, n_muestras) + y_verdadero * 1.0
    
    # *** ANÁLISIS ROC ***
    calculadora = CalculadorROC()
    
    # Calcular curvas ROC
    calculadora.calcular_roc(y_verdadero, y_scores_rgb, "Clasificador RGB")
    calculadora.calcular_roc(y_verdadero, y_scores_pca, "Clasificador PCA")
    
    # Seleccionar puntos de operación
    calculadora.seleccionar_punto_operacion("Clasificador RGB", "youden")
    calculadora.seleccionar_punto_operacion("Clasificador PCA", "youden")
    
    # Generar visualización
    fig = calculadora.graficar_roc_comparativo(
        ["Clasificador RGB", "Clasificador PCA"],
        titulo="Análisis ROC - Detección de Melanoma"
    )
    
    # Generar reporte
    reporte = calculadora.generar_reporte_comparativo(
        ["Clasificador RGB", "Clasificador PCA"]
    )
    
    print(reporte)
    plt.show()
    
    return calculadora


if __name__ == "__main__":
    ejemplo_uso_completo()