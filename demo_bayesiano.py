"""
Script de demostración del Clasificador Bayesiano RGB.
Prueba los diferentes criterios de umbral y muestra los resultados.
"""

import sys
sys.path.append('src')

from preprocesamiento.carga import cargar_imagenes_y_mascaras
from preprocesamiento.particion import particionar_datos
from clasificadores.clasificador_bayesiano import ClasificadorBayesianoRGB
from clasificadores.evaluacion import (
    evaluar_clasificador_en_conjunto, 
    comparar_criterios_umbral,
    imprimir_reporte_evaluacion,
    generar_curva_roc
)
import matplotlib.pyplot as plt
import numpy as np

def demo_clasificador_bayesiano():
    """
    Demuestra el funcionamiento del clasificador bayesiano RGB.
    """
    print("🔬 DEMOSTRACIÓN DEL CLASIFICADOR BAYESIANO RGB")
    print("="*60)
    
    # 1. Cargar y particionar datos
    print("\n📁 Cargando datos...")
    imagenes = cargar_imagenes_y_mascaras()
    train, val, test = particionar_datos(imagenes)
    
    print(f"   • Entrenamiento: {len(train)} imágenes")
    print(f"   • Validación:    {len(val)} imágenes")
    print(f"   • Test:          {len(test)} imágenes")
    
    # 2. Comparar criterios de umbral
    print("\n🎯 Comparando criterios de selección de umbral...")
    resultados_criterios = comparar_criterios_umbral(train, val)
    
    # 3. Mostrar resultados de comparación
    print("\n📊 RESULTADOS DE COMPARACIÓN DE CRITERIOS")
    print("="*60)
    
    for criterio, resultado in resultados_criterios.items():
        print(f"\n🔹 {criterio.upper()}")
        print(f"   Umbral: {resultado['umbral']:.6f}")
        metricas = resultado['metricas']
        print(f"   Exactitud:      {metricas['exactitud']:.4f}")
        print(f"   Sensibilidad:   {metricas['sensibilidad']:.4f}")
        print(f"   Especificidad:  {metricas['especificidad']:.4f}")
        print(f"   Índice Youden:  {metricas['youden']:.4f}")
        print(f"   Jaccard (IoU):  {metricas['jaccard']:.4f}")
    
    # 4. Seleccionar mejor criterio
    mejor_criterio = max(resultados_criterios.keys(), 
                        key=lambda x: resultados_criterios[x]['metricas']['youden'])
    
    print(f"\n🏆 MEJOR CRITERIO: {mejor_criterio.upper()}")
    print(f"   (Máximo índice de Youden: {resultados_criterios[mejor_criterio]['metricas']['youden']:.4f})")
    
    # 5. Entrenar clasificador final
    print(f"\n🤖 Entrenando clasificador final con criterio '{mejor_criterio}'...")
    clasificador_final = ClasificadorBayesianoRGB(criterio_umbral=mejor_criterio)
    clasificador_final.entrenar(train)
    
    # 6. Mostrar parámetros del modelo
    parametros = clasificador_final.obtener_parametros()
    print("\n📋 PARÁMETROS DEL MODELO ENTRENADO")
    print("="*40)
    print(f"Media RGB lesión:  {parametros['mu_lesion']}")
    print(f"Media RGB sana:    {parametros['mu_sana']}")
    print(f"P(lesión):         {parametros['prior_lesion']:.4f}")
    print(f"P(sana):           {parametros['prior_sana']:.4f}")
    print(f"Umbral LR:         {parametros['umbral']:.6f}")
    
    # 7. Evaluación final en test
    print("\n🧪 EVALUACIÓN FINAL EN CONJUNTO DE TEST")
    print("="*45)
    metricas_test = evaluar_clasificador_en_conjunto(clasificador_final, test)
    imprimir_reporte_evaluacion(metricas_test, "EVALUACIÓN FINAL")
    
    # 8. Generar curva ROC
    print("\n📈 Generando curva ROC...")
    try:
        fpr, tpr, auc_score, thresholds = generar_curva_roc(clasificador_final, test)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Clasificador aleatorio')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
        plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad)')
        plt.title('Curva ROC - Clasificador Bayesiano RGB')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        print(f"   ✓ AUC Score: {auc_score:.4f}")
        
    except Exception as e:
        print(f"   ❌ Error generando curva ROC: {e}")
    
    # 9. Justificación del criterio seleccionado
    print(f"\n💡 JUSTIFICACIÓN DEL CRITERIO '{mejor_criterio.upper()}'")
    print("="*50)
    print(clasificador_final.justificar_criterio_umbral())
    
    print(f"\n✅ DEMOSTRACIÓN COMPLETADA")
    print("="*60)
    
    return clasificador_final, metricas_test

if __name__ == "__main__":
    clasificador, metricas = demo_clasificador_bayesiano()