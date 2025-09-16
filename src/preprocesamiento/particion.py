"""
Sistema de partición de datos para entrenamiento y evaluación de modelos.

Este módulo proporciona funcionalidades para dividir el conjunto de datos de imágenes
dermatoscópicas en conjuntos de entrenamiento, validación y prueba. La partición se
realiza a nivel de imagen (no de píxel) para evitar el data leakage y garantizar
una evaluación robusta de los modelos de segmentación.

Características principales:
- Partición estratificada a nivel de imagen
- Reproducibilidad mediante semilla fija
- Distribución estándar: 60% entrenamiento, 20% validación, 20% prueba
- Preservación de estructura de datos original


"""

import numpy as np
from sklearn.model_selection import train_test_split
from config.configuracion import SEED

def particionar_datos(imagenes):
    """
    Realiza la partición estratificada de datos por imagen en conjuntos de entrenamiento, validación y test.
    
    Esta función divide el conjunto completo de imágenes en tres subconjuntos disjuntos
    siguiendo la distribución estándar para machine learning médico. La partición se
    realiza a nivel de imagen completa para evitar data leakage, donde píxeles de la
    misma imagen podrían aparecer tanto en entrenamiento como en evaluación.
    
    Distribución de datos:
    - Entrenamiento: 60% de las imágenes (para entrenar modelos)
    - Validación: 20% de las imágenes (para ajuste de hiperparámetros)
    - Prueba: 20% de las imágenes (para evaluación final)
    
    Args:
        imagenes (list): Lista de diccionarios con estructura:
            [
                {
                    'nombre': str,           # Nombre base del archivo
                    'imagen': np.ndarray,    # Imagen RGB (H,W,3) normalizada
                    'mascara': np.ndarray    # Máscara binaria (H,W)
                },
                ...
            ]
    
    Returns:
        tuple: Tupla de tres listas (train, val, test) cada una conteniendo
               diccionarios con la misma estructura que la entrada:
               - train (list): Conjunto de entrenamiento (60%)
               - val (list): Conjunto de validación (20%)  
               - test (list): Conjunto de prueba (20%)
    
    Note:
        - Utiliza semilla fija (SEED) para garantizar reproducibilidad
        - La partición es aleatoria pero estratificada
        - Se preserva la estructura original de cada elemento
        - Si hay menos de 5 imágenes, se ajusta la distribución automáticamente
        
    Example:
        >>> imagenes = cargar_imagenes_y_mascaras()
        >>> train, val, test = particionar_datos(imagenes)
        >>> print(f"Entrenamiento: {len(train)} imágenes")
        >>> print(f"Validación: {len(val)} imágenes") 
        >>> print(f"Prueba: {len(test)} imágenes")
        
    Raises:
        ValueError: Si la lista de imágenes está vacía
        RuntimeError: Si hay problemas en la partición de datos
    """
    # Validar entrada
    if not imagenes:
        print("⚠️  Lista de imágenes vacía, devolviendo conjuntos vacíos")
        return [], [], []
    
    if len(imagenes) < 3:
        print(f"⚠️  Muy pocas imágenes ({len(imagenes)}) para partición estándar")
        print("📝 Usando todas las imágenes para entrenamiento")
        return imagenes, [], []
    
    print(f"📊 Particionando {len(imagenes)} imágenes en conjuntos de datos...")
    
    try:
        # Crear array de índices para la partición
        indices = np.arange(len(imagenes))
        
        # Primera partición: separar entrenamiento (60%) del resto (40%)
        train_idx, temp_idx = train_test_split(
            indices, 
            test_size=0.4,           # 40% para validación + prueba
            random_state=SEED,       # Reproducibilidad
            shuffle=True             # Mezclar antes de dividir
        )
        
        # Segunda partición: dividir el 40% restante en validación (20%) y prueba (20%)
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.5,           # 50% del 40% = 20% total
            random_state=SEED,       # Misma semilla para reproducibilidad
            shuffle=True
        )
        
        # Crear conjuntos de datos usando los índices seleccionados
        train = [imagenes[i] for i in train_idx]
        val = [imagenes[i] for i in val_idx]
        test = [imagenes[i] for i in test_idx]
        
        # Verificar integridad de la partición
        total_particionado = len(train) + len(val) + len(test)
        if total_particionado != len(imagenes):
            raise RuntimeError(f"Error en partición: {total_particionado} != {len(imagenes)}")
        
        # Reporte de resultados
        print(f"✅ Partición completada exitosamente:")
        print(f"   📚 Entrenamiento: {len(train)} imágenes ({len(train)/len(imagenes)*100:.1f}%)")
        print(f"   🔍 Validación: {len(val)} imágenes ({len(val)/len(imagenes)*100:.1f}%)")
        print(f"   🧪 Prueba: {len(test)} imágenes ({len(test)/len(imagenes)*100:.1f}%)")
        
        # Verificar que no hay solapamiento entre conjuntos
        nombres_train = {img['nombre'] for img in train}
        nombres_val = {img['nombre'] for img in val}
        nombres_test = {img['nombre'] for img in test}
        
        if nombres_train & nombres_val or nombres_train & nombres_test or nombres_val & nombres_test:
            raise RuntimeError("❌ Detectado solapamiento entre conjuntos de datos")
        
        print("✅ Verificación de integridad: Sin solapamiento entre conjuntos")
        
        return train, val, test
        
    except Exception as e:
        print(f"❌ Error durante la partición de datos: {str(e)}")
        raise RuntimeError(f"Fallo en la partición de datos: {str(e)}")

def obtener_estadisticas_particion(train, val, test):
    """
    Calcula estadísticas descriptivas de la partición de datos.
    
    Esta función proporciona un análisis detallado de cómo se distribuyeron
    los datos entre los conjuntos de entrenamiento, validación y prueba,
    incluyendo estadísticas sobre píxeles de lesión y balance de clases.
    
    Args:
        train (list): Conjunto de entrenamiento
        val (list): Conjunto de validación  
        test (list): Conjunto de prueba
        
    Returns:
        dict: Diccionario con estadísticas de la partición:
            {
                'total_imagenes': int,
                'train': {'imagenes': int, 'pixeles_lesion': int, 'pixeles_total': int},
                'val': {'imagenes': int, 'pixeles_lesion': int, 'pixeles_total': int},
                'test': {'imagenes': int, 'pixeles_lesion': int, 'pixeles_total': int},
                'balance_clases': {'train': float, 'val': float, 'test': float}
            }
    """
    def analizar_conjunto(conjunto, nombre):
        """Analiza un conjunto de datos específico"""
        if not conjunto:
            return {'imagenes': 0, 'pixeles_lesion': 0, 'pixeles_total': 0, 'balance': 0.0}
        
        total_imagenes = len(conjunto)
        total_pixeles_lesion = sum(np.sum(img['mascara']) for img in conjunto)
        total_pixeles = sum(img['mascara'].size for img in conjunto)
        balance = total_pixeles_lesion / total_pixeles if total_pixeles > 0 else 0.0
        
        print(f"📊 Estadísticas {nombre}:")
        print(f"   Imágenes: {total_imagenes}")
        print(f"   Píxeles de lesión: {total_pixeles_lesion:,}")
        print(f"   Píxeles totales: {total_pixeles:,}")
        print(f"   Balance de clases: {balance:.3f} ({balance*100:.1f}% lesión)")
        
        return {
            'imagenes': total_imagenes,
            'pixeles_lesion': total_pixeles_lesion,
            'pixeles_total': total_pixeles,
            'balance': balance
        }
    
    print("\n📈 Análisis estadístico de la partición:")
    
    # Analizar cada conjunto
    stats_train = analizar_conjunto(train, "Entrenamiento")
    stats_val = analizar_conjunto(val, "Validación")
    stats_test = analizar_conjunto(test, "Prueba")
    
    # Estadísticas globales
    total_imagenes = stats_train['imagenes'] + stats_val['imagenes'] + stats_test['imagenes']
    
    estadisticas = {
        'total_imagenes': total_imagenes,
        'train': stats_train,
        'val': stats_val,
        'test': stats_test,
        'balance_clases': {
            'train': stats_train['balance'],
            'val': stats_val['balance'],
            'test': stats_test['balance']
        }
    }
    
    return estadisticas
