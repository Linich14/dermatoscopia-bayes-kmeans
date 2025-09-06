"""
Funciones para muestreo equilibrado de píxeles por clase usando máscaras.
"""
import numpy as np
from config.configuracion import SEED


def muestreo_equilibrado(imagenes, n_muestra=1000, semilla=SEED):
    """
    Realiza muestreo equilibrado de píxeles para entrenamiento:
    - Extrae igual cantidad de píxeles de cada clase (lesión/no-lesión) usando la máscara.
    - n_muestra: cantidad total de píxeles a muestrear por clase (por imagen)
    Devuelve dos arrays: X (píxeles RGB), y (clase)
    """
    np.random.seed(semilla)
    X, y = [], []
    
    print(f"🎯 Muestreo equilibrado con semilla: {semilla}")
    
    for i, item in enumerate(imagenes):
        img = item['imagen']  # (H,W,3) normalizada
        mask = item['mascara']  # (H,W) binaria
        
        # Índices de cada clase
        idx_lesion = np.argwhere(mask == 1)
        idx_sana = np.argwhere(mask == 0)
        
        n_lesion = min(len(idx_lesion), n_muestra)
        n_sana = min(len(idx_sana), n_muestra)
        
        # Muestreo aleatorio
        if n_lesion > 0:
            sel_lesion = idx_lesion[np.random.choice(len(idx_lesion), n_lesion, replace=False)]
            for idx in sel_lesion:
                X.append(img[idx[0], idx[1], :])
                y.append(1)
        
        if n_sana > 0:
            sel_sana = idx_sana[np.random.choice(len(idx_sana), n_sana, replace=False)]
            for idx in sel_sana:
                X.append(img[idx[0], idx[1], :])
                y.append(0)
        
        if (i + 1) % 20 == 0:
            print(f"  Procesadas {i + 1}/{len(imagenes)} imágenes...")
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"✅ Muestreo completado: {len(X)} píxeles totales")
    print(f"   Lesión: {np.sum(y)} píxeles ({np.mean(y):.1%})")
    print(f"   No-lesión: {len(y) - np.sum(y)} píxeles ({1-np.mean(y):.1%})")
    
    return X, y
