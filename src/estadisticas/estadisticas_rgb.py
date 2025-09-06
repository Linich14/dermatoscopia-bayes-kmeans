"""
Funciones para calcular histogramas y estadísticos de R, G, B para áreas de lesión y no-lesión.
"""
import numpy as np
def estadisticas_rgb(imagenes):
    """
    Calcula histogramas y estadísticos (media, std) de R, G, B para píxeles de lesión y no-lesión.
    Devuelve un diccionario con los resultados por clase y canal.
    """
    lesion = {'R': [], 'G': [], 'B': []}
    sana = {'R': [], 'G': [], 'B': []}
    for item in imagenes:
        img = item['imagen']  # (H,W,3) normalizada
        mask = item['mascara']  # (H,W) binaria
        # Lesión
        idx_lesion = np.where(mask == 1)
        lesion['R'].extend(img[idx_lesion[0], idx_lesion[1], 0])
        lesion['G'].extend(img[idx_lesion[0], idx_lesion[1], 1])
        lesion['B'].extend(img[idx_lesion[0], idx_lesion[1], 2])
        # No-lesión
        idx_sana = np.where(mask == 0)
        sana['R'].extend(img[idx_sana[0], idx_sana[1], 0])
        sana['G'].extend(img[idx_sana[0], idx_sana[1], 1])
        sana['B'].extend(img[idx_sana[0], idx_sana[1], 2])
    # Calcular estadísticos
    stats = {}
    for clase, datos in zip(['lesion', 'sana'], [lesion, sana]):
        stats[clase] = {}
        for canal in ['R', 'G', 'B']:
            arr = np.array(datos[canal])
            stats[clase][canal] = {
                'media': float(np.mean(arr)) if arr.size > 0 else None,
                'std': float(np.std(arr)) if arr.size > 0 else None,
                'histograma': np.histogram(arr, bins=20, range=(0,1))[0].tolist() if arr.size > 0 else None
            }
    return stats