"""
Funciones para cargar imágenes RGB y máscaras binarias.
"""
import os
import numpy as np
from PIL import Image
from config.configuracion import DATA_PATH, IMG_SIZE, EXTS
def cargar_imagenes_y_mascaras():
    """
    Carga pares de imágenes RGB y sus máscaras binarias desde la carpeta data/.
    - Valida extensiones compatibles (.jpg, .png, .bmp)
    - Redimensiona las imágenes y máscaras al tamaño estándar
    - Normaliza cada canal de la imagen RGB a [0,1]
    - Convierte las máscaras a matrices binarias (1=lesión, 0=no-lesión)
    Devuelve una lista de diccionarios con:
        'nombre': nombre base
        'imagen': matriz NumPy RGB normalizada
        'mascara': matriz binaria
    """
    imagenes = []
    archivos = [f for f in os.listdir(DATA_PATH) if f.lower().endswith(EXTS)]
    # Buscar pares: imagen y máscara (ejemplo: ISIC_0001.jpg y ISIC_0001_expert.png)
    for archivo in archivos:
        if '_expert' in archivo or '_mask' in archivo:
            continue  # Saltar máscaras en este paso
        nombre_base = os.path.splitext(archivo)[0]
        # Buscar máscara asociada
        mascara_archivo = None
        for ext in EXTS:
            posible = f"{nombre_base}_expert{ext}"
            if posible in archivos:
                mascara_archivo = posible
                break
            posible2 = f"{nombre_base}_mask{ext}"
            if posible2 in archivos:
                mascara_archivo = posible2
                break
        if mascara_archivo is None:
            print(f"No se encontró máscara para {archivo}")
            continue
        # Cargar imagen RGB
        ruta_img = os.path.join(DATA_PATH, archivo)
        img = Image.open(ruta_img).convert('RGB').resize(IMG_SIZE)
        arr_img = np.array(img) / 255.0  # Normalización por canal
        # Cargar máscara binaria
        ruta_mask = os.path.join(DATA_PATH, mascara_archivo)
        mask = Image.open(ruta_mask).convert('L').resize(IMG_SIZE)
        arr_mask = np.array(mask)
        arr_mask = np.where(arr_mask > 127, 1, 0)  # binarizar
        imagenes.append({
            'nombre': nombre_base,
            'imagen': arr_img,
            'mascara': arr_mask
        })
    return imagenes