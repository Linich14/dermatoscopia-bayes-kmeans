"""
Sistema de carga de imágenes dermatoscópicas y máscaras de segmentación.

Este módulo proporciona funcionalidades para cargar y preprocesar pares de imágenes
dermatoscópicas RGB y sus correspondientes máscaras de segmentación ground truth.
Se encarga de la normalización, redimensionamiento y validación de datos para
garantizar consistencia en el procesamiento posterior.

Funcionalidades principales:
- Carga automática de pares imagen-máscara desde directorio de datos
- Redimensionamiento estándar de imágenes y máscaras 
- Normalización de valores RGB al rango [0,1]
- Binarización de máscaras de segmentación
- Validación de integridad de datos


"""

import os
import numpy as np
from PIL import Image
from config.configuracion import DATA_PATH, IMG_SIZE, EXTS

def cargar_imagenes_y_mascaras():
    """
    Carga pares de imágenes RGB y sus máscaras binarias desde el directorio de datos.
    
    Esta función realiza el proceso completo de carga y preprocesamiento de datos:
    1. Escanea el directorio de datos buscando pares imagen-máscara
    2. Valida extensiones de archivo compatibles
    3. Redimensiona imágenes y máscaras al tamaño estándar
    4. Normaliza valores RGB al rango [0,1]
    5. Binariza máscaras de segmentación
    
    La función busca automáticamente máscaras asociadas a cada imagen usando
    patrones de nomenclatura estándar:
    - ISIC_XXXXX.jpg → ISIC_XXXXX_expert.png (máscara de experto)
    - ISIC_XXXXX.jpg → ISIC_XXXXX_mask.png (máscara general)
    
    Returns:
        list: Lista de diccionarios, cada uno conteniendo:
            - 'nombre' (str): Nombre base del archivo sin extensión
            - 'imagen' (np.ndarray): Matriz RGB de forma (H,W,3) normalizada [0,1]
            - 'mascara' (np.ndarray): Matriz binaria de forma (H,W) donde:
                1 = píxel de lesión
                0 = píxel de piel sana
                
    Raises:
        FileNotFoundError: Si el directorio de datos no existe
        IOError: Si hay problemas al cargar las imágenes
        
    Note:
        - Las imágenes se redimensionan según IMG_SIZE definido en configuración
        - La normalización RGB se realiza dividiendo por 255.0
        - Las máscaras se binarizan usando umbral de 127 (0.5 en escala [0,255])
        - Se registran advertencias para imágenes sin máscara asociada
        
    Example:
        >>> imagenes = cargar_imagenes_y_mascaras()
        >>> print(f"Cargadas {len(imagenes)} imágenes")
        >>> imagen = imagenes[0]
        >>> print(f"Forma imagen: {imagen['imagen'].shape}")
        >>> print(f"Rango valores: [{imagen['imagen'].min():.3f}, {imagen['imagen'].max():.3f}]")
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"El directorio de datos no existe: {DATA_PATH}")
    
    imagenes = []
    
    # Obtener lista de archivos con extensiones válidas
    archivos = [f for f in os.listdir(DATA_PATH) if f.lower().endswith(EXTS)]
    print(f"Encontrados {len(archivos)} archivos con extensiones válidas en {DATA_PATH}")
    
    # Buscar pares imagen-máscara
    # Formato esperado: imagen.jpg + imagen_expert.png (o _mask.png)
    imagenes_procesadas = 0
    mascaras_no_encontradas = 0
    
    for archivo in archivos:
        # Saltar archivos que ya son máscaras
        if '_expert' in archivo or '_mask' in archivo:
            continue
        
        # Obtener nombre base sin extensión
        nombre_base = os.path.splitext(archivo)[0]
        
        # Buscar máscara asociada con diferentes patrones de nomenclatura
        mascara_archivo = None
        patrones_mascara = [f"{nombre_base}_expert", f"{nombre_base}_mask"]
        
        for patron in patrones_mascara:
            for ext in EXTS:
                posible_mascara = f"{patron}{ext}"
                if posible_mascara in archivos:
                    mascara_archivo = posible_mascara
                    break
            if mascara_archivo:
                break
        
        # Si no se encuentra máscara, reportar y continuar
        if mascara_archivo is None:
            print(f"⚠️  No se encontró máscara para {archivo}")
            mascaras_no_encontradas += 1
            continue
        
        try:
            # Cargar y preprocesar imagen RGB
            ruta_img = os.path.join(DATA_PATH, archivo)
            img_pil = Image.open(ruta_img).convert('RGB')
            
            # Redimensionar a tamaño estándar
            img_pil = img_pil.resize(IMG_SIZE, Image.Resampling.LANCZOS)
            
            # Convertir a array numpy y normalizar [0,1]
            arr_img = np.array(img_pil) / 255.0
            
            # Cargar y preprocesar máscara binaria
            ruta_mask = os.path.join(DATA_PATH, mascara_archivo)
            mask_pil = Image.open(ruta_mask).convert('L')  # Convertir a escala de grises
            
            # Redimensionar máscara al mismo tamaño
            mask_pil = mask_pil.resize(IMG_SIZE, Image.Resampling.NEAREST)
            
            # Convertir a array y binarizar
            arr_mask = np.array(mask_pil)
            arr_mask = np.where(arr_mask > 127, 1, 0).astype(np.uint8)
            
            # Validar dimensiones
            if arr_img.shape[:2] != arr_mask.shape:
                raise ValueError(f"Dimensiones no coinciden: imagen {arr_img.shape[:2]}, máscara {arr_mask.shape}")
            
            # Agregar a la lista de resultados
            imagenes.append({
                'nombre': nombre_base,
                'imagen': arr_img,      # Array (H,W,3) normalizado [0,1]
                'mascara': arr_mask     # Array (H,W) binario {0,1}
            })
            
            imagenes_procesadas += 1
            
        except (IOError, OSError, ValueError) as e:
            print(f"❌ Error procesando {archivo}: {str(e)}")
            continue
        except Exception as e:
            print(f"❌ Error inesperado procesando {archivo}: {str(e)}")
            continue
    
    # Reporte final de carga
    print(f"\n📊 Resumen de carga de datos:")
    print(f"✅ Imágenes procesadas exitosamente: {imagenes_procesadas}")
    print(f"⚠️  Imágenes sin máscara asociada: {mascaras_no_encontradas}")
    print(f"📁 Total de pares imagen-máscara cargados: {len(imagenes)}")
    
    if len(imagenes) == 0:
        print("❌ No se pudieron cargar imágenes. Verifique el directorio de datos y nomenclatura.")
    
    return imagenes