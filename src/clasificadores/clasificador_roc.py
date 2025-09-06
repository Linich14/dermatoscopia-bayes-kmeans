"""
Clasificación por umbrales simples.
"""
import numpy as np
from PIL import Image

def clasificar_umbral(imagen_np):
    """
    Clasifica cada píxel de la imagen según un umbral de color (intensidad de rojo).
    Si la imagen es RGB, usa el canal rojo; si es escala de grises, usa el valor directo.
    Devuelve una matriz binaria: 1 para lesión, 0 para piel sana.
    """
    if len(imagen_np.shape) == 3 and imagen_np.shape[2] >= 3:
        rojo = imagen_np[:,:,0]
    else:
        rojo = imagen_np
    resultado = np.where(rojo > 128, 1, 0)  # 1: lesión, 0: piel sana


    # Pendiente de implementación
