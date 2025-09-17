"""
Módulo de interfaz gráfica para el sistema de dermatoscopia.

Este módulo contiene la interfaz gráfica principal modular con funcionalidad 
de comparación triple integrada, componentes reutilizables, diálogos y estilos.
"""

from .ventana_modular import VentanaPrincipalModular, VentanaPrincipal
from .components import (
    RoundedContainer, RoundedButton, ScrollableFrame, 
    StatusIndicator, ProgressBar, MetricsDisplay, InfoPanel
)
from .dialogs import (
    DialogoBase, DialogoMetricas, DialogoComparacion, DialogoProgreso,
    mostrar_dialogo_metricas, mostrar_dialogo_comparacion, mostrar_dialogo_progreso
)
from .styles import COLORS, DESIGN, STYLES, FONTS

__all__ = [
    # Ventanas principales
    'VentanaPrincipalModular',
    'VentanaPrincipal',  # Alias para compatibilidad
    
    # Componentes
    'RoundedContainer',
    'RoundedButton', 
    'ScrollableFrame',
    'StatusIndicator',
    'ProgressBar',
    'MetricsDisplay',
    'InfoPanel',
    
    # Diálogos
    'DialogoBase',
    'DialogoMetricas',
    'DialogoComparacion', 
    'DialogoProgreso',
    'mostrar_dialogo_metricas',
    'mostrar_dialogo_comparacion',
    'mostrar_dialogo_progreso',
    
    # Estilos y configuración
    'COLORS',
    'DESIGN', 
    'STYLES',
    'FONTS'
]


def crear_ventana_principal():
    """
    Crea y retorna la ventana principal modular con comparador triple integrado.
    
    Returns:
        VentanaPrincipalModular: Instancia de la ventana principal
    """
    return VentanaPrincipalModular


def ejecutar_aplicacion_principal():
    """
    Ejecuta la aplicación con la ventana principal modular.
    
    Nota: Esta función requiere stats_rgb como parámetro.
    Se debe llamar desde main.py con los datos apropiados.
    """
    print("Use main.py para ejecutar la aplicación principal con datos cargados")
    print("La comparación triple está integrada en la interfaz principal")