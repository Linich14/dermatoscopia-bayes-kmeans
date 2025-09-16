"""
Módulo de clasificación Bayesiana refactorizado.

Este paquete contiene la implementación modular del clasificador Bayesiano RGB,
separado en componentes especializados siguiendo principios de código limpio.

Componentes:
- base: Interfaces y clases base abstractas
- modelo: Implementación del modelo gaussiano multivariado
- umbrales: Estrategias de selección de umbral
- evaluacion: Métricas y evaluación de rendimiento
- clasificador: Clase principal que orquesta los componentes
"""

# Importar componentes principales para compatibilidad
try:
    from .clasificador import ClasificadorBayesianoRGB
    from .modelo import ModeloGaussianoMultivariado
    from .umbrales import SelectorUmbral
    from .evaluacion import EvaluadorClasificador
    
    __all__ = [
        'ClasificadorBayesianoRGB',
        'ModeloGaussianoMultivariado', 
        'SelectorUmbral',
        'EvaluadorClasificador'
    ]
except ImportError as e:
    # Fallback para desarrollo
    print(f"Advertencia: Error importando módulos bayesianos: {e}")
    __all__ = []