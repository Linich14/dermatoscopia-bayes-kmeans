"""
Diálogos especializados para evaluación y comparación de modelos.

Este módulo contiene diálogos especializados que muestran resultados
de evaluación de modelos y permiten comparaciones detalladas.
"""

import tkinter as tk
from tkinter import ttk
from typing import Dict, Any
from ..styles import COLORS, STYLES, DESIGN
from ..components import RoundedContainer, ScrollableFrame


class EvaluationDialog(tk.Toplevel):
    """
    Diálogo para mostrar resultados de evaluación de modelos.
    
    Muestra métricas completas, interpretación automática y
    visualización de la matriz de confusión.
    """
    
    def __init__(self, parent, metricas: Dict[str, Any], modelo_info: Dict[str, Any] = None):
        """
        Inicializa el diálogo de evaluación.
        
        Args:
            parent: Ventana padre
            metricas: Diccionario con métricas de evaluación
            modelo_info: Información adicional del modelo (opcional)
        """
        super().__init__(parent)
        self.title("Resultados de Evaluación")
        self.geometry("600x700")
        self.configure(bg=COLORS['background'])
        
        self.metricas = metricas
        self.modelo_info = modelo_info or {}
        
        self._create_widgets()
        
        # Centrar en la ventana padre
        self.transient(parent)
        self.grab_set()
    
    def _create_widgets(self):
        """Crea los widgets del diálogo."""
        # Frame principal scrollable
        main_frame = ScrollableFrame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        content = main_frame.scrollable_frame
        
        # Título
        title_label = tk.Label(content,
                              text="📊 Resultados de Evaluación",
                              font=('Segoe UI', 16, 'bold'),
                              fg=COLORS['primary'],
                              bg=COLORS['background'])
        title_label.pack(pady=(10, 20))
        
        # Información del modelo
        if self.modelo_info:
            self._create_model_info_section(content)
        
        # Métricas principales
        self._create_metrics_section(content)
        
        # Matriz de confusión
        self._create_confusion_matrix_section(content)
        
        # Interpretación
        self._create_interpretation_section(content)
        
        # Botón cerrar
        self._create_close_button()
    
    def _create_model_info_section(self, parent):
        """Crea la sección de información del modelo."""
        model_frame = RoundedContainer(parent, background=COLORS['card_bg'])
        model_frame.pack(fill=tk.X, padx=20, pady=10)
        
        content = model_frame.inner_frame
        
        tk.Label(content,
                text="🤖 Modelo Evaluado",
                font=('Segoe UI', 12, 'bold'),
                fg=COLORS['primary'],
                bg=COLORS['card_bg']).pack(pady=(10, 5))
        
        info_text = ""
        if 'criterio_umbral' in self.modelo_info:
            info_text += f"Criterio: {self.modelo_info['criterio_umbral'].upper()}"
        if 'umbral' in self.modelo_info:
            info_text += f"Umbral: {self.modelo_info['umbral']:.6f}"
        if 'prior_lesion' in self.modelo_info:
            info_text += f"P(lesión): {self.modelo_info['prior_lesion']:.3f}"
        
        if info_text:
            tk.Label(content,
                    text=info_text.strip(),
                    font=('Consolas', 10),
                    fg=COLORS['text'],
                    bg=COLORS['card_bg'],
                    justify=tk.LEFT).pack(pady=(0, 10))
    
    def _create_metrics_section(self, parent):
        """Crea la sección de métricas principales."""
        metrics_frame = RoundedContainer(parent, background=COLORS['card_bg'])
        metrics_frame.pack(fill=tk.X, padx=20, pady=10)
        
        content = metrics_frame.inner_frame
        
        tk.Label(content,
                text="📈 Métricas de Rendimiento",
                font=('Segoe UI', 12, 'bold'),
                fg=COLORS['primary'],
                bg=COLORS['card_bg']).pack(pady=(10, 5))
        
        metrics_text = f"""
Exactitud:      {self.metricas['exactitud']:.4f} ({self.metricas['exactitud']*100:.1f}%)
Precisión:      {self.metricas['precision']:.4f} ({self.metricas['precision']*100:.1f}%)
Sensibilidad:   {self.metricas['sensibilidad']:.4f} ({self.metricas['sensibilidad']*100:.1f}%)
Especificidad:  {self.metricas['especificidad']:.4f} ({self.metricas['especificidad']*100:.1f}%)
F1-Score:       {self.metricas['f1_score']:.4f}
Índice Jaccard: {self.metricas['jaccard']:.4f} ({self.metricas['jaccard']*100:.1f}%)
Índice Youden:  {self.metricas['youden']:.4f}
        """
        
        tk.Label(content,
                text=metrics_text.strip(),
                font=('Consolas', 11),
                fg=COLORS['text'],
                bg=COLORS['card_bg'],
                justify=tk.LEFT).pack(pady=(0, 10))
    
    def _create_confusion_matrix_section(self, parent):
        """Crea la sección de matriz de confusión."""
        matrix_frame = RoundedContainer(parent, background=COLORS['card_bg'])
        matrix_frame.pack(fill=tk.X, padx=20, pady=10)
        
        content = matrix_frame.inner_frame
        
        tk.Label(content,
                text="🔢 Matriz de Confusión",
                font=('Segoe UI', 12, 'bold'),
                fg=COLORS['primary'],
                bg=COLORS['card_bg']).pack(pady=(10, 5))
        
        mc = self.metricas['matriz_confusion']
        matrix_text = f"""
           Predicción
         Lesión    Sana
Real Lesión  {mc['TP']:6d}  {mc['FN']:6d}
     Sana    {mc['FP']:6d}  {mc['TN']:6d}
        """
        
        tk.Label(content,
                text=matrix_text.strip(),
                font=('Consolas', 11),
                fg=COLORS['text'],
                bg=COLORS['card_bg'],
                justify=tk.CENTER).pack(pady=(0, 10))
    
    def _create_interpretation_section(self, parent):
        """Crea la sección de interpretación automática."""
        interp_frame = RoundedContainer(parent, background=COLORS['accent_light'])
        interp_frame.pack(fill=tk.X, padx=20, pady=10)
        
        content = interp_frame.inner_frame
        
        tk.Label(content,
                text="💡 Interpretación Automática",
                font=('Segoe UI', 12, 'bold'),
                fg=COLORS['primary'],
                bg=COLORS['accent_light']).pack(pady=(10, 5))
        
        # Generar interpretación automática
        interpretation = self._generate_interpretation()
        
        tk.Label(content,
                text=interpretation,
                font=('Segoe UI', 10),
                fg=COLORS['text'],
                bg=COLORS['accent_light'],
                wraplength=500,
                justify=tk.LEFT).pack(pady=(0, 10), padx=10)
    
    def _generate_interpretation(self) -> str:
        """Genera interpretación automática de las métricas."""
        youden = self.metricas['youden']
        sensibilidad = self.metricas['sensibilidad']
        especificidad = self.metricas['especificidad']
        precision = self.metricas['precision']
        jaccard = self.metricas['jaccard']
        
        # Calificar rendimiento general
        if youden > 0.7:
            calidad = "Excelente"
        elif youden > 0.5:
            calidad = "Bueno"
        elif youden > 0.3:
            calidad = "Moderado"
        else:
            calidad = "Pobre"
        
        interpretation = f"""
• Rendimiento general: {calidad} (Índice Youden = {youden:.3f})

• De cada 100 lesiones reales, el modelo detecta {sensibilidad*100:.0f} correctamente
• De cada 100 píxeles sanos, clasifica {especificidad*100:.0f} correctamente  
• De cada 100 detecciones positivas, {precision*100:.0f} son correctas
• Solapamiento con ground truth: {jaccard*100:.1f}% (Jaccard Index)

• Balance sensibilidad/especificidad: {"Equilibrado" if abs(sensibilidad - especificidad) < 0.1 else "Desbalanceado"}
        """
        
        # Agregar recomendaciones específicas
        if sensibilidad < 0.7:
            interpretation += "\\n⚠️ Sensibilidad baja: considere ajustar umbral para detectar más lesiones"
        if especificidad < 0.7:
            interpretation += "\\n⚠️ Especificidad baja: considere ajustar umbral para reducir falsos positivos"
        if precision < 0.5:
            interpretation += "\\n⚠️ Precisión baja: muchas detecciones son falsos positivos"
        
        return interpretation.strip()
    
    def _create_close_button(self):
        """Crea el botón de cerrar."""
        button_frame = tk.Frame(self, bg=COLORS['background'])
        button_frame.pack(fill=tk.X, pady=10)
        
        tk.Button(button_frame,
                 text="Cerrar",
                 command=self.destroy,
                 bg=COLORS['primary'],
                 fg='white',
                 font=('Segoe UI', 11),
                 padx=20,
                 pady=5).pack()


class ComparisonDialog(tk.Toplevel):
    """
    Diálogo para comparar múltiples criterios de umbral.
    
    Muestra comparación lado a lado de diferentes criterios
    con ranking automático y recomendaciones.
    """
    
    def __init__(self, parent, resultados: Dict[str, Dict[str, Any]]):
        """
        Inicializa el diálogo de comparación.
        
        Args:
            parent: Ventana padre
            resultados: Diccionario con resultados para cada criterio
        """
        super().__init__(parent)
        self.title("Comparación de Criterios de Umbral")
        self.geometry("900x800")
        self.configure(bg=COLORS['background'])
        
        self.resultados = resultados
        
        self._create_widgets()
        
        # Centrar en la ventana padre
        self.transient(parent)
        self.grab_set()
    
    def _create_widgets(self):
        """Crea los widgets del diálogo."""
        # Frame principal scrollable
        main_frame = ScrollableFrame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        content = main_frame.scrollable_frame
        
        # Título
        title_label = tk.Label(content,
                              text="⚖️ Comparación de Criterios de Umbral",
                              font=('Segoe UI', 16, 'bold'),
                              fg=COLORS['primary'],
                              bg=COLORS['background'])
        title_label.pack(pady=(10, 20))
        
        # Ordenar criterios por rendimiento
        criterios_ordenados = sorted(self.resultados.items(), 
                                   key=lambda x: x[1]['metricas']['youden'], 
                                   reverse=True)
        
        # Crear tarjetas para cada criterio
        for i, (criterio, resultado) in enumerate(criterios_ordenados):
            self._create_criterion_card(content, criterio, resultado, i)
        
        # Sección de recomendación final
        self._create_recommendation_section(content, criterios_ordenados)
        
        # Botón cerrar
        self._create_close_button()
    
    def _create_criterion_card(self, parent, criterio: str, resultado: Dict[str, Any], rank: int):
        """Crea una tarjeta para un criterio específico."""
        # Frame del criterio con color según ranking
        bg_color = COLORS['accent_light'] if rank == 0 else COLORS['card_bg']
        
        criterion_frame = RoundedContainer(parent, background=bg_color)
        criterion_frame.pack(fill=tk.X, padx=20, pady=10)
        
        content = criterion_frame.inner_frame
        
        # Título con ranking
        ranking_emoji = "🥇" if rank == 0 else "🥈" if rank == 1 else "🥉"
        title = f"{ranking_emoji} #{rank+1} - {criterio.upper()}"
        
        tk.Label(content,
                text=title,
                font=('Segoe UI', 14, 'bold'),
                fg=COLORS['primary'] if rank == 0 else COLORS['text'],
                bg=bg_color).pack(pady=(10, 5))
        
        # Umbral
        tk.Label(content,
                text=f"Umbral óptimo: {resultado['umbral']:.6f}",
                font=('Segoe UI', 11, 'bold'),
                fg=COLORS['text'],
                bg=bg_color).pack()
        
        # Métricas en formato compacto
        metricas = resultado['metricas']
        metrics_text = f"""Exactitud: {metricas['exactitud']:.3f} | Youden: {metricas['youden']:.3f} | F1: {metricas['f1_score']:.3f}
Sensibilidad: {metricas['sensibilidad']:.3f} | Especificidad: {metricas['especificidad']:.3f} | Jaccard: {metricas['jaccard']:.3f}"""
        
        tk.Label(content,
                text=metrics_text,
                font=('Consolas', 10),
                fg=COLORS['text'],
                bg=bg_color,
                justify=tk.CENTER).pack(pady=5)
        
        # Justificación del criterio
        if 'justificacion' in resultado:
            justificacion = resultado['justificacion'][:150] + "..." if len(resultado['justificacion']) > 150 else resultado['justificacion']
            
            tk.Label(content,
                    text=f"💡 {justificacion}",
                    font=('Segoe UI', 9),
                    fg=COLORS['secondary'],
                    bg=bg_color,
                    wraplength=800,
                    justify=tk.LEFT).pack(pady=(5, 10), padx=10)
    
    def _create_recommendation_section(self, parent, criterios_ordenados):
        """Crea la sección de recomendación final."""
        mejor_criterio = criterios_ordenados[0][0]
        mejor_resultado = criterios_ordenados[0][1]
        
        rec_frame = RoundedContainer(parent, background=COLORS['success'], highlightthickness=2, highlightbackground=COLORS['primary'])
        rec_frame.pack(fill=tk.X, padx=20, pady=20)
        
        content = rec_frame.inner_frame
        
        tk.Label(content,
                text="🎯 Recomendación Final",
                font=('Segoe UI', 14, 'bold'),
                fg='white',
                bg=COLORS['success']).pack(pady=(10, 5))
        
        rec_text = f"""CRITERIO RECOMENDADO: {mejor_criterio.upper()}

Índice de Youden: {mejor_resultado['metricas']['youden']:.3f}
Justificación: Este criterio ofrece el mejor balance entre sensibilidad y especificidad,
maximizando la capacidad discriminativa del clasificador para aplicaciones médicas."""
        
        if len(criterios_ordenados) > 1:
            segundo_criterio = criterios_ordenados[1][0]
            segundo_resultado = criterios_ordenados[1][1]
            diferencia = mejor_resultado['metricas']['youden'] - segundo_resultado['metricas']['youden']
            rec_text += f"\\n\\nVentaja sobre {segundo_criterio}: +{diferencia:.3f} puntos en índice Youden"
        
        tk.Label(content,
                text=rec_text,
                font=('Segoe UI', 10),
                fg='white',
                bg=COLORS['success'],
                wraplength=800,
                justify=tk.LEFT).pack(pady=(0, 10), padx=15)
    
    def _create_close_button(self):
        """Crea el botón de cerrar."""
        button_frame = tk.Frame(self, bg=COLORS['background'])
        button_frame.pack(fill=tk.X, pady=10)
        
        tk.Button(button_frame,
                 text="Cerrar",
                 command=self.destroy,
                 bg=COLORS['primary'],
                 fg='white',
                 font=('Segoe UI', 11),
                 padx=20,
                 pady=5).pack()


__all__ = ['EvaluationDialog', 'ComparisonDialog']