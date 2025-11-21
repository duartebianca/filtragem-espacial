"""
Filtro de Mediana
Substitui cada pixel pela mediana dos valores na janela.
Preserva bordas melhor que filtros lineares.
"""
import numpy as np
import cv2


def median_filter(image: np.ndarray, window_size: int) -> np.ndarray:
    """
    Aplica filtro de mediana na imagem.
    
    O filtro de mediana é excelente para:
    - Ruído impulsivo (salt & pepper)
    - Preservação de bordas
    
    Args:
        image: Imagem de entrada (grayscale)
        window_size: Tamanho da janela (deve ser ímpar)
    
    Returns:
        Imagem filtrada
    """
    if window_size % 2 == 0:
        raise ValueError("window_size deve ser ímpar")
    
    # Aplicar filtro de mediana do OpenCV
    filtered = cv2.medianBlur(image, window_size)
    
    return filtered
