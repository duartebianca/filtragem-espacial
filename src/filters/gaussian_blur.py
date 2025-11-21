"""
Filtro Gaussian Blur
Aplica convolução Gaussiana 2D para suavização de imagem.
"""
import numpy as np
import cv2


def gaussian_blur(image: np.ndarray, window_size: int) -> np.ndarray:
    """
    Aplica filtro Gaussian Blur na imagem.
    
    Args:
        image: Imagem de entrada (grayscale)
        window_size: Tamanho da janela (deve ser ímpar)
    
    Returns:
        Imagem filtrada
    """
    if window_size % 2 == 0:
        raise ValueError("window_size deve ser ímpar")
    
    # Calcular sigma baseado no tamanho da janela
    # Regra empírica: sigma = 0.3 * ((window_size - 1) * 0.5 - 1) + 0.8
    sigma = 0.3 * ((window_size - 1) * 0.5 - 1) + 0.8
    
    # Aplicar GaussianBlur do OpenCV
    filtered = cv2.GaussianBlur(image, (window_size, window_size), sigma)
    
    return filtered
