"""
Filtro Moving Average (Média Móvel)
Substitui cada pixel pela média uniforme de sua vizinhança.
"""
import numpy as np
import cv2


def moving_average(image: np.ndarray, window_size: int) -> np.ndarray:
    """
    Aplica filtro de média móvel na imagem.
    
    Fórmula: g(x,y) = (1/n(W)) · Σ f(i,j) para (i,j) ∈ W(x,y)
    
    Args:
        image: Imagem de entrada (grayscale)
        window_size: Tamanho da janela (deve ser ímpar)
    
    Returns:
        Imagem filtrada
    """
    if window_size % 2 == 0:
        raise ValueError("window_size deve ser ímpar")
    
    # Criar kernel uniforme normalizado
    kernel = np.ones((window_size, window_size), dtype=np.float32) / (window_size ** 2)
    
    # Aplicar filtro de convolução
    filtered = cv2.filter2D(image, -1, kernel)
    
    return filtered
