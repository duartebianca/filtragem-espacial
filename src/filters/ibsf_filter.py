"""
IBSF - Interference-Based Speckle Filter
Filtro baseado na física de interferência de ondas ultrassônicas.

Referência: Cardoso et al. (2012) - "Interference-Based Speckle Filter"
"""
import numpy as np
import cv2


def ibsf_filter(image: np.ndarray, window_size_large: int, 
                window_size_small: int = 3) -> np.ndarray:
    """
    Aplica filtro IBSF (ISF - Interference-based Speckle Filter).
    
    O filtro possui 3 passos:
    1. Aplicar mediana com janela grande
    2. Supressão de interferência destrutiva: I_C = max(I, I_Med)
    3. Aplicar mediana com janela pequena (refinamento)
    
    Args:
        image: Imagem de entrada (grayscale)
        window_size_large: Tamanho da janela grande para primeira mediana
        window_size_small: Tamanho da janela pequena (padrão: 3)
    
    Returns:
        Imagem filtrada
    """
    if window_size_large % 2 == 0 or window_size_small % 2 == 0:
        raise ValueError("Tamanhos de janela devem ser ímpares")
    
    # ===== PASSO 1: Median Filter com janela grande =====
    # Remove speckles mantendo estruturas maiores
    I_med_large = cv2.medianBlur(image, window_size_large)
    
    # ===== PASSO 2: Supressão de Interferência Destrutiva =====
    # Seleciona o pixel mais brilhante entre original e mediana
    # Elimina pixels escuros (interferência destrutiva)
    # Preserva pixels brilhantes (interferência construtiva)
    I_constructive = np.maximum(image, I_med_large)
    
    # ===== PASSO 3: Median Filter com janela pequena =====
    # Remove speckles brilhantes isolados restantes
    I_final = cv2.medianBlur(I_constructive, window_size_small)
    
    return I_final


def ibsf_filter_radius(image: np.ndarray, radius_large: int, 
                       radius_small: int = 1) -> np.ndarray:
    """
    Versão alternativa usando raios ao invés de tamanhos de janela.
    
    Args:
        image: Imagem de entrada (grayscale)
        radius_large: Raio da janela grande (window_size = 2*radius + 1)
        radius_small: Raio da janela pequena (padrão: 1 → janela 3x3)
    
    Returns:
        Imagem filtrada
    """
    window_size_large = 2 * radius_large + 1
    window_size_small = 2 * radius_small + 1
    
    return ibsf_filter(image, window_size_large, window_size_small)
