"""
Harris Corner Detection
Detecta pontos de junção (corners) na imagem.

Referência: Burger & Burge - "Core Algorithms Vol. 2"
"""
import numpy as np
import cv2
from typing import Tuple


def detect_harris_corners(image: np.ndarray, 
                          block_size: int = 2,
                          ksize: int = 3,
                          k: float = 0.04,
                          threshold: float = 0.01) -> Tuple[int, np.ndarray]:
    """
    Detecta corners usando o algoritmo de Harris.
    
    Algoritmo:
    1. Calcular gradientes Ix e Iy (usando Sobel)
    2. Calcular produtos Ix², Iy², Ixy
    3. Aplicar suavização Gaussiana
    4. Calcular resposta de Harris: R = det(M) - k·trace(M)²
       onde M é a matriz de estrutura local
    5. Aplicar threshold e non-maximum suppression
    
    Args:
        image: Imagem de entrada (grayscale)
        block_size: Tamanho da vizinhança para derivada (padrão: 2)
        ksize: Abertura do operador Sobel (padrão: 3)
        k: Parâmetro livre de Harris (típico: 0.04-0.06)
        threshold: Threshold para detecção (% do máximo, padrão: 0.01 = 1%)
    
    Returns:
        Tupla (número_de_corners, mapa_de_corners)
    """
    # Garantir que imagem está em uint8
    if image.dtype != np.uint8:
        image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    else:
        image_uint8 = image
    
    # Aplicar detector de Harris do OpenCV
    harris_response = cv2.cornerHarris(image_uint8, block_size, ksize, k)
    
    # Dilatar para encontrar máximos locais
    harris_response_dilated = cv2.dilate(harris_response, None)
    
    # Aplicar threshold
    # Corners são pixels onde resposta > threshold * máximo
    threshold_value = threshold * harris_response.max()
    
    # Criar máscara de corners
    corner_mask = (harris_response > threshold_value) & \
                  (harris_response == harris_response_dilated)
    
    # Contar número de corners
    num_corners = np.sum(corner_mask)
    
    return int(num_corners), corner_mask


def detect_harris_corners_adaptive(image: np.ndarray, 
                                   max_corners: int = 1000,
                                   quality_level: float = 0.01,
                                   min_distance: int = 10) -> int:
    """
    Detecta corners usando goodFeaturesToTrack (método adaptativo).
    
    Esta é uma alternativa mais robusta que seleciona os melhores corners
    automaticamente com supressão de não-máximos.
    
    Args:
        image: Imagem de entrada (grayscale)
        max_corners: Número máximo de corners a detectar
        quality_level: Qualidade mínima dos corners (0-1)
        min_distance: Distância mínima entre corners em pixels
    
    Returns:
        Número de corners detectados
    """
    # Garantir que imagem está em uint8
    if image.dtype != np.uint8:
        image_uint8 = np.clip(image, 0, 255).astype(np.uint8)
    else:
        image_uint8 = image
    
    # Detectar corners usando goodFeaturesToTrack
    # Este método usa Harris internamente mas com seleção adaptativa
    corners = cv2.goodFeaturesToTrack(
        image_uint8,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        useHarrisDetector=True,
        k=0.04
    )
    
    # Retornar número de corners
    if corners is not None:
        return len(corners)
    else:
        return 0


def visualize_corners(image: np.ndarray, corner_mask: np.ndarray) -> np.ndarray:
    """
    Cria visualização dos corners detectados sobre a imagem.
    
    Args:
        image: Imagem original (grayscale)
        corner_mask: Máscara booleana de corners
    
    Returns:
        Imagem RGB com corners marcados em vermelho
    """
    # Converter para RGB
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image.copy()
    
    # Marcar corners em vermelho
    image_rgb[corner_mask] = [255, 0, 0]
    
    return image_rgb
