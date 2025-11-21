"""
Filtro de Wiener (Lee Filter)
Filtro adaptativo baseado em estatísticas locais.
"""
import numpy as np
import cv2
from typing import Optional


def wiener_filter(image: np.ndarray, window_size: int, 
                  noise_variance: Optional[float] = None) -> np.ndarray:
    """
    Aplica filtro de Wiener adaptativo (Lee Filter) na imagem.
    
    Fórmula:
        g(x,y) = α · f(x,y) + (1 - α) · f̄(x,y)
        α = 1 - (σ_w² / σ_H²)
        
    Onde:
        - α: coeficiente adaptativo [DEVE estar em [0, 1]]
        - σ_w²: variância na janela local
        - σ_H²: variância de referência (região homogênea)
        - f̄(x,y): média local
    
    Comportamento:
        - α → 0: região homogênea → suavização
        - α → 1: região de borda → preservação
    
    Args:
        image: Imagem de entrada (grayscale, normalizada [0, 1])
        window_size: Tamanho da janela (deve ser ímpar)
        noise_variance: Variância do ruído (σ_H²). Se None, estima automaticamente.
    
    Returns:
        Imagem filtrada
    """
    if window_size % 2 == 0:
        raise ValueError("window_size deve ser ímpar")
    
    # Converter para float64 para evitar overflow
    img_float = image.astype(np.float64)
    
    # Criar imagem de saída
    filtered = np.zeros_like(img_float)
    
    # Calcular padding
    pad = window_size // 2
    img_padded = cv2.copyMakeBorder(img_float, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    
    # Estimar variância de ruído se não fornecida
    # Usar mediana das variâncias locais como estimativa de região homogênea
    if noise_variance is None:
        variance_map = np.zeros_like(img_float)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                window = img_padded[i:i+window_size, j:j+window_size]
                variance_map[i, j] = np.var(window)
        noise_variance = np.median(variance_map)
        
        # Garantir que não seja zero
        if noise_variance < 1e-10:
            noise_variance = 1e-10
    
    # Aplicar filtro de Wiener pixel a pixel
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extrair janela local
            window = img_padded[i:i+window_size, j:j+window_size]
            
            # Calcular estatísticas locais
            local_mean = np.mean(window)
            local_var = np.var(window)
            
            # Calcular coeficiente α
            if local_var < 1e-10:
                # Região completamente homogênea → usar média
                alpha = 0.0
            else:
                alpha = 1.0 - (noise_variance / local_var)
            
            # ⚠️ CRÍTICO: Garantir que α ∈ [0, 1]
            alpha = np.clip(alpha, 0.0, 1.0)
            
            # Aplicar fórmula de Wiener
            filtered[i, j] = alpha * img_float[i, j] + (1.0 - alpha) * local_mean
    
    return filtered
