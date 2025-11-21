"""
RMSE - Root Mean Squared Error
Métrica de erro quadrático médio normalizado.
"""
import numpy as np


def calculate_rmse(original: np.ndarray, filtered: np.ndarray) -> float:
    """
    Calcula o Root Mean Squared Error entre duas imagens.
    
    Fórmula:
        RMSE = √[ Σ Σ (I_F(x,y) - I_GS(x,y))² / Σ Σ I_GS(x,y)² ]
    
    Args:
        original: Imagem original (ground-truth)
        filtered: Imagem filtrada
    
    Returns:
        Valor de RMSE (menor = melhor)
    """
    if original.shape != filtered.shape:
        raise ValueError("Imagens devem ter o mesmo tamanho")
    
    # Converter para float para evitar overflow
    original_float = original.astype(np.float64)
    filtered_float = filtered.astype(np.float64)
    
    # Calcular diferença quadrática
    squared_diff = (filtered_float - original_float) ** 2
    
    # Calcular potência do sinal (energia da imagem original)
    signal_power = np.sum(original_float ** 2)
    
    # Evitar divisão por zero
    if signal_power < 1e-10:
        return 0.0
    
    # Calcular RMSE normalizado
    rmse = np.sqrt(np.sum(squared_diff) / signal_power)
    
    return rmse
