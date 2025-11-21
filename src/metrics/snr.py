"""
SNR - Signal-to-Noise Ratio
Relação sinal-ruído em decibéis.
"""
import numpy as np


def calculate_snr(original: np.ndarray, filtered: np.ndarray) -> float:
    """
    Calcula a relação sinal-ruído (SNR) entre duas imagens.
    
    Fórmulas:
        SNR = 10 log₁₀(P_sinal / P_ruído)
        
        P_sinal = Σ Σ [I_original(x,y)]²
        
        P_ruído = Σ Σ [I_original(x,y) - I_filtrada(x,y)]²
    
    Args:
        original: Imagem original (ground-truth)
        filtered: Imagem filtrada
    
    Returns:
        Valor de SNR em dB (maior = melhor)
    """
    if original.shape != filtered.shape:
        raise ValueError("Imagens devem ter o mesmo tamanho")
    
    # Converter para float
    original_float = original.astype(np.float64)
    filtered_float = filtered.astype(np.float64)
    
    # Calcular potência do sinal
    P_signal = np.sum(original_float ** 2)
    
    # Calcular potência do ruído (diferença entre original e filtrada)
    noise = original_float - filtered_float
    P_noise = np.sum(noise ** 2)
    
    # Evitar divisão por zero ou log de zero
    if P_noise < 1e-10:
        # Ruído extremamente baixo → SNR muito alto
        return 100.0  # Valor arbitrariamente alto
    
    if P_signal < 1e-10:
        # Sinal muito fraco
        return 0.0
    
    # Calcular SNR em dB
    snr_db = 10.0 * np.log10(P_signal / P_noise)
    
    return snr_db


def calculate_psnr(original: np.ndarray, filtered: np.ndarray, 
                   max_value: float = 255.0) -> float:
    """
    Calcula o Peak Signal-to-Noise Ratio (PSNR) - métrica alternativa.
    
    PSNR = 10 log₁₀(MAX² / MSE)
    
    Args:
        original: Imagem original
        filtered: Imagem filtrada
        max_value: Valor máximo possível do pixel (255 para 8-bit)
    
    Returns:
        Valor de PSNR em dB
    """
    if original.shape != filtered.shape:
        raise ValueError("Imagens devem ter o mesmo tamanho")
    
    # Calcular MSE (Mean Squared Error)
    mse = np.mean((original.astype(np.float64) - filtered.astype(np.float64)) ** 2)
    
    # Evitar divisão por zero
    if mse < 1e-10:
        return 100.0  # Imagens idênticas
    
    # Calcular PSNR
    psnr = 10.0 * np.log10((max_value ** 2) / mse)
    
    return psnr
