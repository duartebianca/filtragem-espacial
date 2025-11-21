"""
Coeficiente de Correlação
Mede a correlação linear entre duas imagens.
"""
import numpy as np


def calculate_correlation(original: np.ndarray, filtered: np.ndarray) -> float:
    """
    Calcula o coeficiente de correlação (r) entre duas imagens.
    
    Fórmula:
        r = σ_GS,F / (σ_GS · σ_F)
        
        onde σ_GS,F = (1/(N-1)) Σ (I_GS(x,y) - μ_GS)(I_F(x,y) - μ_F)
    
    Também pode ser calculado como correlação de Pearson.
    
    Args:
        original: Imagem original (ground-truth)
        filtered: Imagem filtrada
    
    Returns:
        Coeficiente de correlação r (varia de -1 a 1, 1 = correlação perfeita)
    """
    if original.shape != filtered.shape:
        raise ValueError("Imagens devem ter o mesmo tamanho")
    
    # Achatar imagens para vetores
    original_flat = original.flatten().astype(np.float64)
    filtered_flat = filtered.flatten().astype(np.float64)
    
    # Calcular médias
    mu_original = np.mean(original_flat)
    mu_filtered = np.mean(filtered_flat)
    
    # Calcular desvios
    dev_original = original_flat - mu_original
    dev_filtered = filtered_flat - mu_filtered
    
    # Calcular covariância
    covariance = np.sum(dev_original * dev_filtered)
    
    # Calcular desvios padrão
    std_original = np.sqrt(np.sum(dev_original ** 2))
    std_filtered = np.sqrt(np.sum(dev_filtered ** 2))
    
    # Evitar divisão por zero
    if std_original < 1e-10 or std_filtered < 1e-10:
        return 0.0
    
    # Calcular correlação
    correlation = covariance / (std_original * std_filtered)
    
    # Garantir que está no intervalo [-1, 1] (por erros numéricos)
    correlation = np.clip(correlation, -1.0, 1.0)
    
    return correlation


def calculate_correlation_pearson(original: np.ndarray, filtered: np.ndarray) -> float:
    """
    Calcula correlação usando numpy.corrcoef (alternativa).
    
    Args:
        original: Imagem original
        filtered: Imagem filtrada
    
    Returns:
        Coeficiente de correlação
    """
    if original.shape != filtered.shape:
        raise ValueError("Imagens devem ter o mesmo tamanho")
    
    # Achatar imagens
    original_flat = original.flatten().astype(np.float64)
    filtered_flat = filtered.flatten().astype(np.float64)
    
    # Calcular matriz de correlação
    corr_matrix = np.corrcoef(original_flat, filtered_flat)
    
    # Retornar correlação (elemento [0,1] ou [1,0])
    return corr_matrix[0, 1]
