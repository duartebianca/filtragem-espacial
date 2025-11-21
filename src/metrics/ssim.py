"""
SSIM - Structural Similarity Index
Métrica de similaridade estrutural entre imagens.

Referência: Wang et al. (2004) - "Image Quality Assessment: 
From Error Visibility to Structural Similarity"
"""
import numpy as np
from skimage.metrics import structural_similarity as sk_ssim


def calculate_ssim(original: np.ndarray, filtered: np.ndarray, 
                   data_range: int = 255) -> float:
    """
    Calcula o SSIM (Structural Similarity Index) entre duas imagens.
    
    Componentes do SSIM:
        - Luminância: l(x,y) = (2μ_GS μ_F + C₁) / (μ_GS² + μ_F² + C₁)
        - Contraste: c(x,y) = (2σ_GS σ_F + C₂) / (σ_GS² + σ_F² + C₂)
        - Estrutura: s(x,y) = (σ_GS,F + C₃) / (σ_GS σ_F + C₃)
    
    SSIM(x,y) = [(2μ_GS μ_F + C₁)(2σ_GS,F + C₂)] / 
                [(μ_GS² + μ_F² + C₁)(σ_GS² + σ_F² + C₂)]
    
    Constantes:
        - C₁ = (0.01 × d_r)²
        - C₂ = (0.03 × d_r)²
        - d_r = 255 (para imagens 8-bit)
    
    Args:
        original: Imagem original (ground-truth)
        filtered: Imagem filtrada
        data_range: Faixa de valores dos pixels (padrão: 255 para 8-bit)
    
    Returns:
        Valor de SSIM (varia de -1 a 1, 1 = similaridade perfeita)
    """
    if original.shape != filtered.shape:
        raise ValueError("Imagens devem ter o mesmo tamanho")
    
    # Usar implementação do scikit-image com janela Gaussiana
    # win_size=11, sigma=1.5 conforme especificação do artigo
    ssim_value = sk_ssim(
        original, 
        filtered,
        data_range=data_range,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False
    )
    
    return ssim_value


def calculate_ssim_manual(original: np.ndarray, filtered: np.ndarray,
                          window_size: int = 11, data_range: int = 255) -> float:
    """
    Implementação manual do SSIM para referência/debug.
    
    Args:
        original: Imagem original (ground-truth)
        filtered: Imagem filtrada
        window_size: Tamanho da janela (padrão: 11)
        data_range: Faixa de valores dos pixels
    
    Returns:
        Valor de SSIM
    """
    from scipy.ndimage import uniform_filter
    
    if original.shape != filtered.shape:
        raise ValueError("Imagens devem ter o mesmo tamanho")
    
    # Converter para float
    img1 = original.astype(np.float64)
    img2 = filtered.astype(np.float64)
    
    # Constantes
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    # Calcular médias locais
    mu1 = uniform_filter(img1, size=window_size)
    mu2 = uniform_filter(img2, size=window_size)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Calcular variâncias e covariância locais
    sigma1_sq = uniform_filter(img1 ** 2, size=window_size) - mu1_sq
    sigma2_sq = uniform_filter(img2 ** 2, size=window_size) - mu2_sq
    sigma12 = uniform_filter(img1 * img2, size=window_size) - mu1_mu2
    
    # Fórmula do SSIM
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim_map = numerator / denominator
    
    # Retornar média (MSSIM)
    return np.mean(ssim_map)
