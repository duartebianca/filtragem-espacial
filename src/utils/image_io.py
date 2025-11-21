"""
Utilitários para carregamento e salvamento de imagens.
"""
import numpy as np
import cv2
import tifffile
from pathlib import Path
from typing import Optional, Tuple


def load_image(filepath: str, as_gray: bool = True) -> np.ndarray:
    """
    Carrega uma imagem do disco.
    
    Args:
        filepath: Caminho para o arquivo de imagem
        as_gray: Se True, converte para grayscale
    
    Returns:
        Array numpy com a imagem
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")
    
    # Tentar carregar como TIFF primeiro
    if filepath.suffix.lower() in ['.tif', '.tiff']:
        try:
            image = tifffile.imread(str(filepath))
            
            # Converter para grayscale se necessário
            if as_gray and len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            return image
        except Exception as e:
            print(f"Aviso: Erro ao carregar com tifffile, tentando OpenCV: {e}")
    
    # Usar OpenCV como fallback
    if as_gray:
        image = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(str(filepath))
    
    if image is None:
        raise ValueError(f"Não foi possível carregar a imagem: {filepath}")
    
    return image


def save_image(image: np.ndarray, filepath: str, create_dirs: bool = True) -> None:
    """
    Salva uma imagem no disco.
    
    Args:
        image: Array numpy com a imagem
        filepath: Caminho para salvar o arquivo
        create_dirs: Se True, cria diretórios se não existirem
    """
    filepath = Path(filepath)
    
    if create_dirs:
        filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Salvar usando formato apropriado
    if filepath.suffix.lower() in ['.tif', '.tiff']:
        tifffile.imwrite(str(filepath), image)
    else:
        cv2.imwrite(str(filepath), image)


def load_all_images(input_dir: str, pattern: str = "*.tif") -> dict:
    """
    Carrega todas as imagens de um diretório.
    
    Args:
        input_dir: Diretório com as imagens
        pattern: Padrão de arquivo (padrão: "*.tif")
    
    Returns:
        Dicionário {nome_arquivo: imagem}
    """
    input_path = Path(input_dir)
    images = {}
    
    for filepath in sorted(input_path.glob(pattern)):
        try:
            image = load_image(str(filepath))
            images[filepath.stem] = image
            print(f"✓ Carregada: {filepath.name} - Shape: {image.shape}, Dtype: {image.dtype}")
        except Exception as e:
            print(f"✗ Erro ao carregar {filepath.name}: {e}")
    
    return images


def normalize_image(image: np.ndarray, target_range: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    """
    Normaliza imagem para um intervalo específico.
    
    Args:
        image: Imagem de entrada
        target_range: Tupla (min, max) do intervalo alvo
    
    Returns:
        Imagem normalizada
    """
    img_float = image.astype(np.float64)
    
    # Normalizar para [0, 1]
    img_min = img_float.min()
    img_max = img_float.max()
    
    if img_max - img_min < 1e-10:
        # Imagem constante
        return np.full_like(img_float, target_range[0])
    
    img_norm = (img_float - img_min) / (img_max - img_min)
    
    # Escalar para target_range
    target_min, target_max = target_range
    img_scaled = img_norm * (target_max - target_min) + target_min
    
    return img_scaled


def denormalize_image(image: np.ndarray, original_dtype: np.dtype = np.uint8) -> np.ndarray:
    """
    Desnormaliza imagem de volta para o tipo original.
    
    Args:
        image: Imagem normalizada
        original_dtype: Tipo de dado original
    
    Returns:
        Imagem no tipo original
    """
    if original_dtype == np.uint8:
        # Escalar de [0, 1] ou qualquer range para [0, 255]
        img_denorm = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    elif original_dtype == np.uint16:
        img_denorm = np.clip(image * 65535.0, 0, 65535).astype(np.uint16)
    else:
        img_denorm = image.astype(original_dtype)
    
    return img_denorm


def get_image_stats(image: np.ndarray) -> dict:
    """
    Calcula estatísticas descritivas de uma imagem.
    
    Args:
        image: Imagem de entrada
    
    Returns:
        Dicionário com estatísticas
    """
    return {
        'shape': image.shape,
        'dtype': str(image.dtype),
        'min': float(image.min()),
        'max': float(image.max()),
        'mean': float(image.mean()),
        'std': float(image.std()),
        'median': float(np.median(image))
    }
