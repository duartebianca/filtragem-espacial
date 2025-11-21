"""
Utilitários para visualização e geração de gráficos.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional


def setup_plot_style():
    """Configura estilo padrão para os gráficos."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10


def plot_images_comparison(images: Dict[str, np.ndarray], 
                           titles: Optional[List[str]] = None,
                           save_path: Optional[str] = None,
                           figsize: tuple = (15, 10)) -> None:
    """
    Plota múltiplas imagens lado a lado para comparação.
    
    Args:
        images: Dicionário {nome: imagem}
        titles: Lista de títulos (opcional, usa keys do dict)
        save_path: Caminho para salvar a figura (opcional)
        figsize: Tamanho da figura
    """
    n_images = len(images)
    if n_images == 0:
        return
    
    # Calcular layout do grid
    cols = min(3, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Garantir que axes seja sempre uma lista
    if n_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if rows > 1 else axes
    
    # Plotar imagens
    for idx, (name, image) in enumerate(images.items()):
        ax = axes[idx] if n_images > 1 else axes[0]
        ax.imshow(image, cmap='gray', vmin=0, vmax=255)
        ax.set_title(titles[idx] if titles else name)
        ax.axis('off')
    
    # Remover eixos extras
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico salvo: {save_path}")
    
    plt.close()


def plot_boxplot_comparison(df: pd.DataFrame, 
                            metric_name: str,
                            save_path: Optional[str] = None,
                            figsize: tuple = (12, 6)) -> None:
    """
    Gera boxplot comparando filtros para uma métrica específica.
    
    Args:
        df: DataFrame com colunas [Filter, Metric_Name, Value]
        metric_name: Nome da métrica (para título)
        save_path: Caminho para salvar (opcional)
        figsize: Tamanho da figura
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Criar boxplot
    sns.boxplot(data=df, x='Filter', y='Value', ax=ax, palette='Set2')
    
    ax.set_title(f'Comparação de Filtros - {metric_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Filtro', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Boxplot salvo: {save_path}")
    
    plt.close()


def plot_violinplot_comparison(df: pd.DataFrame,
                               metric_name: str,
                               save_path: Optional[str] = None,
                               figsize: tuple = (12, 6)) -> None:
    """
    Gera violinplot comparando filtros para uma métrica específica.
    
    Args:
        df: DataFrame com colunas [Filter, Metric_Name, Value]
        metric_name: Nome da métrica (para título)
        save_path: Caminho para salvar (opcional)
        figsize: Tamanho da figura
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Criar violinplot
    sns.violinplot(data=df, x='Filter', y='Value', ax=ax, palette='Set3')
    
    ax.set_title(f'Comparação de Filtros - {metric_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Filtro', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Violinplot salvo: {save_path}")
    
    plt.close()


def plot_all_metrics_comparison(results_dict: Dict[str, pd.DataFrame],
                                save_dir: str,
                                plot_type: str = 'boxplot') -> None:
    """
    Gera gráficos comparativos para todas as métricas.
    
    Args:
        results_dict: Dicionário {métrica: DataFrame}
        save_dir: Diretório para salvar gráficos
        plot_type: Tipo de gráfico ('boxplot' ou 'violinplot')
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    for metric_name, df in results_dict.items():
        output_file = save_path / f"{metric_name}_{plot_type}.png"
        
        if plot_type == 'boxplot':
            plot_boxplot_comparison(df, metric_name, str(output_file))
        elif plot_type == 'violinplot':
            plot_violinplot_comparison(df, metric_name, str(output_file))
        else:
            print(f"Tipo de gráfico desconhecido: {plot_type}")


def plot_metric_heatmap(df: pd.DataFrame,
                       save_path: Optional[str] = None,
                       figsize: tuple = (10, 8)) -> None:
    """
    Gera heatmap de métricas por filtro.
    
    Args:
        df: DataFrame pivotado [Filtro x Métrica]
        save_path: Caminho para salvar (opcional)
        figsize: Tamanho da figura
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Criar heatmap
    sns.heatmap(df, annot=True, fmt='.4f', cmap='RdYlGn', ax=ax, 
                cbar_kws={'label': 'Valor da Métrica'})
    
    ax.set_title('Heatmap de Desempenho dos Filtros', fontsize=14, fontweight='bold')
    ax.set_xlabel('Métrica', fontsize=12)
    ax.set_ylabel('Filtro', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Heatmap salvo: {save_path}")
    
    plt.close()
