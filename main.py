"""
Pipeline principal de processamento de imagens.
Sistema completo de filtragem espacial e avaliação de métricas.

Grupo 5 - Tamanhos de janela:
- Gaussian Blur: 5x5
- Moving Average: 7x7
- Median: 9x9
- Wiener: 11x11
- IBSF: 3x3 (primeira mediana)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Importar filtros
from src.filters.gaussian_blur import gaussian_blur
from src.filters.moving_average import moving_average
from src.filters.median_filter import median_filter
from src.filters.wiener_filter import wiener_filter
from src.filters.ibsf_filter import ibsf_filter

# Importar métricas
from src.metrics.rmse import calculate_rmse
from src.metrics.ssim import calculate_ssim
from src.metrics.correlation import calculate_correlation
from src.metrics.snr import calculate_snr
from src.metrics.harris_corners import detect_harris_corners_adaptive

# Importar utilitários
from src.utils.image_io import load_all_images, save_image
from src.utils.visualization import (
    plot_images_comparison, 
    plot_all_metrics_comparison,
    plot_metric_heatmap
)


# ===== CONFIGURAÇÃO DO GRUPO 5 =====
GROUP_CONFIG = {
    'group_number': 5,
    'window_sizes': {
        'gaussian_blur': 5,
        'moving_average': 7,
        'median': 9,
        'wiener': 11,
        'ibsf': 3  # Primeira mediana
    }
}


class ImageFilteringPipeline:
    """Pipeline completo de filtragem e avaliação."""
    
    def __init__(self, input_dir: str, output_dir: str, config: dict):
        """
        Inicializa o pipeline.
        
        Args:
            input_dir: Diretório com imagens de entrada
            output_dir: Diretório para saída
            config: Configuração do grupo (tamanhos de janela)
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.config = config
        
        # Criar diretórios de saída
        self.table_dir = self.output_dir / 'table'
        self.graphic_dir = self.output_dir / 'graphic'
        self.filtered_dir = self.output_dir / 'filtered_images'
        
        for dir_path in [self.table_dir, self.graphic_dir, self.filtered_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Armazenar resultados
        self.results = {}
        self.images = {}
        self.reference_image = None
    
    def load_images(self) -> None:
        """Carrega todas as imagens do diretório de entrada."""
        print("\n" + "="*70)
        print("CARREGANDO IMAGENS")
        print("="*70)
        
        self.images = load_all_images(str(self.input_dir))
        
        # Separar imagem de referência
        if 'Referencia' in self.images:
            self.reference_image = self.images.pop('Referencia')
            print(f"\n✓ Imagem de referência carregada: Shape {self.reference_image.shape}")
        else:
            raise ValueError("Imagem de referência 'Referencia.tif' não encontrada!")
        
        # Listar imagens ruidosas
        noise_images = [k for k in self.images.keys() if k.startswith('Noise_')]
        print(f"\n✓ {len(noise_images)} imagens ruidosas carregadas")
    
    def apply_filter(self, image: np.ndarray, filter_name: str) -> np.ndarray:
        """
        Aplica um filtro específico na imagem.
        
        Args:
            image: Imagem de entrada
            filter_name: Nome do filtro
        
        Returns:
            Imagem filtrada
        """
        window_size = self.config['window_sizes'][filter_name]
        
        if filter_name == 'gaussian_blur':
            return gaussian_blur(image, window_size)
        
        elif filter_name == 'moving_average':
            return moving_average(image, window_size)
        
        elif filter_name == 'median':
            return median_filter(image, window_size)
        
        elif filter_name == 'wiener':
            # Normalizar para [0, 1] para Wiener
            img_norm = image.astype(np.float64) / 255.0
            filtered_norm = wiener_filter(img_norm, window_size)
            # Desnormalizar de volta para [0, 255]
            return np.clip(filtered_norm * 255.0, 0, 255).astype(np.uint8)
        
        elif filter_name == 'ibsf':
            # Para IBSF, window_size é para primeira mediana, segunda é sempre 3x3
            return ibsf_filter(image, window_size, window_size_small=3)
        
        else:
            raise ValueError(f"Filtro desconhecido: {filter_name}")
    
    def calculate_metrics(self, original: np.ndarray, filtered: np.ndarray) -> Dict[str, float]:
        """
        Calcula todas as métricas para um par de imagens.
        
        Args:
            original: Imagem original (referência)
            filtered: Imagem filtrada
        
        Returns:
            Dicionário com todas as métricas
        """
        metrics = {}
        
        try:
            metrics['RMSE'] = calculate_rmse(original, filtered)
        except Exception as e:
            print(f"  ⚠ Erro ao calcular RMSE: {e}")
            metrics['RMSE'] = np.nan
        
        try:
            metrics['SSIM'] = calculate_ssim(original, filtered, data_range=255)
        except Exception as e:
            print(f"  ⚠ Erro ao calcular SSIM: {e}")
            metrics['SSIM'] = np.nan
        
        try:
            metrics['r'] = calculate_correlation(original, filtered)
        except Exception as e:
            print(f"  ⚠ Erro ao calcular correlação: {e}")
            metrics['r'] = np.nan
        
        try:
            metrics['SNR'] = calculate_snr(original, filtered)
        except Exception as e:
            print(f"  ⚠ Erro ao calcular SNR: {e}")
            metrics['SNR'] = np.nan
        
        try:
            num_corners = detect_harris_corners_adaptive(filtered)
            metrics['Corners'] = num_corners
        except Exception as e:
            print(f"  ⚠ Erro ao detectar corners: {e}")
            metrics['Corners'] = 0
        
        return metrics
    
    def process_all_filters(self) -> None:
        """Processa todas as combinações de filtro × imagem ruidosa."""
        print("\n" + "="*70)
        print("PROCESSANDO FILTROS")
        print("="*70)
        
        filter_names = list(self.config['window_sizes'].keys())
        
        for filter_name in filter_names:
            print(f"\n--- Filtro: {filter_name.upper()} ---")
            print(f"    Tamanho da janela: {self.config['window_sizes'][filter_name]}x{self.config['window_sizes'][filter_name]}")
            
            filter_results = []
            
            for img_name, noisy_image in sorted(self.images.items()):
                print(f"  Processando: {img_name}...", end=' ')
                
                try:
                    # Aplicar filtro
                    filtered_image = self.apply_filter(noisy_image, filter_name)
                    
                    # Salvar imagem filtrada
                    save_path = self.filtered_dir / filter_name / f"{img_name}_filtered.tif"
                    save_image(filtered_image, str(save_path))
                    
                    # Calcular métricas
                    metrics = self.calculate_metrics(self.reference_image, filtered_image)
                    metrics['Image'] = img_name
                    filter_results.append(metrics)
                    
                    print("✓")
                
                except Exception as e:
                    print(f"✗ Erro: {e}")
                    continue
            
            # Armazenar resultados do filtro
            self.results[filter_name] = pd.DataFrame(filter_results)
    
    def generate_tables(self) -> None:
        """Gera tabelas de resultados para cada filtro."""
        print("\n" + "="*70)
        print("GERANDO TABELAS DE RESULTADOS")
        print("="*70)
        
        for filter_name, df in self.results.items():
            # Reorganizar colunas
            cols = ['Image', 'RMSE', 'SSIM', 'r', 'SNR', 'Corners']
            df = df[cols]
            
            # Calcular estatísticas
            stats_df = df.copy()
            
            # Adicionar linha de média
            mean_row = df.select_dtypes(include=[np.number]).mean()
            mean_row['Image'] = 'Média'
            mean_df = pd.DataFrame([mean_row])
            
            # Adicionar linha de desvio padrão
            std_row = df.select_dtypes(include=[np.number]).std()
            std_row['Image'] = 'Desvio Padrão'
            std_df = pd.DataFrame([std_row])
            
            # Concatenar
            final_df = pd.concat([stats_df, mean_df, std_df], ignore_index=True)
            
            # Salvar como CSV
            csv_path = self.table_dir / f"tabela_{filter_name}.csv"
            final_df.to_csv(csv_path, index=False, float_format='%.6f')
            print(f"✓ Tabela salva: {csv_path.name}")
            
            # Salvar como Excel (opcional)
            try:
                excel_path = self.table_dir / f"tabela_{filter_name}.xlsx"
                final_df.to_excel(excel_path, index=False, float_format='%.6f')
            except ImportError:
                pass  # openpyxl não instalado, apenas CSV é suficiente
            
            # Mostrar resumo
            print(f"\n  {filter_name.upper()}:")
            print(f"    RMSE médio: {mean_row['RMSE']:.6f} ± {std_row['RMSE']:.6f}")
            print(f"    SSIM médio: {mean_row['SSIM']:.6f} ± {std_row['SSIM']:.6f}")
            print(f"    Correlação média: {mean_row['r']:.6f} ± {std_row['r']:.6f}")
            print(f"    SNR médio: {mean_row['SNR']:.2f} ± {std_row['SNR']:.2f} dB")
            print(f"    Corners médio: {mean_row['Corners']:.0f} ± {std_row['Corners']:.0f}")
    
    def generate_visualizations(self) -> None:
        """Gera visualizações comparativas."""
        print("\n" + "="*70)
        print("GERANDO VISUALIZAÇÕES")
        print("="*70)
        
        # Preparar dados para boxplots
        metrics_data = {
            'RMSE': [],
            'SSIM': [],
            'Correlação': [],
            'SNR': []
        }
        
        for filter_name, df in self.results.items():
            for metric_key, metric_col in [('RMSE', 'RMSE'), 
                                           ('SSIM', 'SSIM'), 
                                           ('Correlação', 'r'),
                                           ('SNR', 'SNR')]:
                for value in df[metric_col]:
                    if not np.isnan(value):
                        metrics_data[metric_key].append({
                            'Filter': filter_name,
                            'Value': value
                        })
        
        # Gerar boxplots
        for metric_name, data_list in metrics_data.items():
            if len(data_list) > 0:
                df_metric = pd.DataFrame(data_list)
                
                # Boxplot
                save_path = self.graphic_dir / f"boxplot_{metric_name.lower()}.png"
                from src.utils.visualization import plot_boxplot_comparison
                plot_boxplot_comparison(df_metric, metric_name, str(save_path))
                
                # Violinplot
                save_path = self.graphic_dir / f"violinplot_{metric_name.lower()}.png"
                from src.utils.visualization import plot_violinplot_comparison
                plot_violinplot_comparison(df_metric, metric_name, str(save_path))
        
        # Gerar heatmap comparativo
        print("\n✓ Gerando heatmap comparativo...")
        heatmap_data = []
        for filter_name, df in self.results.items():
            row = {
                'Filtro': filter_name,
                'RMSE': df['RMSE'].mean(),
                'SSIM': df['SSIM'].mean(),
                'Correlação': df['r'].mean(),
                'SNR': df['SNR'].mean()
            }
            heatmap_data.append(row)
        
        heatmap_df = pd.DataFrame(heatmap_data).set_index('Filtro')
        save_path = self.graphic_dir / "heatmap_metricas.png"
        plot_metric_heatmap(heatmap_df, str(save_path))
    
    def run(self) -> None:
        """Executa o pipeline completo."""
        print("\n" + "="*70)
        print("PIPELINE DE FILTRAGEM ESPACIAL - GRUPO 5")
        print("="*70)
        print(f"Diretório de entrada: {self.input_dir}")
        print(f"Diretório de saída: {self.output_dir}")
        print(f"\nConfigurações de janela:")
        for filter_name, size in self.config['window_sizes'].items():
            print(f"  • {filter_name}: {size}x{size}")
        
        # Executar etapas
        self.load_images()
        self.process_all_filters()
        self.generate_tables()
        self.generate_visualizations()
        
        print("\n" + "="*70)
        print("PIPELINE CONCLUÍDO COM SUCESSO!")
        print("="*70)
        print(f"\nResultados salvos em: {self.output_dir}")
        print(f"  • Tabelas: {self.table_dir}")
        print(f"  • Gráficos: {self.graphic_dir}")
        print(f"  • Imagens filtradas: {self.filtered_dir}")


def main():
    """Função principal."""
    # Configurar caminhos
    input_dir = "data/input"
    output_dir = "data/output"
    
    # Criar e executar pipeline
    pipeline = ImageFilteringPipeline(input_dir, output_dir, GROUP_CONFIG)
    pipeline.run()


if __name__ == "__main__":
    main()
