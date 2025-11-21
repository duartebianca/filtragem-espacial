# Sistema de Filtragem Espacial e Avaliação de Métricas

**ES235 - Processamento de Imagem**  
**Universidade Federal de Pernambuco**  

- Estudante: Bianca Duarte Santos (bds@cin.ufpe.br)
Obs: fiz com base no grupo 5.

## 📋 Descrição

Sistema completo para aplicação de filtros espaciais em imagens ruidosas e avaliação quantitativa de desempenho usando múltiplas métricas. Implementado como parte do Lab 03 de Filtragem Espacial.

## 🎯 Objetivos

- Implementar 5 filtros espaciais para remoção de ruído
- Calcular 5 métricas de avaliação de qualidade
- Comparar desempenho dos filtros em 10 imagens ruidosas
- Gerar análises estatísticas e visualizações comparativas

## 🔧 Filtros Implementados (Grupo 5)

| Filtro | Tamanho da Janela | Descrição |
|--------|-------------------|-----------|
| **Gaussian Blur (GB)** | 5×5 | Convolução Gaussiana para suavização |
| **Moving Average (MA)** | 7×7 | Média uniforme da vizinhança |
| **Median (Med)** | 9×9 | Mediana - preserva bordas |
| **Wiener (Wien)** | 11×11 | Filtro adaptativo baseado em estatísticas locais |
| **IBSF** | 3×3 | Interference-Based Speckle Filter (3 passos) |

### Detalhes dos Filtros

#### Gaussian Blur
- Aplica kernel Gaussiano 2D
- Sigma calculado automaticamente baseado no tamanho da janela
- Ótimo para ruído Gaussiano

#### Moving Average
- Kernel uniforme normalizado
- Simples e eficiente
- Pode borrar bordas

#### Median Filter
- Não-linear, preserva bordas
- Excelente para ruído impulsivo (salt & pepper)
- Computacionalmente mais custoso

#### Wiener Filter (Lee Filter)
- **Adaptativo**: ajusta comportamento baseado em estatísticas locais
- Fórmula: `g(x,y) = α·f(x,y) + (1-α)·f̄(x,y)`
- **⚠️ Validação crítica**: α ∈ [0, 1] garantido com `np.clip()`
- α → 0: região homogênea (suavização)
- α → 1: região de borda (preservação)

#### IBSF (Interference-Based Speckle Filter)
Algoritmo em 3 passos:
1. Mediana com janela grande (3×3)
2. Supressão de interferência destrutiva: `I_C = max(I, I_Med)`
3. Mediana com janela pequena (3×3)

## 📊 Métricas de Avaliação

| Métrica | Fórmula | Interpretação |
|---------|---------|---------------|
| **RMSE** | `√[Σ(I_F - I_GS)² / ΣI_GS²]` | Menor = melhor |
| **SSIM** | Similaridade estrutural | -1 a 1, 1 = perfeito |
| **r** | Correlação de Pearson | -1 a 1, 1 = correlação perfeita |
| **SNR** | `10·log₁₀(P_signal / P_noise)` | Maior = melhor (dB) |
| **Corners** | Harris corner detection | Preservação de estruturas |

### Detalhes das Métricas

#### RMSE (Root Mean Squared Error)
- Erro quadrático médio normalizado
- Mede diferença pixel a pixel
- Sensível a outliers

#### SSIM (Structural Similarity Index)
- Baseado em luminância, contraste e estrutura
- Janela Gaussiana 11×11, σ=1.5
- Mais próximo da percepção humana que RMSE
- Implementação: `scikit-image`

#### Correlação (r)
- Mede relação linear entre imagens
- Independente de escala/offset
- r = 1: relação linear perfeita

#### SNR (Signal-to-Noise Ratio)
- Relação entre potência do sinal e do ruído
- Expressado em decibéis (dB)
- Valores típicos: 10-30 dB para boa qualidade

#### Harris Corners
- Detecta pontos de interesse (corners)
- Indica preservação de estruturas
- Imagens bem filtradas preservam corners reais

## 📁 Estrutura do Projeto

```
filtragem-espacial/
├── data/
│   ├── input/                     # Imagens originais
│   │   ├── Referencia.tif        # Ground-truth
│   │   └── Noise_*.tif           # 10 imagens ruidosas
│   └── output/
│       ├── table/                # Tabelas CSV/Excel
│       ├── graphic/              # Gráficos comparativos
│       └── filtered_images/      # Imagens processadas
├── src/
│   ├── filters/                  # Implementação dos filtros
│   │   ├── gaussian_blur.py
│   │   ├── moving_average.py
│   │   ├── median_filter.py
│   │   ├── wiener_filter.py
│   │   └── ibsf_filter.py
│   ├── metrics/                  # Implementação das métricas
│   │   ├── rmse.py
│   │   ├── ssim.py
│   │   ├── correlation.py
│   │   ├── snr.py
│   │   └── harris_corners.py
│   └── utils/                    # Utilitários
│       ├── image_io.py
│       └── visualization.py
├── main.py                       # Pipeline principal
├── pyproject.toml               # Dependências
└── README.md                    # Este arquivo
```

## 🚀 Instalação

### Pré-requisitos
- Python 3.9 ou superior
- pip

### Configuração do Ambiente

```powershell
# Criar ambiente virtual
python -m venv .venv

# Ativar ambiente (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Instalar dependências
pip install numpy opencv-python scikit-image pandas matplotlib seaborn tifffile openpyxl
```

## 💻 Uso

### Execução Básica

```powershell
# Ativar ambiente virtual
.\.venv\Scripts\Activate.ps1

# Executar pipeline completo
python main.py
```

### O que o Pipeline Faz

1. **Carrega imagens** de `data/input/`
2. **Aplica cada filtro** nas 10 imagens ruidosas
3. **Calcula métricas** comparando com a referência
4. **Gera tabelas** (CSV + Excel) com resultados
5. **Cria visualizações** (boxplots, violinplots, heatmaps)
6. **Salva imagens filtradas** organizadas por filtro

### Resultados Gerados

#### Tabelas (`data/output/table/`)
- `tabela_gaussian_blur.csv` / `.xlsx`
- `tabela_moving_average.csv` / `.xlsx`
- `tabela_median.csv` / `.xlsx`
- `tabela_wiener.csv` / `.xlsx`
- `tabela_ibsf.csv` / `.xlsx`

Cada tabela contém:
- Métricas para cada imagem ruidosa
- Linha de **Média**
- Linha de **Desvio Padrão**

#### Gráficos (`data/output/graphic/`)
- `boxplot_*.png` - Boxplots por métrica
- `violinplot_*.png` - Violinplots por métrica
- `heatmap_metricas.png` - Comparação geral

#### Imagens Filtradas (`data/output/filtered_images/`)
```
filtered_images/
├── gaussian_blur/
│   ├── Noise_1_filtered.tif
│   └── ...
├── moving_average/
├── median/
├── wiener/
└── ibsf/
```

## 📈 Exemplo de Resultados (Grupo 5)

### Médias das Métricas

| Filtro | RMSE ↓ | SSIM ↑ | r ↑ | SNR (dB) ↑ | Corners |
|--------|--------|--------|-----|------------|---------|
| Gaussian Blur | 0.279 | 0.372 | 0.818 | 12.66 | 390 |
| Moving Average | 0.277 | 0.567 | 0.905 | 12.29 | 395 |
| Median | 0.246 | 0.571 | 0.911 | 14.03 | 279 |
| Wiener | 0.290 | 0.305 | 0.777 | 19.86 | 353 |
| IBSF | 0.244 | 0.556 | 0.890 | 14.18 | 272 |

**Observações**:
- ↑ = maior é melhor
- ↓ = menor é melhor
- Median e IBSF apresentam melhor RMSE
- Moving Average e Median têm melhor SSIM
- Wiener tem maior SNR mas variância alta

## 🔍 Análise Técnica

### Filtro de Wiener - Validação de α
```python
# Garantir α ∈ [0, 1] 
alpha = 1.0 - (noise_variance / local_var)
alpha = np.clip(alpha, 0.0, 1.0)  # Validação obrigatória
```

Sem esta validação, o filtro pode gerar **aberrações** quando:
- `local_var < noise_variance` → α negativo
- Divisão por zero em regiões uniformes

### IBSF - Física da Interferência
O filtro explora a natureza do speckle:
- **Interferência destrutiva** → pixels escuros (removidos)
- **Interferência construtiva** → pixels brilhantes (preservados)
- `max(I, I_Med)` remove apenas escuros, mantém claros

### SSIM vs RMSE
- **RMSE**: métrica pixel-wise, sensível a deslocamentos
- **SSIM**: métrica estrutural, mais robusta
- SSIM correlaciona melhor com percepção humana

## 🛠️ Dependências Principais

```
numpy>=1.24.0          # Operações matriciais
opencv-python>=4.8.0   # Processamento de imagens
scikit-image>=0.21.0   # SSIM e outras métricas
pandas>=2.0.0          # Manipulação de dados
matplotlib>=3.7.0      # Visualização
seaborn>=0.12.0        # Gráficos estatísticos
tifffile>=2023.0.0     # I/O de TIFF
openpyxl>=3.0.0        # Exportação para Excel
```