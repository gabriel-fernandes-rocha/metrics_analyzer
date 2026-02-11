# 📊 Sistema de Análise de Métricas de Classificação

Sistema modularizado e profissional para análise de performance de modelos de classificação binária, com geração automática de curvas ROC, matrizes de confusão e métricas detalhadas.

## 🎯 Características Principais

✅ **Detecção Automática Inteligente**
- Detecta automaticamente formato de arquivo (CSV ou Parquet)
- **Suporta diretórios Parquet** com múltiplos arquivos `part-*`
- Identifica delimitador CSV (vírgula ou ponto-e-vírgula)
- Detecta colunas de score, label e município automaticamente
- Extração automática de UF a partir de códigos de município (2 primeiros dígitos)
- **Conversão automática de tipos**: score → float (2 decimais), label → int (0/1)

✅ **Análises Flexíveis**
- Análise por UF individual
- Análise geral (todos os dados)
- Configurável via arquivo YAML simples

✅ **Visualizações Profissionais**
- Curvas ROC com ponto ótimo (Youden)
- Matriz de confusão por threshold
- Gráficos customizáveis (títulos, labels, cores)

✅ **Métricas Completas**
- AUC (Area Under Curve)
- Acurácia, Precisão, Recall, F1-Score
- Especificidade
- Threshold ótimo
- Matriz de confusão (VP, FP, VN, FN)

✅ **Tratamento de Erros Robusto**
- Validação completa de configurações
- Mensagens de erro claras e descritivas
- Logs informativos em cada etapa

## 📁 Estrutura do Projeto

```
.
├── metrics_analyzer.py      # Módulo principal (código modularizado)
├── config.yaml              # Arquivo de configuração (VOCÊ EDITA ESTE)
├── config_example.yaml      # Exemplo de configuração
├── exemplo_uso.ipynb        # Notebook Jupyter com exemplos
└── README.md                # Esta documentação
```

## 🚀 Instalação

### Requisitos

```bash
pip install pandas numpy matplotlib scikit-learn pyyaml openpyxl
```

Para suporte a Parquet:
```bash
pip install pyarrow
```

## ⚙️ Configuração

### 1. Criar arquivo `config.yaml`

Copie o arquivo de exemplo e edite conforme suas necessidades:

```bash
cp config_example.yaml config.yaml
```

### 2. Configurar parâmetros essenciais

```yaml
# Configurações de Entrada
input:
  # Pode ser um arquivo único ou diretório (para Parquet particionado)
  data_path: "/caminho/para/seus/dados.csv"  # ou .parquet ou diretório/
  score_column: null      # Nome da coluna de score (null = auto-detecta)
  label_column: null      # Nome da coluna de label (null = auto-detecta)
  uf_column: null         # Nome da coluna de município (null = auto-detecta)

# Configurações de Análise
analysis:
  analyze_by_uf: true     # Analisar por UF?
  analyze_overall: true   # Analisar dados gerais?
  threshold_step: 0.05    # Intervalo de thresholds

# Configurações de Saída
output:
  output_dir: "/caminho/para/output"  # OBRIGATÓRIO
  save_files: true        # Salvar gráficos e CSVs?
  display_plots: false    # Exibir gráficos no notebook?
  save_metrics: true      # Salvar métricas em CSV?

# Configurações de Visualização
visualization:
  plot_title_roc: "Curva ROC – UF {uf}"
  plot_title_confusion: "Matriz de Confusão x Ponto de Corte – UF {uf}"
  xlabel_roc: "FPR"
  ylabel_roc: "TPR"
  # ... (veja config_example.yaml para todas as opções)
```

## 💻 Uso

### Opção 1: Linha de Comando

```bash
python metrics_analyzer.py config.yaml
```

### Opção 2: Dentro de um Script Python

```python
from metrics_analyzer import main

# Executar análise completa
df_metricas_uf, df_metricas_geral = main("config.yaml")

# Visualizar resultados
print(df_metricas_uf)
print(df_metricas_geral)
```

### Opção 3: Notebook Jupyter (Recomendado)

Abra o arquivo `exemplo_uso.ipynb` para exemplos interativos completos.

```python
from metrics_analyzer import MetricsAnalyzer

# Criar analisador
analyzer = MetricsAnalyzer("config.yaml")

# Carregar e preparar dados
analyzer.load_and_prepare_data()

# Executar análise
df_uf, df_geral = analyzer.run_analysis()
```

### Opção 4: Uso Rápido sem Salvar Arquivos

Se você quer apenas visualizar no notebook sem salvar:

```yaml
output:
  save_files: false
  display_plots: true
  save_metrics: false
```

## 📊 Detecção Automática de Colunas

O sistema detecta automaticamente as seguintes colunas:

### Coluna de Score
Procura por: `score`, `prob`, `confidence`, `similarity`

### Coluna de Label
Procura por: `match`, `label`, `class`, `target`, `true`

### Coluna de Município
Procura por: `municipio`, `município`, `city`, `cidade`, `ibge`

**Importante**: Se as colunas não forem detectadas automaticamente, especifique manualmente no `config.yaml`:

```yaml
input:
  score_column: "score_final"
  label_column: "MATCH_FINAL"
  uf_column: "consulta_municipio"
```

## 📈 Saídas Geradas

### 1. Arquivos CSV

- `metricas_geral.csv`: Métricas consolidadas de todos os dados
- `metricas_por_uf.csv`: Métricas para cada UF

**Colunas nos CSVs:**
- `identificador`: UF ou "GERAL"
- `threshold_otimo`: Threshold de decisão ótimo (Youden)
- `VP`, `FP`, `VN`, `FN`: Matriz de confusão
- `auc`: Area Under ROC Curve
- `acuracia`, `precisao`, `recall`, `f1_score`, `especificidade`
- `n_registros`: Quantidade de registros analisados

### 2. Gráficos (pasta `plots/`)

- `roc_GERAL.png`: Curva ROC com todos os dados
- `roc_UF_28.png`: Curva ROC para UF específica
- `matriz_confusao_threshold_GERAL.png`: Matriz por threshold (geral)
- `matriz_confusao_threshold_UF_28.png`: Matriz por threshold (por UF)

## 🎨 Personalização de Visualizações

### Títulos e Labels

```yaml
visualization:
  plot_title_roc: "Curva ROC – Estado {uf}"  # Use {uf} como placeholder
  xlabel_roc: "Taxa de Falsos Positivos"
  ylabel_roc: "Taxa de Verdadeiros Positivos"
```

### Legendas

```yaml
visualization:
  legend_vp: "Verdadeiro Positivo"
  legend_fp: "Falso Positivo"
  legend_vn: "Verdadeiro Negativo"
  legend_fn: "Falso Negativo"
```

### Tamanho e Resolução

```yaml
visualization:
  dpi: 150                    # Resolução das imagens
  figsize_roc: [8, 7]        # Largura x Altura (ROC)
  figsize_confusion: [12, 6] # Largura x Altura (Confusão)
```

## 🔍 Exemplos de Uso

### Exemplo 1: Análise Apenas Geral (Sem UF)

```yaml
analysis:
  analyze_by_uf: false
  analyze_overall: true
```

### Exemplo 2: Apenas Visualizar (Não Salvar)

```yaml
output:
  save_files: false
  display_plots: true
  save_metrics: false
```

### Exemplo 3: Análise Completa com Tudo

```yaml
analysis:
  analyze_by_uf: true
  analyze_overall: true

output:
  save_files: true
  display_plots: true
  save_metrics: true
```

### Exemplo 4: Arquivo Parquet com Delimitação Manual

```yaml
input:
  data_path: "/dados/dataset.parquet"
  score_column: "probabilidade"
  label_column: "match_verdadeiro"
  uf_column: "cod_municipio"
```

### Exemplo 5: Diretório Parquet Particionado (múltiplos part-*)

Se seus dados estão em um diretório com múltiplos arquivos Parquet:

```
/home/usuario/dados/otimizacao/
├── part-00000-abc123.snappy.parquet
├── part-00001-abc123.snappy.parquet
├── part-00002-abc123.snappy.parquet
└── ...
```

Configure assim:

```yaml
input:
  data_path: "/home/usuario/dados/otimizacao"  # Diretório, não arquivo
  score_column: null  # Auto-detecta
  label_column: null  # Auto-detecta
```

O sistema irá:
1. Detectar que é um diretório
2. Encontrar todos os arquivos `.parquet` ou `.pq`
3. Carregar e concatenar automaticamente
4. Processar como um único dataset

## 🛠️ Tratamento de Erros

O sistema fornece mensagens claras para diversos cenários:

### ❌ Arquivo não encontrado
```
❌ Arquivo não encontrado: /caminho/invalido/dados.csv
```

### ❌ Formato não suportado
```
❌ Formato não suportado: .xlsx. Use .csv ou .parquet
```

### ❌ Coluna não detectada
```
❌ Coluna de score não encontrada. Colunas disponíveis: ['col1', 'col2', ...]
```

### ⚠️ Dados insuficientes para UF
```
⚠️ UF_12: Sem dados válidos
⚠️ UF_15: Apenas uma classe presente
```

### ✅ Execução bem-sucedida
```
✅ Configuração carregada: config.yaml
✅ Dados carregados: 1,250,000 linhas x 15 colunas
✅ Coluna de score detectada: 'score_final'
✅ Coluna de label detectada: 'MATCH_FINAL'
✅ UFs extraídas: ['28', '29', '31', '33', '35']
📈 Analisando: UF_28 (85,234 registros)
  ✅ AUC: 0.9234 | F1: 0.8765 | Threshold: 0.6500
✅ ANÁLISE CONCLUÍDA COM SUCESSO
```

## 📚 Estrutura do Código

### Classes Principais

1. **ConfigLoader**: Carrega e valida configurações YAML
2. **DataLoader**: Carrega dados com detecção automática de formato
3. **ColumnDetector**: Detecta colunas relevantes automaticamente
4. **MetricsCalculator**: Calcula métricas de classificação
5. **PlotGenerator**: Gera gráficos profissionais
6. **MetricsAnalyzer**: Orquestra toda a análise

### Funções Principais

- `load_config()`: Carrega arquivo YAML
- `load_data()`: Carrega CSV ou Parquet
- `detect_score_column()`: Detecta coluna de score
- `calculate_metrics()`: Calcula métricas para threshold
- `plot_roc_curve()`: Gera curva ROC
- `run_analysis()`: Executa análise completa

## 🧪 Validações Automáticas

O sistema valida automaticamente:

- ✅ Existência de arquivo de dados
- ✅ Formato de arquivo suportado
- ✅ Estrutura do YAML
- ✅ Presença de colunas essenciais
- ✅ Dados suficientes por UF
- ✅ Classes balanceadas (pelo menos 2 classes)
- ✅ Diretórios de saída

## 🎓 Interpretação de Resultados

### Threshold Ótimo
Calculado usando o **Índice de Youden** (TPR - FPR), que maximiza a diferença entre verdadeiros positivos e falsos positivos.

### AUC (Area Under Curve)
- **0.9 - 1.0**: Excelente
- **0.8 - 0.9**: Muito Bom
- **0.7 - 0.8**: Bom
- **0.6 - 0.7**: Razoável
- **< 0.6**: Pobre

### F1-Score
Média harmônica entre Precisão e Recall. Ideal para dados desbalanceados.

---

**Desenvolvido com ❤️ para análise de dados de saúde pública**
