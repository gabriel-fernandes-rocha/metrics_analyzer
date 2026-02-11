import os
import sys
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, 
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


class ConfigLoader:
    """Carregador e validador de configurações YAML."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Carrega arquivo de configuração YAML.
        
        Args:
            config_path: Caminho para arquivo YAML
            
        Returns:
            Dicionário com configurações
            
        Raises:
            FileNotFoundError: Se arquivo não existe
            yaml.YAMLError: Se YAML é inválido
        """
        try:
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"❌ Arquivo de configuração não encontrado: {config_path}")
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            print(f"✅ Configuração carregada: {config_path}")
            return config
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"❌ Erro ao parsear YAML: {e}")
        except Exception as e:
            raise Exception(f"❌ Erro ao carregar configuração: {e}")
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> None:
        """
        Valida estrutura de configuração.
        
        Args:
            config: Dicionário de configuração
            
        Raises:
            ValueError: Se configuração é inválida
        """
        required_sections = ['input', 'analysis', 'output', 'visualization']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"❌ Seção obrigatória ausente: {section}")
        
        # Valida campos essenciais
        if 'data_path' not in config['input']:
            raise ValueError("❌ Campo 'data_path' ausente em 'input'")
        
        if 'output_dir' not in config['output']:
            raise ValueError("❌ Campo 'output_dir' ausente em 'output'")
        
        print("✅ Configuração validada com sucesso")


class DataLoader:
    """Carregador inteligente de dados com detecção automática de formato."""
    
    @staticmethod
    def detect_file_type(file_path: str) -> str:
        """
        Detecta tipo de arquivo pela extensão.
        
        Args:
            file_path: Caminho do arquivo
            
        Returns:
            Tipo do arquivo ('csv' ou 'parquet')
            
        Raises:
            ValueError: Se formato não suportado
        """
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.csv':
            return 'csv'
        elif ext in ['.parquet', '.pq']:
            return 'parquet'
        else:
            raise ValueError(f"❌ Formato não suportado: {ext}. Use .csv ou .parquet")
    
    @staticmethod
    def detect_csv_delimiter(file_path: str, n_lines: int = 5) -> str:
        """
        Detecta delimitador de arquivo CSV.
        
        Args:
            file_path: Caminho do arquivo CSV
            n_lines: Número de linhas para análise
            
        Returns:
            Delimitador detectado (',' ou ';')
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                sample = ''.join([f.readline() for _ in range(n_lines)])
            
            comma_count = sample.count(',')
            semicolon_count = sample.count(';')
            
            delimiter = ',' if comma_count > semicolon_count else ';'
            print(f"✅ Delimitador detectado: '{delimiter}'")
            return delimiter
            
        except Exception as e:
            print(f"⚠️  Erro ao detectar delimitador, usando ',': {e}")
            return ','
    
    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        """
        Carrega dados com detecção automática de formato.
        Suporta diretórios Parquet com múltiplos arquivos part-*.
        
        Args:
            file_path: Caminho do arquivo ou diretório
            
        Returns:
            DataFrame com dados carregados
            
        Raises:
            FileNotFoundError: Se arquivo/diretório não existe
            Exception: Se erro ao carregar dados
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"❌ Arquivo/diretório não encontrado: {file_path}")
            
            # Verificar se é diretório (comum em Parquet particionado)
            if os.path.isdir(file_path):
                print(f"📁 Detectado diretório Parquet: {os.path.basename(file_path)}")
                
                # Buscar arquivos parquet no diretório
                parquet_files = []
                for root, dirs, files in os.walk(file_path):
                    for file in files:
                        if file.endswith('.parquet') or file.endswith('.pq'):
                            parquet_files.append(os.path.join(root, file))
                
                if not parquet_files:
                    raise FileNotFoundError(f"❌ Nenhum arquivo Parquet encontrado em: {file_path}")
                
                print(f"📊 Encontrados {len(parquet_files)} arquivo(s) Parquet")
                
                # Carregar todos os arquivos e concatenar
                dfs = []
                for pq_file in sorted(parquet_files):
                    print(f"  • Lendo: {os.path.basename(pq_file)}")
                    dfs.append(pd.read_parquet(pq_file))
                
                df = pd.concat(dfs, ignore_index=True)
                print(f"✅ Dados carregados: {len(df):,} linhas x {len(df.columns)} colunas")
                return df
            
            # Se for arquivo único
            file_type = DataLoader.detect_file_type(file_path)
            print(f"📁 Carregando arquivo {file_type.upper()}: {os.path.basename(file_path)}")
            
            if file_type == 'csv':
                delimiter = DataLoader.detect_csv_delimiter(file_path)
                df = pd.read_csv(file_path, delimiter=delimiter)
            else:
                df = pd.read_parquet(file_path)
            
            print(f"✅ Dados carregados: {len(df):,} linhas x {len(df.columns)} colunas")
            return df
            
        except Exception as e:
            raise Exception(f"❌ Erro ao carregar dados: {e}")


class DataProcessor:
    """Processador de dados para conversão e limpeza."""
    
    @staticmethod
    def convert_and_validate_columns(
        df: pd.DataFrame,
        score_col: str,
        label_col: str,
        uf_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Converte colunas para os tipos corretos e valida dados.
        
        Args:
            df: DataFrame original
            score_col: Nome da coluna de score
            label_col: Nome da coluna de label
            uf_col: Nome da coluna de UF (opcional)
            
        Returns:
            DataFrame com colunas convertidas
            
        Raises:
            ValueError: Se conversão falhar
        """
        try:
            df_clean = df.copy()
            
            print("\n🔧 Convertendo e validando colunas...")

            # Converter coluna de score para float com 2 casas decimais
            try:
                df_clean[score_col] = pd.to_numeric(df_clean[score_col], errors='coerce')
                
                n_invalidos = df_clean[score_col].isna().sum()
                if n_invalidos > 0:
                    print(f"  ⚠️  {n_invalidos} valores inválidos em '{score_col}' convertidos para NaN")
                
                print(f"  ✅ Coluna '{score_col}' → float (sem alteração de precisão)")
                print(f"     Range: [{df_clean[score_col].min():.6f}, {df_clean[score_col].max():.6f}]")
            
            except Exception as e:
                raise ValueError(f"❌ Erro ao converter coluna de score '{score_col}': {e}")

            # Converter coluna de label para int (0 ou 1)
            try:
                df_clean[label_col] = pd.to_numeric(df_clean[label_col], errors='coerce')
                df_clean[label_col] = df_clean[label_col].astype('Int64')  # Int64 permite NaN
                
                # Verificar se é binário
                valores_unicos = df_clean[label_col].dropna().unique()
                if not set(valores_unicos).issubset({0, 1}):
                    print(f"  ⚠️  Coluna '{label_col}' contém valores além de 0 e 1: {valores_unicos}")
                    print(f"     Convertendo para binário (0=negativo, 1=positivo)...")
                    df_clean[label_col] = (df_clean[label_col] > 0).astype(int)
                
                n_invalidos = df_clean[label_col].isna().sum()
                if n_invalidos > 0:
                    print(f"  ⚠️  {n_invalidos} valores inválidos em '{label_col}' convertidos para NaN")
                
                print(f"  ✅ Coluna '{label_col}' → int")
                print(f"     Distribuição: 0={(df_clean[label_col]==0).sum()}, 1={(df_clean[label_col]==1).sum()}")
                
            except Exception as e:
                raise ValueError(f"❌ Erro ao converter coluna de label '{label_col}': {e}")
            
            # Converter coluna de UF para string (se existir)
            if uf_col and uf_col in df_clean.columns:
                try:
                    df_clean[uf_col] = df_clean[uf_col].astype(str)
                    print(f"  ✅ Coluna '{uf_col}' → string")
                except Exception as e:
                    print(f"  ⚠️  Erro ao converter coluna de UF '{uf_col}': {e}")
            
            # Remover linhas com valores inválidos nas colunas essenciais
            antes = len(df_clean)
            df_clean = df_clean.dropna(subset=[score_col, label_col])
            depois = len(df_clean)
            
            if antes > depois:
                print(f"\n  🧹 {antes - depois} linhas removidas por valores inválidos")
                print(f"     Restam: {depois:,} linhas válidas")
            
            return df_clean
            
        except Exception as e:
            raise ValueError(f"❌ Erro ao processar colunas: {e}")


class ColumnDetector:
    """Detector inteligente de colunas relevantes."""
    
    @staticmethod
    def detect_score_column(df: pd.DataFrame, config_column: Optional[str] = None) -> str:
        """
        Detecta coluna de score.
        
        Args:
            df: DataFrame
            config_column: Nome da coluna na config (prioritário)
            
        Returns:
            Nome da coluna de score
            
        Raises:
            ValueError: Se coluna não encontrada
        """
        if config_column and config_column in df.columns:
            print(f"✅ Coluna de score (config): '{config_column}'")
            return config_column
        
        # Busca por padrões comuns
        patterns = ['score', 'prob', 'confidence', 'similarity']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in patterns):
                print(f"✅ Coluna de score detectada: '{col}'")
                return col
        
        raise ValueError(f"❌ Coluna de score não encontrada. Colunas disponíveis: {list(df.columns)}")
    
    @staticmethod
    def detect_label_column(df: pd.DataFrame, config_column: Optional[str] = None) -> str:
        """
        Detecta coluna de label/classificação.
        
        Args:
            df: DataFrame
            config_column: Nome da coluna na config (prioritário)
            
        Returns:
            Nome da coluna de label
            
        Raises:
            ValueError: Se coluna não encontrada
        """
        if config_column and config_column in df.columns:
            print(f"✅ Coluna de label (config): '{config_column}'")
            return config_column
        
        # Busca por padrões comuns
        patterns = ['match', 'label', 'class', 'target', 'true']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in patterns):
                # Verifica se é binária ou categórica com VP/FP
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) <= 10:  # Limite razoável para categorias
                    print(f"✅ Coluna de label detectada: '{col}' (valores: {unique_vals})")
                    return col
        
        raise ValueError(f"❌ Coluna de label não encontrada. Colunas disponíveis: {list(df.columns)}")
    
    @staticmethod
    def detect_uf_column(df: pd.DataFrame, config_column: Optional[str] = None) -> Optional[str]:
        """
        Detecta coluna de município para extração de UF.
        
        Args:
            df: DataFrame
            config_column: Nome da coluna na config (prioritário)
            
        Returns:
            Nome da coluna de município ou None
        """
        if config_column and config_column in df.columns:
            print(f"✅ Coluna de município (config): '{config_column}'")
            return config_column
        
        # Busca por padrões comuns
        patterns = ['municipio', 'município', 'city', 'cidade', 'ibge']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(pattern in col_lower for pattern in patterns):
                print(f"✅ Coluna de município detectada: '{col}'")
                return col
        
        print("⚠️  Coluna de município não detectada - análise por UF desabilitada")
        return None


class MetricsCalculator:
    """Calculador de métricas de classificação."""
    
    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray, 
        y_score: np.ndarray, 
        threshold: float
    ) -> Tuple[float, float, float, float, float, int, int, int, int]:
        """
        Calcula métricas de classificação para um threshold.
        
        Args:
            y_true: Labels verdadeiros
            y_score: Scores preditos
            threshold: Threshold de decisão
            
        Returns:
            Tupla com (acurácia, precisão, recall, f1, especificidade, VP, FP, FN, VN)
        """
        try:
            y_true = y_true.astype(int)
            y_pred = (y_score >= threshold).astype(int)
            
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            especificidade = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            return acc, prec, rec, f1, especificidade, tp, fp, fn, tn
            
        except Exception as e:
            print(f"⚠️  Erro ao calcular métricas: {e}")
            return 0, 0, 0, 0, 0, 0, 0, 0, 0
    
    @staticmethod
    def find_optimal_threshold(
        y_true: np.ndarray, 
        y_score: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """
        Encontra threshold ótimo usando índice de Youden.
        
        Args:
            y_true: Labels verdadeiros
            y_score: Scores preditos
            
        Returns:
            Tupla com (threshold_ótimo, AUC, FPR_ótimo, TPR_ótimo)
        """
        try:
            fpr, tpr, thresholds = roc_curve(y_true.astype(int), y_score)
            auc_score = roc_auc_score(y_true.astype(int), y_score)
            
            youden_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[youden_idx]
            optimal_fpr = fpr[youden_idx]
            optimal_tpr = tpr[youden_idx]
            
            return optimal_threshold, auc_score, optimal_fpr, optimal_tpr
            
        except Exception as e:
            print(f"⚠️  Erro ao calcular threshold ótimo: {e}")
            return 0.5, 0.5, 0, 0


class PlotGenerator:
    """Gerador de gráficos de análise."""
    
    def __init__(self, config: Dict[str, Any], output_dir: str):
        """
        Inicializa gerador de gráficos.
        
        Args:
            config: Configuração de visualização
            output_dir: Diretório para salvar gráficos
        """
        self.config = config
        self.output_dir = output_dir
        self.plots_dir = os.path.join(output_dir, "plots")
        
        if config['output']['save_files']:
            os.makedirs(self.plots_dir, exist_ok=True)
    
    def plot_confusion_matrix_by_threshold(
        self, 
        df: pd.DataFrame, 
        label_col: str, 
        score_col: str, 
        identifier: str
    ) -> None:
        """
        Plota matriz de confusão por threshold.
        
        Args:
            df: DataFrame com dados
            label_col: Nome da coluna de label
            score_col: Nome da coluna de score
            identifier: Identificador (UF ou GERAL)
        """
        try:
            df_clean = df[[label_col, score_col]].dropna()
            df_clean[label_col] = df_clean[label_col].astype(int)
            
            # Thresholds baseados no range real dos scores
            min_score = df_clean[score_col].min()
            max_score = df_clean[score_col].max()
            
            start = np.floor(min_score / self.config['analysis']['threshold_step']) * self.config['analysis']['threshold_step']
            start = round(start, 2)
            
            thresholds = np.round(
                np.arange(start, max_score + 0.001, self.config['analysis']['threshold_step']), 
                2
            )
            
            resultados = []
            for t in thresholds:
                pred = (df_clean[score_col] >= t).astype(int)
                VP = ((df_clean[label_col] == 1) & (pred == 1)).sum()
                FP = ((df_clean[label_col] == 0) & (pred == 1)).sum()
                VN = ((df_clean[label_col] == 0) & (pred == 0)).sum()
                FN = ((df_clean[label_col] == 1) & (pred == 0)).sum()
                
                resultados.append({
                    "Threshold": t,
                    "VP": VP,
                    "FP": FP,
                    "VN": VN,
                    "FN": FN
                })
            
            df_res = pd.DataFrame(resultados)
            
            # Criar gráfico
            fig, ax = plt.subplots(figsize=self.config['visualization']['figsize_confusion'])
            
            ax.plot(df_res["Threshold"], df_res["VP"], "o--", label=self.config['visualization']['legend_vp'], color="#2E8B57")
            ax.plot(df_res["Threshold"], df_res["FP"], "o--", label=self.config['visualization']['legend_fp'], color="#C0392B")
            ax.plot(df_res["Threshold"], df_res["VN"], "o--", label=self.config['visualization']['legend_vn'], color="#2C7FB8")
            ax.plot(df_res["Threshold"], df_res["FN"], "o--", label=self.config['visualization']['legend_fn'], color="#E67E22")
            
            title = self.config['visualization']['plot_title_confusion'].format(uf=identifier)
            ax.set_title(title)
            ax.set_xlabel(self.config['visualization']['xlabel_confusion'])
            ax.set_ylabel(self.config['visualization']['ylabel_confusion'])
            
            # Ajustar eixo X
            xmin = df_res["Threshold"].min()
            xmax = df_res["Threshold"].max()
            range_x = xmax - xmin
            margem = range_x * 0.03
            
            ax.set_xlim(xmin - margem, xmax + margem)
            ax.set_xticks(df_res["Threshold"])
            ax.set_xticklabels([f"{t:.2f}" for t in df_res["Threshold"]], rotation=45)
            
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(title="Classe", loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=10)
            
            fig.subplots_adjust(right=0.82)
            
            if self.config['output']['save_files']:
                path = os.path.join(self.plots_dir, f"matriz_confusao_threshold_{identifier}.png")
                plt.savefig(path, dpi=self.config['visualization']['dpi'])
                print(f"  💾 Gráfico salvo: {os.path.basename(path)}")
            
            if self.config['output']['display_plots']:
                plt.show()
            else:
                plt.close()
                
        except Exception as e:
            print(f"❌ Erro ao plotar matriz de confusão para {identifier}: {e}")
            plt.close()
    
    def plot_roc_curve(
        self, 
        df: pd.DataFrame, 
        label_col: str, 
        score_col: str, 
        identifier: str
    ) -> Tuple[float, float]:
        """
        Plota curva ROC.
        
        Args:
            df: DataFrame com dados
            label_col: Nome da coluna de label
            score_col: Nome da coluna de score
            identifier: Identificador (UF ou GERAL)
            
        Returns:
            Tupla com (threshold_ótimo, AUC)
        """
        try:
            df_clean = df[[label_col, score_col]].dropna()
            y_true = df_clean[label_col].astype(int).values
            y_score = df_clean[score_col].values
            
            fpr, tpr, thresholds = roc_curve(y_true, y_score)
            auc_score = roc_auc_score(y_true, y_score)
            
            youden_idx = np.argmax(tpr - fpr)
            thr_otimo = thresholds[youden_idx]
            
            # 👉 criar variável antes
            thr_display = thr_otimo
            
            plt.figure(figsize=self.config['visualization']['figsize_roc'])
            plt.plot(fpr, tpr, label=f"ROC (AUC={auc_score:.2f})", linewidth=2)
            
            plt.scatter(
                fpr[youden_idx], 
                tpr[youden_idx], 
                s=100,
                label=f"Youden (thr={thr_display:.4f})",
                color='red',
                zorder=5
            )
            
            plt.plot([0, 1], [0, 1], "--", color='gray', alpha=0.5)

            
            title = self.config['visualization']['plot_title_roc'].format(uf=identifier)
            plt.title(title)
            plt.xlabel(self.config['visualization']['xlabel_roc'])
            plt.ylabel(self.config['visualization']['ylabel_roc'])
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.legend(loc="lower right")
            
            if self.config['output']['save_files']:
                path = os.path.join(self.plots_dir, f"roc_{identifier}.png")
                plt.savefig(path, dpi=self.config['visualization']['dpi'], bbox_inches="tight")
                print(f"  💾 Gráfico salvo: {os.path.basename(path)}")
            
            if self.config['output']['display_plots']:
                plt.show()
            else:
                plt.close()
            
            return thr_otimo, auc_score
            
        except Exception as e:
            print(f"❌ Erro ao plotar curva ROC para {identifier}: {e}")
            plt.close()
            return 0.5, 0.5


class MetricsAnalyzer:
    """Analisador principal de métricas."""
    
    def __init__(self, config_path: str):
        """
        Inicializa analisador de métricas.
        
        Args:
            config_path: Caminho para arquivo de configuração YAML
        """
        self.config = ConfigLoader.load_config(config_path)
        ConfigLoader.validate_config(self.config)
        
        self.df = None
        self.score_col = None
        self.label_col = None
        self.uf_col = None
        self.plot_generator = None
        
    def load_and_prepare_data(self) -> None:
        """Carrega e prepara dados para análise."""
        print("\n" + "="*60)
        print("📊 CARREGANDO E PREPARANDO DADOS")
        print("="*60)
        
        # Carregar dados
        self.df = DataLoader.load_data(self.config['input']['data_path'])
        
        # Detectar colunas
        print("\n🔍 Detectando colunas relevantes...")
        self.score_col = ColumnDetector.detect_score_column(
            self.df, 
            self.config['input'].get('score_column')
        )
        self.label_col = ColumnDetector.detect_label_column(
            self.df, 
            self.config['input'].get('label_column')
        )
        
        if self.config['analysis']['analyze_by_uf']:
            self.uf_col = ColumnDetector.detect_uf_column(
                self.df, 
                self.config['input'].get('uf_column')
            )
        
        # Converter e validar tipos de dados
        self.df = DataProcessor.convert_and_validate_columns(
            self.df,
            self.score_col,
            self.label_col,
            self.uf_col
        )
        
        # Extrair UF se necessário
        if self.config['analysis']['analyze_by_uf'] and self.uf_col:
            # Extrair UF (2 primeiros dígitos)
            self.df['uf_extracted'] = self.df[self.uf_col].astype(str).str[:2]
            print(f"\n🗺️  UFs extraídas: {sorted(self.df['uf_extracted'].unique())}")
        
        # Preparar diretório de saída
        if self.config['output']['save_files']:
            os.makedirs(self.config['output']['output_dir'], exist_ok=True)
            print(f"\n📁 Diretório de saída: {self.config['output']['output_dir']}")
        
        # Inicializar gerador de plots
        self.plot_generator = PlotGenerator(
            self.config, 
            self.config['output']['output_dir']
        )
    
    def analyze_subset(
        self, 
        df_subset: pd.DataFrame, 
        identifier: str
    ) -> Optional[Dict[str, Any]]:
        """
        Analisa um subconjunto de dados.
        
        Args:
            df_subset: DataFrame para análise
            identifier: Identificador do subconjunto
            
        Returns:
            Dicionário com métricas ou None se erro
        """
        try:
            # Verificar dados válidos
            df_clean = df_subset[[self.label_col, self.score_col]].dropna()
            
            if df_clean.empty:
                print(f"  ⚠️  {identifier}: Sem dados válidos")
                return None
            
            if df_clean[self.label_col].nunique() < 2:
                print(f"  ⚠️  {identifier}: Apenas uma classe presente")
                return None
            
            print(f"\n📈 Analisando: {identifier} ({len(df_clean):,} registros)")
            
            # Gerar gráficos
            self.plot_generator.plot_confusion_matrix_by_threshold(
                df_clean, 
                self.label_col, 
                self.score_col, 
                identifier
            )
            
            thr_otimo, auc_score = self.plot_generator.plot_roc_curve(
                df_clean, 
                self.label_col, 
                self.score_col, 
                identifier
            )
            
            # Calcular métricas
            y_true = df_clean[self.label_col].astype(int).values
            y_score = df_clean[self.score_col].values
            
            acc, prec, rec, f1, esp, VP, FP, FN, VN = MetricsCalculator.calculate_metrics(
                y_true, y_score, thr_otimo
            )
            
            metrics = {
                "identificador": identifier,
                "threshold_otimo": round(thr_otimo, 6),
                "VP": int(VP),
                "FP": int(FP),
                "FN": int(FN),
                "VN": int(VN),
                "auc": round(auc_score, 4),
                "acuracia": round(acc, 4),
                "precisao": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "especificidade": round(esp, 4),
                "n_registros": len(df_clean)
            }
            
            print(f"  ✅ AUC: {auc_score:.4f} | F1: {f1:.4f} | Threshold: {thr_otimo:.4f}")
            
            return metrics
            
        except Exception as e:
            print(f"  ❌ Erro ao analisar {identifier}: {e}")
            return None
    
    def run_analysis(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """
        Executa análise completa.
        
        Returns:
            Tupla com (métricas_por_uf, métricas_geral)
        """
        print("\n" + "="*60)
        print("🚀 INICIANDO ANÁLISE DE MÉTRICAS")
        print("="*60)
        
        metricas_por_uf = []
        metricas_geral = None
        
        # Análise por UF
        if self.config['analysis']['analyze_by_uf'] and self.uf_col:
            print("\n📍 Análise por UF")
            print("-" * 60)
            
            ufs = sorted(self.df['uf_extracted'].dropna().unique())
            
            for uf in ufs:
                df_uf = self.df[self.df['uf_extracted'] == uf]
                metrics = self.analyze_subset(df_uf, f"UF_{uf}")
                
                if metrics:
                    metricas_por_uf.append(metrics)
        
        # Análise geral
        if self.config['analysis']['analyze_overall']:
            print("\n🌍 Análise Geral")
            print("-" * 60)
            
            metrics_geral = self.analyze_subset(self.df, "GERAL")
            
            if metrics_geral:
                metricas_geral = pd.DataFrame([metrics_geral])
        
        # Salvar resultados
        df_metricas_uf = None
        if metricas_por_uf:
            df_metricas_uf = pd.DataFrame(metricas_por_uf)
            
            if self.config['output']['save_files'] and self.config['output']['save_metrics']:
                path_uf = os.path.join(self.config['output']['output_dir'], "metricas_por_uf.csv")
                df_metricas_uf.to_csv(path_uf, index=False)
                print(f"\n💾 Métricas por UF salvas: {path_uf}")
        
        if metricas_geral is not None and self.config['output']['save_files'] and self.config['output']['save_metrics']:
            path_geral = os.path.join(self.config['output']['output_dir'], "metricas_geral.csv")
            metricas_geral.to_csv(path_geral, index=False)
            print(f"💾 Métricas gerais salvas: {path_geral}")
        
        
        from IPython.display import display
        
        print("\n" + "="*60)
        print("✅ ANÁLISE CONCLUÍDA COM SUCESSO")
        print("="*60)
        
        # Consolidar métricas UF + Geral
        dfs_para_consolidar = []
        
        if df_metricas_uf is not None:
            dfs_para_consolidar.append(df_metricas_uf)
        
        if metricas_geral is not None:
            dfs_para_consolidar.append(metricas_geral)
        
        df_final = None
        if dfs_para_consolidar:
            df_final = pd.concat(dfs_para_consolidar, ignore_index=True)
        
        # 👇 EXIBIR BONITO NO NOTEBOOK
        if df_final is not None:
            print("\n📊 Métricas Consolidadas:")
            display(df_final.style.format({
                "auc": "{:.4f}",
                "acuracia": "{:.4f}",
                "precisao": "{:.4f}",
                "recall": "{:.4f}",
                "f1_score": "{:.4f}",
                "especificidade": "{:.4f}"
            }))
        
        return df_metricas_uf, metricas_geral



def main(config_path: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Função principal para execução da análise.
    
    Args:
        config_path: Caminho para arquivo de configuração YAML
        
    Returns:
        Tupla com (métricas_por_uf, métricas_geral)
    """
    try:
        analyzer = MetricsAnalyzer(config_path)
        analyzer.load_and_prepare_data()
        return analyzer.run_analysis()
        
    except Exception as e:
        print(f"\n❌ ERRO FATAL: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    # Exemplo de uso
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = "config.yaml"
    
    df_uf, df_geral = main(config_path)
    
    if df_uf is not None:
        print("\n📊 Prévia - Métricas por UF:")
        print(df_uf.head())
    
    if df_geral is not None:
        print("\n📊 Prévia - Métricas Gerais:")
        print(df_geral)
