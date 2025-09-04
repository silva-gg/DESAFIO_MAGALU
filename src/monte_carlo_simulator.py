"""
Módulo de Simulação Monte Carlo para Análise de Riscos
Simula cenários de incerteza para validar a robustez da decisão de localização
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class MonteCarloSimulator:
    """Classe para simulação Monte Carlo de cenários"""
    
    def __init__(self, config):
        self.config = config
        self.n_simulations = config.MONTE_CARLO_ITERACOES
        
    def define_uncertainty_distributions(self) -> Dict:
        """Define distribuições de incerteza para variáveis-chave"""
        distributions = {
            'preco_imovel_variacao': {
                'type': 'normal',
                'params': {'mean': 1.0, 'std': 0.15}  # ±15% variação
            },
            'tempo_entrega_variacao': {
                'type': 'normal', 
                'params': {'mean': 1.0, 'std': 0.10}  # ±10% variação
            },
            'crescimento_mercado': {
                'type': 'triangular',
                'params': {'left': 0.02, 'mode': 0.05, 'right': 0.08}  # 2-8% crescimento anual
            },
            'inflacao_custos': {
                'type': 'normal',
                'params': {'mean': 0.06, 'std': 0.02}  # 6% ±2% inflação
            },
            'variacao_demanda': {
                'type': 'normal',
                'params': {'mean': 1.0, 'std': 0.20}  # ±20% variação demanda
            },
            'risco_regulatorio': {
                'type': 'bernoulli',
                'params': {'p': 0.15}  # 15% chance de mudança regulatória
            }
        }
        return distributions
    
    def generate_random_samples(self, distributions: Dict) -> Dict:
        """Gera amostras aleatórias das distribuições"""
        samples = {}
        
        for var_name, dist_info in distributions.items():
            if dist_info['type'] == 'normal':
                samples[var_name] = np.random.normal(
                    dist_info['params']['mean'],
                    dist_info['params']['std'],
                    self.n_simulations
                )
            elif dist_info['type'] == 'triangular':
                samples[var_name] = np.random.triangular(
                    dist_info['params']['left'],
                    dist_info['params']['mode'],
                    dist_info['params']['right'],
                    self.n_simulations
                )
            elif dist_info['type'] == 'bernoulli':
                samples[var_name] = np.random.binomial(
                    1, dist_info['params']['p'], self.n_simulations
                )
        
        return samples
    
    def simulate_tco_scenarios(self, dados_base: Dict) -> Dict:
        """Simula cenários de TCO para ambas as cidades"""
        distributions = self.define_uncertainty_distributions()
        samples = self.generate_random_samples(distributions)
        
        resultados_simulacao = {
            'recife': {'tco_scenarios': [], 'roi_scenarios': []},
            'salvador': {'tco_scenarios': [], 'roi_scenarios': []}
        }
        
        for i in range(self.n_simulations):
            for cidade in ['recife', 'salvador']:
                cidade_data = dados_base[cidade].copy()
                
                # Aplicar variações estocásticas
                preco_base = cidade_data['imobiliario']['preco_m2_industrial']
                preco_ajustado = preco_base * samples['preco_imovel_variacao'][i]
                
                # Calcular TCO ajustado
                tco_base = self.calculate_adjusted_tco(
                    cidade_data, preco_ajustado, samples, i
                )
                
                # Calcular ROI considerando crescimento de mercado
                roi = self.calculate_roi_scenario(
                    tco_base, samples['crescimento_mercado'][i], 
                    samples['variacao_demanda'][i]
                )
                
                resultados_simulacao[cidade]['tco_scenarios'].append(tco_base)
                resultados_simulacao[cidade]['roi_scenarios'].append(roi)
        
        return resultados_simulacao
    
    def calculate_adjusted_tco(self, cidade_data: Dict, preco_ajustado: float, 
                             samples: Dict, iteration: int) -> float:
        """Calcula TCO ajustado para uma iteração específica"""
        area_m2 = self.config.CAPACIDADE_CD_M2
        
        # Custo de instalação ajustado
        custo_instalacao = area_m2 * preco_ajustado
        
        # Custos operacionais com inflação
        num_funcionarios = (area_m2 / 1000) * self.config.FUNCIONARIOS_POR_1000M2
        salario_base = 45000
        
        custos_anuais = []
        for ano in range(self.config.TCO_HORIZONTE_ANOS):
            inflacao_acumulada = (1 + samples['inflacao_custos'][iteration]) ** ano
            custo_mao_obra = num_funcionarios * salario_base * inflacao_acumulada
            
            # Custos de frete com variação
            custos_frete_base = [d['custo_frete_estimado'] for d in cidade_data['distancias'].values()]
            custo_frete = np.mean(custos_frete_base) * 1000 * inflacao_acumulada
            custo_frete *= samples['tempo_entrega_variacao'][iteration]
            
            # Impacto de risco regulatório
            if samples['risco_regulatorio'][iteration] == 1:
                custo_adicional = custo_mao_obra * 0.10  # 10% adicional por compliance
            else:
                custo_adicional = 0
            
            custo_anual_total = custo_mao_obra + custo_frete + custo_adicional
            custos_anuais.append(custo_anual_total)
        
        # Valor presente dos custos
        vp_custos = sum([custo / (1 + self.config.TAXA_DESCONTO)**i 
                        for i, custo in enumerate(custos_anuais)])
        
        tco_total = custo_instalacao + vp_custos
        return tco_total / 1e6  # Em milhões
    
    def calculate_roi_scenario(self, tco: float, crescimento_mercado: float, 
                             variacao_demanda: float) -> float:
        """Calcula ROI para um cenário específico"""
        # Receita estimada baseada no potencial de mercado
        receita_anual_base = 50e6  # R$ 50 milhões base
        
        receitas_anuais = []
        for ano in range(self.config.TCO_HORIZONTE_ANOS):
            crescimento_acumulado = (1 + crescimento_mercado) ** ano
            receita_ano = receita_anual_base * crescimento_acumulado * variacao_demanda
            receitas_anuais.append(receita_ano)
        
        vp_receitas = sum([receita / (1 + self.config.TAXA_DESCONTO)**i 
                          for i, receita in enumerate(receitas_anuais)])
        
        roi = (vp_receitas/1e6 - tco) / tco * 100  # ROI em %
        return roi
    
    def analyze_risk_metrics(self, simulacao_resultados: Dict) -> Dict:
        """Calcula métricas de risco a partir dos resultados da simulação"""
        metricas = {}
        
        for cidade in ['recife', 'salvador']:
            tco_scenarios = np.array(simulacao_resultados[cidade]['tco_scenarios'])
            roi_scenarios = np.array(simulacao_resultados[cidade]['roi_scenarios'])
            
            metricas[cidade] = {
                'tco_stats': {
                    'mean': np.mean(tco_scenarios),
                    'std': np.std(tco_scenarios),
                    'percentile_5': np.percentile(tco_scenarios, 5),
                    'percentile_95': np.percentile(tco_scenarios, 95),
                    'var_95': np.percentile(tco_scenarios, 95) - np.mean(tco_scenarios)
                },
                'roi_stats': {
                    'mean': np.mean(roi_scenarios),
                    'std': np.std(roi_scenarios),
                    'percentile_5': np.percentile(roi_scenarios, 5),
                    'percentile_95': np.percentile(roi_scenarios, 95),
                    'prob_positive': np.sum(roi_scenarios > 0) / len(roi_scenarios)
                },
                'risk_metrics': {
                    'coefficient_variation_tco': np.std(tco_scenarios) / np.mean(tco_scenarios),
                    'sharpe_ratio': np.mean(roi_scenarios) / np.std(roi_scenarios) if np.std(roi_scenarios) > 0 else 0,
                    'downside_risk': np.std(roi_scenarios[roi_scenarios < 0]) if np.any(roi_scenarios < 0) else 0
                }
            }
        
        return metricas
    
    def run_monte_carlo_analysis(self, dados: Dict) -> Dict:
        """Executa análise completa Monte Carlo"""
        print(f"Executando simulação Monte Carlo com {self.n_simulations:,} iterações...")
        
        # Executar simulação
        resultados_simulacao = self.simulate_tco_scenarios(dados)
        
        # Calcular métricas de risco
        metricas_risco = self.analyze_risk_metrics(resultados_simulacao)
        
        # Análise comparativa
        tco_recife = np.array(resultados_simulacao['recife']['tco_scenarios'])
        tco_salvador = np.array(resultados_simulacao['salvador']['tco_scenarios'])
        
        probabilidade_recife_melhor = np.sum(tco_recife < tco_salvador) / self.n_simulations
        
        roi_recife = np.array(resultados_simulacao['recife']['roi_scenarios'])
        roi_salvador = np.array(resultados_simulacao['salvador']['roi_scenarios'])
        
        prob_roi_recife_melhor = np.sum(roi_recife > roi_salvador) / self.n_simulations
        
        relatorio_mc = {
            'resultados_simulacao': resultados_simulacao,
            'metricas_risco': metricas_risco,
            'analise_comparativa': {
                'prob_recife_menor_tco': probabilidade_recife_melhor,
                'prob_recife_maior_roi': prob_roi_recife_melhor,
                'recomendacao_risco': 'Recife' if probabilidade_recife_melhor > 0.6 else 'Salvador' if probabilidade_recife_melhor < 0.4 else 'Decisão Marginal'
            },
            'parametros_simulacao': {
                'n_iteracoes': self.n_simulations,
                'horizonte_anos': self.config.TCO_HORIZONTE_ANOS,
                'taxa_desconto': self.config.TAXA_DESCONTO
            }
        }
        
        print("Simulação Monte Carlo concluída!")
        return relatorio_mc