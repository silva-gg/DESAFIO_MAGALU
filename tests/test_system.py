"""
Script de Teste e Validação do Sistema de Análise de Localização CD Magalu
Valida todos os módulos e funcionalidades implementadas
"""

import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd

# Adicionar diretórios ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))

# Importar módulos do projeto
from data_collector import DataCollector
from mcda_analyzer import MCDAAnalyzer
from monte_carlo_simulator import MonteCarloSimulator
from predictive_analyzer import PredictiveAnalyzer
import config

class TestDataCollector(unittest.TestCase):
    """Testes para o módulo de coleta de dados"""
    
    def setUp(self):
        self.collector = DataCollector(config)
    
    def test_real_estate_data(self):
        """Testa coleta de dados imobiliários"""
        recife_data = self.collector.get_real_estate_data('Recife')
        salvador_data = self.collector.get_real_estate_data('Salvador')
        
        self.assertIn('preco_m2_industrial', recife_data)
        self.assertIn('preco_m2_industrial', salvador_data)
        self.assertGreater(recife_data['preco_m2_industrial'], 0)
        self.assertGreater(salvador_data['preco_m2_industrial'], 0)
    
    def test_distance_calculation(self):
        """Testa cálculo de distâncias"""
        origem = config.RECIFE_COORDS
        destinos = config.CAPITAIS_NORDESTE
        
        distancias = self.collector.calculate_distances_osm(origem, destinos)
        
        self.assertEqual(len(distancias), len(destinos))
        for cidade, dados in distancias.items():
            self.assertIn('distancia_km', dados)
            self.assertIn('tempo_horas', dados)
            self.assertGreater(dados['distancia_km'], 0)
            self.assertGreater(dados['tempo_horas'], 0)
    
    def test_infrastructure_score(self):
        """Testa avaliação de infraestrutura"""
        recife_infra = self.collector.get_infrastructure_score('Recife')
        salvador_infra = self.collector.get_infrastructure_score('Salvador')
        
        required_keys = ['porto_proximidade', 'aeroporto_qualidade', 'rodovias_qualidade']
        
        for key in required_keys:
            self.assertIn(key, recife_infra)
            self.assertIn(key, salvador_infra)
            self.assertGreaterEqual(recife_infra[key], 1)
            self.assertLessEqual(recife_infra[key], 5)

class TestMCDAAnalyzer(unittest.TestCase):
    """Testes para análise multicritério"""
    
    def setUp(self):
        self.analyzer = MCDAAnalyzer(config)
        # Dados de teste simplificados
        self.test_data = {
            'recife': {
                'imobiliario': {'preco_m2_industrial': 1200},
                'infraestrutura': {'porto_proximidade': 5, 'aeroporto_qualidade': 4, 'rodovias_qualidade': 4, 'mao_obra_disponivel': 4, 'universidades_proximas': 5},
                'distancias': {
                    'Fortaleza': {'tempo_horas': 8, 'custo_frete_estimado': 800},
                    'Natal': {'tempo_horas': 3, 'custo_frete_estimado': 300}
                }
            },
            'salvador': {
                'imobiliario': {'preco_m2_industrial': 1400},
                'infraestrutura': {'porto_proximidade': 4, 'aeroporto_qualidade': 4, 'rodovias_qualidade': 3, 'mao_obra_disponivel': 4, 'universidades_proximas': 4},
                'distancias': {
                    'Fortaleza': {'tempo_horas': 12, 'custo_frete_estimado': 1200},
                    'Natal': {'tempo_horas': 8, 'custo_frete_estimado': 800}
                }
            }
        }
    
    def test_decision_matrix_preparation(self):
        """Testa preparação da matriz de decisão"""
        matriz, alternativas = self.analyzer.prepare_decision_matrix(self.test_data)
        
        self.assertEqual(len(alternativas), 2)
        self.assertEqual(matriz.shape[0], 2)  # 2 alternativas
        self.assertEqual(matriz.shape[1], 5)  # 5 critérios
        self.assertTrue(np.all(matriz > 0))   # Todos os valores devem ser positivos
    
    def test_topsis_analysis(self):
        """Testa análise TOPSIS"""
        matriz, alternativas = self.analyzer.prepare_decision_matrix(self.test_data)
        resultado = self.analyzer.topsis_analysis(matriz, alternativas)
        
        self.assertIn('scores', resultado)
        self.assertIn('ranking', resultado)
        self.assertEqual(len(resultado['scores']), 2)
        
        # Scores devem estar entre 0 e 1
        for score in resultado['scores'].values():
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)
    
    def test_tco_calculation(self):
        """Testa cálculo de TCO"""
        tco_recife = self.analyzer.calculate_tco_estimate(self.test_data['recife'])
        tco_salvador = self.analyzer.calculate_tco_estimate(self.test_data['salvador'])
        
        self.assertGreater(tco_recife, 0)
        self.assertGreater(tco_salvador, 0)
        self.assertIsInstance(tco_recife, float)
        self.assertIsInstance(tco_salvador, float)

class TestMonteCarloSimulator(unittest.TestCase):
    """Testes para simulação Monte Carlo"""
    
    def setUp(self):
        # Configuração de teste com menos iterações
        test_config = MagicMock()
        test_config.MONTE_CARLO_ITERACOES = 100  # Reduzido para testes
        test_config.TCO_HORIZONTE_ANOS = 10
        test_config.TAXA_DESCONTO = 0.12
        test_config.CAPACIDADE_CD_M2 = 50000
        test_config.FUNCIONARIOS_POR_1000M2 = 25
        
        self.simulator = MonteCarloSimulator(test_config)
        
        # Dados de teste
        self.test_data = {
            'recife': {
                'imobiliario': {'preco_m2_industrial': 1200},
                'distancias': {
                    'Fortaleza': {'custo_frete_estimado': 800},
                    'Natal': {'custo_frete_estimado': 300}
                }
            },
            'salvador': {
                'imobiliario': {'preco_m2_industrial': 1400},
                'distancias': {
                    'Fortaleza': {'custo_frete_estimado': 1200},
                    'Natal': {'custo_frete_estimado': 800}
                }
            }
        }
    
    def test_uncertainty_distributions(self):
        """Testa definição de distribuições de incerteza"""
        distributions = self.simulator.define_uncertainty_distributions()
        
        required_vars = ['preco_imovel_variacao', 'tempo_entrega_variacao', 'crescimento_mercado']
        for var in required_vars:
            self.assertIn(var, distributions)
            self.assertIn('type', distributions[var])
            self.assertIn('params', distributions[var])
    
    def test_random_samples_generation(self):
        """Testa geração de amostras aleatórias"""
        distributions = self.simulator.define_uncertainty_distributions()
        samples = self.simulator.generate_random_samples(distributions)
        
        for var_name, sample_array in samples.items():
            self.assertEqual(len(sample_array), self.simulator.n_simulations)
            self.assertTrue(np.all(np.isfinite(sample_array)))
    
    def test_tco_scenarios_simulation(self):
        """Testa simulação de cenários de TCO"""
        resultados = self.simulator.simulate_tco_scenarios(self.test_data)
        
        self.assertIn('recife', resultados)
        self.assertIn('salvador', resultados)
        
        for cidade in ['recife', 'salvador']:
            self.assertIn('tco_scenarios', resultados[cidade])
            self.assertIn('roi_scenarios', resultados[cidade])
            self.assertEqual(len(resultados[cidade]['tco_scenarios']), self.simulator.n_simulations)
            self.assertEqual(len(resultados[cidade]['roi_scenarios']), self.simulator.n_simulations)

class TestPredictiveAnalyzer(unittest.TestCase):
    """Testes para análise preditiva"""
    
    def setUp(self):
        self.analyzer = PredictiveAnalyzer(config)
    
    def test_synthetic_data_generation(self):
        """Testa geração de dados sintéticos"""
        df = self.analyzer.generate_synthetic_demand_data()
        
        required_columns = ['ds', 'y', 'ecommerce_penetration']
        for col in required_columns:
            self.assertIn(col, df.columns)
        
        self.assertGreater(len(df), 0)
        self.assertTrue(df['y'].min() >= 0)  # Demanda não pode ser negativa
    
    def test_p_median_problem(self):
        """Testa resolução do problema p-mediana"""
        test_data = {
            'recife': {'distancias': {'Fortaleza': {'tempo_horas': 8}}},
            'salvador': {'distancias': {'Fortaleza': {'tempo_horas': 12}}}
        }
        
        resultado = self.analyzer.solve_p_median_problem(test_data)
        
        self.assertIn('selected_facilities', resultado)
        self.assertIn('total_weighted_distance', resultado)
        self.assertEqual(len(resultado['selected_facilities']), 1)
        self.assertIn(resultado['selected_facilities'][0], ['Recife', 'Salvador'])
    
    def test_network_efficiency_calculation(self):
        """Testa cálculo de eficiência da rede"""
        test_data = {
            'recife': {
                'distancias': {
                    'Fortaleza': {'tempo_horas': 8, 'custo_frete_estimado': 800},
                    'Natal': {'tempo_horas': 3, 'custo_frete_estimado': 300}
                }
            },
            'salvador': {
                'distancias': {
                    'Fortaleza': {'tempo_horas': 12, 'custo_frete_estimado': 1200},
                    'Natal': {'tempo_horas': 8, 'custo_frete_estimado': 800}
                }
            }
        }
        
        efficiency = self.analyzer.calculate_network_efficiency(test_data)
        
        self.assertIn('Recife', efficiency)
        self.assertIn('Salvador', efficiency)
        
        for cidade in efficiency.values():
            self.assertIn('tempo_medio_entrega', cidade)
            self.assertIn('cobertura_24h_pct', cidade)
            self.assertGreater(cidade['tempo_medio_entrega'], 0)

class TestSystemIntegration(unittest.TestCase):
    """Testes de integração do sistema completo"""
    
    def test_config_validity(self):
        """Testa validade da configuração"""
        self.assertIsNotNone(config.RECIFE_COORDS)
        self.assertIsNotNone(config.SALVADOR_COORDS)
        self.assertIsNotNone(config.CAPITAIS_NORDESTE)
        self.assertIsNotNone(config.CRITERIOS_PESOS)
        
        # Verificar se pesos somam 1
        total_peso = sum(config.CRITERIOS_PESOS.values())
        self.assertAlmostEqual(total_peso, 1.0, places=2)
    
    def test_module_imports(self):
        """Testa se todos os módulos podem ser importados"""
        try:
            from data_collector import DataCollector
            from mcda_analyzer import MCDAAnalyzer
            from monte_carlo_simulator import MonteCarloSimulator
            from predictive_analyzer import PredictiveAnalyzer
        except ImportError as e:
            self.fail(f"Falha ao importar módulos: {e}")
    
    @patch('requests.Session.get')
    def test_api_integration_mock(self, mock_get):
        """Testa integração com APIs (mock)"""
        # Mock da resposta da API
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'test': 'data'}
        mock_get.return_value = mock_response
        
        collector = DataCollector(config)
        result = collector.get_ibge_demographics('26')  # Pernambuco
        
        self.assertIsNotNone(result)

def run_performance_tests():
    """Executa testes de performance"""
    print("\n🚀 TESTES DE PERFORMANCE")
    print("-" * 50)
    
    import time
    
    # Teste de coleta de dados
    start_time = time.time()
    collector = DataCollector(config)
    dados = collector.collect_all_data()
    collection_time = time.time() - start_time
    print(f"⏱️  Coleta de dados: {collection_time:.2f}s")
    
    # Teste MCDA
    start_time = time.time()
    analyzer = MCDAAnalyzer(config)
    mcda_result = analyzer.generate_mcda_report(dados)
    mcda_time = time.time() - start_time
    print(f"⏱️  Análise MCDA: {mcda_time:.2f}s")
    
    # Teste Monte Carlo (versão reduzida)
    test_config = MagicMock()
    test_config.MONTE_CARLO_ITERACOES = 1000  # Reduzido para teste
    test_config.TCO_HORIZONTE_ANOS = 10
    test_config.TAXA_DESCONTO = 0.12
    test_config.CAPACIDADE_CD_M2 = 50000
    test_config.FUNCIONARIOS_POR_1000M2 = 25
    
    start_time = time.time()
    mc_simulator = MonteCarloSimulator(test_config)
    mc_result = mc_simulator.run_monte_carlo_analysis(dados)
    mc_time = time.time() - start_time
    print(f"⏱️  Monte Carlo (1k iter): {mc_time:.2f}s")
    
    print(f"\n✅ Tempo total de execução: {collection_time + mcda_time + mc_time:.2f}s")

def run_validation_tests():
    """Executa validação completa do sistema"""
    print("\n🔍 VALIDAÇÃO DE RESULTADOS")
    print("-" * 50)
    
    # Executar análise simplificada
    collector = DataCollector(config)
    dados = collector.collect_all_data()
    
    analyzer = MCDAAnalyzer(config)
    mcda_result = analyzer.generate_mcda_report(dados)
    
    # Validações lógicas
    print("✅ Dados coletados para ambas as cidades")
    print("✅ Matriz MCDA gerada com 5 critérios")
    print(f"✅ MCDA recomenda: {mcda_result['melhor_alternativa']}")
    print(f"✅ Confiança da decisão: {mcda_result['confianca_decisao']}")
    
    # Validar consistência dos dados
    assert 'recife' in dados and 'salvador' in dados
    assert len(mcda_result['criterios_utilizados']) == 5
    assert mcda_result['melhor_alternativa'] in ['Recife', 'Salvador']
    
    print("\n✅ Todas as validações passaram!")

def main():
    """Função principal de testes"""
    print("="*80)
    print("🧪 SISTEMA DE TESTES - ANÁLISE LOCALIZAÇÃO CD MAGALU")
    print("="*80)
    
    # Executar testes unitários
    print("\n📋 EXECUTANDO TESTES UNITÁRIOS")
    print("-" * 50)
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Adicionar classes de teste
    test_classes = [
        TestDataCollector,
        TestMCDAAnalyzer,
        TestMonteCarloSimulator,
        TestPredictiveAnalyzer,
        TestSystemIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Executar testes
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Resumo dos testes
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    success_rate = ((total_tests - failures - errors) / total_tests) * 100 if total_tests > 0 else 0
    
    print("\n" + "="*50)
    print(f"📊 RESUMO DOS TESTES")
    print("="*50)
    print(f"Total de testes: {total_tests}")
    print(f"Sucessos: {total_tests - failures - errors}")
    print(f"Falhas: {failures}")
    print(f"Erros: {errors}")
    print(f"Taxa de sucesso: {success_rate:.1f}%")
    
    if failures > 0 or errors > 0:
        print("\n❌ ALGUNS TESTES FALHARAM!")
        return False
    else:
        print("\n✅ TODOS OS TESTES PASSARAM!")
    
    # Executar testes de performance
    run_performance_tests()
    
    # Executar validação
    run_validation_tests()
    
    print("\n" + "="*80)
    print("✅ SISTEMA VALIDADO E PRONTO PARA USO!")
    print("="*80)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)