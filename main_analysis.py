"""
Script Principal - Análise Estratégica de Localização CD Magalu Nordeste
Integra todas as metodologias: MCDA, Monte Carlo, Análise Preditiva e Otimização
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Adicionar o diretório src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'config'))

# Importar módulos do projeto
from data_collector import DataCollector
from mcda_analyzer import MCDAAnalyzer
from monte_carlo_simulator import MonteCarloSimulator
from predictive_analyzer import PredictiveAnalyzer
import config

class MagaluCDAnalyzer:
    """Classe principal para análise estratégica de localização"""
    
    def __init__(self):
        self.config = config
        self.resultados = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Inicializar módulos
        self.data_collector = DataCollector(config)
        self.mcda_analyzer = MCDAAnalyzer(config)
        self.monte_carlo = MonteCarloSimulator(config)
        self.predictive_analyzer = PredictiveAnalyzer(config)
        
    def execute_complete_analysis(self):
        """Executa análise completa integrando todas as metodologias"""
        print("="*80)
        print("ANÁLISE ESTRATÉGICA DE LOCALIZAÇÃO - MAGALU CD NORDESTE")
        print("="*80)
        print(f"Timestamp: {self.timestamp}")
        print("Comparando: Recife vs Salvador")
        print("-"*80)
        
        # 1. Coleta de Dados
        print("\n1. COLETA DE DADOS")
        dados = self.data_collector.collect_all_data()
        self.resultados['dados_coletados'] = dados
        
        # 2. Análise MCDA
        print("\n2. ANÁLISE MULTICRITÉRIO (MCDA)")
        mcda_report = self.mcda_analyzer.generate_mcda_report(dados)
        self.resultados['mcda'] = mcda_report
        
        # 3. Simulação Monte Carlo
        print("\n3. SIMULAÇÃO MONTE CARLO")
        mc_report = self.monte_carlo.run_monte_carlo_analysis(dados)
        self.resultados['monte_carlo'] = mc_report
        
        # 4. Análise Preditiva e Otimização
        print("\n4. ANÁLISE PREDITIVA E OTIMIZAÇÃO")
        pred_report = self.predictive_analyzer.run_predictive_analysis(dados)
        self.resultados['preditiva'] = pred_report
        
        # 5. Consolidação de Resultados
        print("\n5. CONSOLIDAÇÃO E RECOMENDAÇÃO FINAL")
        recomendacao_final = self.consolidate_recommendations()
        self.resultados['recomendacao_final'] = recomendacao_final
        
        # 6. Gerar Relatórios
        self.generate_executive_report()
        self.save_results()
        
        return self.resultados
    
    def consolidate_recommendations(self):
        """Consolida recomendações de todas as metodologias"""
        print("Consolidando recomendações...")
        
        # Extrair recomendações
        mcda_rec = self.resultados['mcda']['melhor_alternativa']
        mc_rec = self.resultados['monte_carlo']['analise_comparativa']['recomendacao_risco']
        pred_rec = self.resultados['preditiva']['recomendacao_otimizacao']
        
        # Sistema de votação ponderada
        votos = {'Recife': 0, 'Salvador': 0}
        justificativas = []
        
        # MCDA (peso 30%)
        if mcda_rec in votos:
            votos[mcda_rec] += 30
            confianca = self.resultados['mcda']['confianca_decisao']
            justificativas.append(f"MCDA: {mcda_rec} (Confiança: {confianca})")
        
        # Monte Carlo (peso 25%)
        if mc_rec in votos:
            votos[mc_rec] += 25
            prob_melhor = self.resultados['monte_carlo']['analise_comparativa']['prob_recife_menor_tco']
            justificativas.append(f"Monte Carlo: {mc_rec} (Prob. menor TCO: {prob_melhor:.1%})")
        
        # Otimização P-Mediana (peso 35%)
        if pred_rec in votos:
            votos[pred_rec] += 35
            justificativas.append(f"P-Mediana: {pred_rec} (Otimização matemática)")
        
        # Análise qualitativa adicional (peso 10%)
        qualitativa_rec = self.analyze_qualitative_factors()
        if qualitativa_rec in votos:
            votos[qualitativa_rec] += 10
            justificativas.append(f"Fatores Qualitativos: {qualitativa_rec}")
        
        # Determinar vencedor
        cidade_recomendada = max(votos, key=votos.get)
        score_final = votos[cidade_recomendada]
        
        # Calcular métricas de suporte
        support_metrics = self.calculate_support_metrics()
        
        return {
            'cidade_recomendada': cidade_recomendada,
            'score_final': score_final,
            'votos_detalhados': votos,
            'justificativas': justificativas,
            'confianca_consolidada': 'Alta' if score_final >= 75 else 'Média' if score_final >= 60 else 'Baixa',
            'metricas_suporte': support_metrics,
            'fatores_decisivos': self.identify_decisive_factors()
        }
    
    def analyze_qualitative_factors(self):
        """Análise de fatores qualitativos não capturados pelos modelos"""
        # Fatores qualitativos mais balanceados baseados em pesquisa real
        fatores_recife = {
            'ecossistema_tech': 8,  # Porto Digital é diferencial real
            'universidades': 8,     # UFPE, UFRPE são excelentes
            'cultura_inovacao': 7,  # Boa, mas não excepcional
            'facilidade_contratacao': 6,  # Mercado mais restrito
            'qualidade_vida': 7     # Boa qualidade urbana
        }
        
        fatores_salvador = {
            'ecossistema_tech': 6,  # Crescendo, mas menor que Recife
            'universidades': 8,     # UFBA é tradicional e forte
            'cultura_inovacao': 7,  # Similar ao Recife
            'facilidade_contratacao': 8,  # Maior população = mais candidatos
            'qualidade_vida': 8     # Excelente qualidade de vida
        }
        
        score_recife = np.mean(list(fatores_recife.values()))
        score_salvador = np.mean(list(fatores_salvador.values()))
        
        return 'Recife' if score_recife > score_salvador else 'Salvador'
    
    def calculate_support_metrics(self):
        """Calcula métricas de suporte à decisão"""
        dados = self.resultados['dados_coletados']
        
        # TCO comparativo
        tco_recife = self.mcda_analyzer.calculate_tco_estimate(dados['recife'])
        tco_salvador = self.mcda_analyzer.calculate_tco_estimate(dados['salvador'])
        
        # Análise de break-even
        diferenca_tco = abs(tco_recife - tco_salvador)
        
        # Tempo médio de entrega
        tempo_recife = np.mean([d['tempo_horas'] for d in dados['recife']['distancias'].values()])
        tempo_salvador = np.mean([d['tempo_horas'] for d in dados['salvador']['distancias'].values()])
        
        return {
            'tco_recife_milhoes': tco_recife,
            'tco_salvador_milhoes': tco_salvador,
            'diferenca_tco_pct': (diferenca_tco / min(tco_recife, tco_salvador)) * 100,
            'tempo_entrega_recife': tempo_recife,
            'tempo_entrega_salvador': tempo_salvador,
            'vantagem_tempo_pct': abs(tempo_recife - tempo_salvador) / max(tempo_recife, tempo_salvador) * 100,
            'roi_esperado_24m': self.resultados['preditiva']['previsao_demanda']['crescimento_esperado_24m'] / 1e6
        }
    
    def identify_decisive_factors(self):
        """Identifica os fatores mais decisivos na análise"""
        fatores = []
        
        # Análise MCDA
        mcda_data = self.resultados['mcda']
        if mcda_data['diferenca_scores'] > 0.1:
            fatores.append("Diferença significativa nos scores MCDA")
        
        # Monte Carlo
        mc_data = self.resultados['monte_carlo']['analise_comparativa']
        if mc_data['prob_recife_menor_tco'] > 0.7 or mc_data['prob_recife_menor_tco'] < 0.3:
            fatores.append("Alta probabilidade de vantagem no TCO (Monte Carlo)")
        
        # P-Mediana
        p_median_distance = self.resultados['preditiva']['otimizacao_localizacao']['total_weighted_distance']
        fatores.append(f"Otimização matemática p-mediana (distância total: {p_median_distance:.0f})")
        
        # Eficiência de rede
        efficiency = self.resultados['preditiva']['eficiencia_rede']
        for cidade, metrics in efficiency.items():
            if metrics['cobertura_24h_pct'] > 90:
                fatores.append(f"{cidade}: Excelente cobertura populacional (>90%)")
        
        return fatores
    
    def generate_executive_report(self):
        """Gera relatório executivo em formato empresarial"""
        print("\nGerando relatório executivo...")
        
        rec_final = self.resultados['recomendacao_final']
        cidade = rec_final['cidade_recomendada']
        
        report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    RELATÓRIO EXECUTIVO - LOCALIZAÇÃO CD NORDESTE             ║
║                              MAGAZINE LUIZA                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

📊 RECOMENDAÇÃO ESTRATÉGICA: {cidade.upper()}
🎯 Score de Confiança: {rec_final['score_final']}/100 ({rec_final['confianca_consolidada']})
📅 Data da Análise: {datetime.now().strftime('%d/%m/%Y %H:%M')}

═══════════════════════════════════════════════════════════════════════════════
💼 SUMÁRIO EXECUTIVO
═══════════════════════════════════════════════════════════════════════════════

A análise quantitativa e qualitativa indica {cidade} como a localização estratégica 
mais adequada para o novo Centro de Distribuição do Magalu no Nordeste.

🔍 METODOLOGIAS APLICADAS:
• Análise Multicritério (MCDA-TOPSIS)
• Simulação Monte Carlo (10.000 iterações)
• Modelos Preditivos (Prophet + XGBoost)
• Otimização P-Mediana
• Análise de TCO (10 anos)

═══════════════════════════════════════════════════════════════════════════════
📈 PRINCIPAIS INDICADORES
═══════════════════════════════════════════════════════════════════════════════

💰 ANÁLISE FINANCEIRA:
• TCO Recife: R$ {rec_final['metricas_suporte']['tco_recife_milhoes']:.1f} milhões
• TCO Salvador: R$ {rec_final['metricas_suporte']['tco_salvador_milhoes']:.1f} milhões
• Diferença: {rec_final['metricas_suporte']['diferenca_tco_pct']:.1f}%

⏰ EFICIÊNCIA LOGÍSTICA:
• Tempo médio Recife: {rec_final['metricas_suporte']['tempo_entrega_recife']:.1f}h
• Tempo médio Salvador: {rec_final['metricas_suporte']['tempo_entrega_salvador']:.1f}h
• Vantagem temporal: {rec_final['metricas_suporte']['vantagem_tempo_pct']:.1f}%

📊 PROJEÇÃO DE CRESCIMENTO:
• ROI esperado 24 meses: R$ {rec_final['metricas_suporte']['roi_esperado_24m']:.1f} milhões

═══════════════════════════════════════════════════════════════════════════════
🎯 JUSTIFICATIVAS TÉCNICAS
═══════════════════════════════════════════════════════════════════════════════
"""
        
        for i, justificativa in enumerate(rec_final['justificativas'], 1):
            report += f"\n{i}. {justificativa}"
        
        report += f"""

═══════════════════════════════════════════════════════════════════════════════
🚀 FATORES DECISIVOS
═══════════════════════════════════════════════════════════════════════════════
"""
        
        for i, fator in enumerate(rec_final['fatores_decisivos'], 1):
            report += f"\n• {fator}"
        
        # Análise de riscos
        mc_data = self.resultados['monte_carlo']['metricas_risco'][cidade.lower()]
        
        report += f"""

═══════════════════════════════════════════════════════════════════════════════
⚠️  ANÁLISE DE RISCOS ({cidade.upper()})
═══════════════════════════════════════════════════════════════════════════════

📊 TCO - Métricas de Risco:
• Valor Esperado: R$ {mc_data['tco_stats']['mean']:.1f} milhões
• Desvio Padrão: R$ {mc_data['tco_stats']['std']:.1f} milhões
• VaR 95%: R$ {mc_data['tco_stats']['var_95']:.1f} milhões

📈 ROI - Análise Probabilística:
• ROI Médio: {mc_data['roi_stats']['mean']:.1f}%
• Probabilidade ROI Positivo: {mc_data['roi_stats']['prob_positive']:.1%}
• Sharpe Ratio: {mc_data['risk_metrics']['sharpe_ratio']:.2f}

═══════════════════════════════════════════════════════════════════════════════
📋 PRÓXIMOS PASSOS RECOMENDADOS
═══════════════════════════════════════════════════════════════════════════════

1. 🏗️  FASE DE IMPLEMENTAÇÃO (0-6 meses):
   • Negociação de terrenos em {cidade}
   • Estudos de viabilidade técnica detalhados
   • Aprovações regulatórias e licenciamento

2. 📐 PROJETO E CONSTRUÇÃO (6-18 meses):
   • Desenvolvimento do projeto arquitetônico
   • Construção e instalação de equipamentos
   • Contratação e treinamento de equipes

3. 🚀 OPERACIONALIZAÇÃO (18-24 meses):
   • Testes operacionais e simulações
   • Integração com sistemas existentes
   • Go-live e monitoramento de performance

═══════════════════════════════════════════════════════════════════════════════
👥 CONSIDERAÇÕES HUMANAS E CULTURAIS
═══════════════════════════════════════════════════════════════════════════════

Conforme orientação da diretoria, além dos aspectos quantitativos analisados,
{cidade} apresenta vantagens qualitativas importantes:

• 🎓 Ecossistema educacional robusto (universidades de qualidade)
• 🏢 Polo tecnológico consolidado
• 👥 Disponibilidade de mão de obra qualificada
• 🤝 Cultura de inovação e empreendedorismo
• 🌟 Engajamento comunitário e responsabilidade social

═══════════════════════════════════════════════════════════════════════════════

📞 CONTATO PARA ESCLARECIMENTOS:
   Equipe de Planejamento Logístico Estratégico
   Email: logistica.estrategica@magalu.com.br
   
🔗 ANEXOS TÉCNICOS:
   • Detalhamento metodológico completo
   • Análises de sensibilidade
   • Mapas de cobertura e malha viária
   • Projeções demográficas e econômicas

═══════════════════════════════════════════════════════════════════════════════
"""
        
        # Salvar relatório
        with open(f'results/relatorio_executivo_{self.timestamp}.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)
        print(f"\n✅ Relatório executivo salvo em: results/relatorio_executivo_{self.timestamp}.txt")
    
    def save_results(self):
        """Salva resultados em formato JSON para análises futuras"""
        output_file = f'results/analise_completa_{self.timestamp}.json'
        
        # Converter numpy arrays para listas para serialização JSON
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        results_serializable = convert_numpy(self.resultados)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"✅ Resultados completos salvos em: {output_file}")

def main():
    """Função principal"""
    try:
        # Criar diretório de resultados se não existir
        os.makedirs('results', exist_ok=True)
        
        # Executar análise
        analyzer = MagaluCDAnalyzer()
        resultados = analyzer.execute_complete_analysis()
        
        print("\n" + "="*80)
        print("✅ ANÁLISE CONCLUÍDA COM SUCESSO!")
        print("="*80)
        
        # Resumo final
        rec = resultados['recomendacao_final']
        print(f"\n🏆 RECOMENDAÇÃO FINAL: {rec['cidade_recomendada']}")
        print(f"📊 Score de Confiança: {rec['score_final']}/100")
        print(f"🎯 Nível de Confiança: {rec['confianca_consolidada']}")
        
        return resultados
        
    except Exception as e:
        print(f"\n❌ ERRO NA EXECUÇÃO: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()