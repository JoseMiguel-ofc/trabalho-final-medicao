# Assistentes de IA no desenvolvimento de software

Survey pré-registrada sobre produtividade percebida, confiança e verificação do código gerado por assistentes de IA, com desenvolvedores que atuam no Brasil.

Disciplina: Medição e Experimentação em Engenharia de Software
Grupo: José Miguel Quintão Magalhães, Gabriel Peçanha Santiago, Enrico Vieira Paz de Bessa, Otavio Salomão Ferreira Silva, Pedro Moreira Ramos

- Pré-registro: 13/11/2026, antes da primeira resposta
- Pré-teste: 4 desenvolvedores, de 24/11 a 25/11/2026, sem mudanças no questionário (resultado em [DESVIOS.md](DESVIOS.md#resultado-do-pré-teste))
- Coleta: encerrada em 27/11/2026
- Amostra: 50 respostas brutas, 32 válidas
- Desvios do pré-registro: ver [DESVIOS.md](DESVIOS.md)

## Estrutura

| Arquivo | Conteúdo |
|---|---|
| `Protocolo_Survey_Assistentes_IA.pdf` | Protocolo pré-registrado em 13/11/2026, sem alterações |
| `analise.py` | Limpeza (seção 5.1), descritiva e confiabilidade (5.2) e testes confirmatórios com Holm-Bonferroni (5.3) |
| `graficos.py` | Gera os gráficos da apresentação a partir dos mesmos dados |
| `requirements.txt` | Dependências Python |
| `DESVIOS.md` | Adendo com os desvios do pré-registro (D1–D6), suas justificativas e o resultado do pré-teste |
| `saida/dados_limpos.csv` | Dataset válido e anonimizado (sem respostas abertas) |
| `saida/codificacao_A1.csv` | Respostas abertas válidas, revisadas para nomes, com colunas para os dois avaliadores |
| `saida/resultados.txt` | Relatório completo da análise |
| `saida/resultados.json` | Números principais, lidos pelo dashboard e por `graficos.py` |
| `relatorio/` | Seção de resultados, desvios e limitações (PDF) e a apresentação (PDF) |
| `saida/figuras/` | Barras divergentes Likert, mapa de calor das correlações, gráfico de pontos (QP2) e forest plot (H2/H3), em SVG e PNG |

## Como reproduzir

```bash
pip install -r requirements.txt
python analise.py "Práticas de desenvolvimento de software com e sem IA.csv"
python graficos.py "Práticas de desenvolvimento de software com e sem IA.csv"
```

O CSV bruto exportado do Google Forms não está no repositório (seção 6 do protocolo: os dados brutos ficam restritos ao grupo e ao professor). Os scripts localizam as colunas pelo texto das perguntas, não pela posição.

## Resultados em uma linha por hipótese

| Hipótese | Resultado |
|---|---|
| H2: produtividade percebida × experiência | Não rejeitada (nenhum teste significativo após Holm) |
| H3: confiança × verificação | Não rejeitada (maior efeito: CONF × V2, ρ = −0,33, p Holm = 0,67) |
| H4a, H4b: uso × política e porte | Inviáveis: 100% dos válidos usam IA |

Com n = 32, só correlações a partir de |ρ| ≈ 0,48 seriam detectadas com 80% de poder. As conclusões são exploratórias e restritas à percepção dos respondentes.
