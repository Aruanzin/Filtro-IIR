# Processamento Digital de Sinais — Filtros IIR (Grupo B9)

Resolução do Projeto Final da disciplina **SEL0615 - Processamento Digital de Sinais (EESC-USP)**. O objetivo é desenvolver uma rotina de filtragem digital que mitigue o **ruído de medição** em sinais de corrente elétrica de uma planta industrial, **minimizando o RMSE** em relação ao sinal de referência (sem ruído) e avaliando o **custo computacional** associado.

## 📋 Contexto do Problema

A introdução de cargas elétricas não-lineares (retificadores, inversores, acionamentos de motores) provoca componentes harmônicas na corrente consumida. Um medidor no ponto de conexão com a rede de média tensão registra a corrente a **128 amostras por ciclo**. Como a fundamental da rede é $60\text{ Hz}$:

$$f_s = 128 \times 60 = 7680\text{ Hz}$$

O trabalho do **Grupo B9** consiste em projetar e aplicar **Filtros IIR** ao sinal `sinal_9`, comparando aproximações e estratégias de filtragem.

## ✨ O que esta implementação faz

- **Projeto via Seções de Segunda Ordem (SOS)** — cascata de biquads, numericamente muito mais estável que a forma direta `(b, a)` em ordens altas e cortes baixos.
- **Quatro aproximações** comparadas: Butterworth, Chebyshev I, Chebyshev II e Elíptica.
- **Fase zero vs. causal** — compara `sosfiltfilt` (offline, fase zero) com `sosfilt` (causal, realizável em tempo real), evidenciando o que é de fato implementável num medidor.
- **Busca em grade ampla** — varre tipo × ordem (1–8) × frequência de corte (300–3300 Hz) × método, com **verificação de estabilidade** (polos dentro do círculo unitário).
- **Métricas** — RMSE, **melhoria de SNR (dB)** e **custo computacional** (mediana do pipeline projeto+filtragem, robusta a *jitter* do SO).
- **Análise em frequência** — resposta em magnitude (`sosfreqz`) e **espectro FFT** antes/depois.
- **Análise no plano Z** — **diagrama de polos e zeros** do melhor filtro (estabilidade).
- **Custo vs. ordem** — gráfico de eixo duplo (RMSE × tempo) para escolher a ordem com consciência de custo.
- **Execução *zero-config*** — `python filtroIir.py` roda tudo e salva todas as figuras; *fallback* sintético se os CSVs não forem encontrados.

## 🛠️ Tecnologias e Dependências

Desenvolvido em **Python 3** com `numpy`, `scipy` (módulo `scipy.signal`), `pandas` e `matplotlib`.

## 📂 Estrutura do Repositório

```text
├── sinais/
│   ├── sinal_9_ruido.csv          # Sinal com ruído de medição (entrada)
│   └── sinal_9_semruido.csv       # Sinal de referência (validação)
├── filtroIir.py                   # Script principal (busca, métricas e figuras)
├── requirements.txt               # Dependências
├── resultado_filtros_iir_b9.png   # Comparativo no tempo (melhor de cada tipo)
├── melhor_filtro_iir_b9.png       # Destaque do melhor filtro global
├── resposta_frequencia_b9.png     # Resposta em frequência (freqz)
├── espectro_fft_b9.png            # Espectro FFT antes/depois
├── polos_zeros_b9.png             # Diagrama de polos e zeros (plano Z)
├── custo_vs_ordem_b9.png          # RMSE × custo computacional vs. ordem
├── causal_vs_fasezero_b9.png      # Filtragem causal vs. fase zero
├── resultados_busca_b9.csv        # Tabela completa da busca (gerada)
└── README.md
```

## ▶️ Como Executar

```bash
pip install -r requirements.txt
python filtroIir.py            # análise completa + todas as figuras
python filtroIir.py --rapido   # apenas a busca e o resumo no terminal
python filtroIir.py --fs 7680  # sobrescreve a frequência de amostragem
```

## 📈 Resultado

Para o `sinal_9`, o melhor filtro global é um **Chebyshev II de ordem 3** com $f_c = 2300\text{ Hz}$, atingindo **RMSE ≈ 0,32** (redução de ~46 % em relação ao sinal ruidoso). A análise de custo × ordem mostra que o RMSE estabiliza a partir da ordem 3, de modo que ordens maiores apenas elevam o custo sem ganho de qualidade.
