# Processamento Digital de Sinais - Filtros IIR

Este repositório contém a resolução do Projeto Final da disciplina **SEL0615 - Processamento Digital de Sinais (EESC-USP)**. O objetivo principal é o desenvolvimento de uma rotina de filtragem digital para mitigar o ruído de medição em sinais de corrente elétrica de uma planta industrial.

## 📋 Contexto do Problema

Numa determinada planta industrial, a introdução de cargas elétricas não-lineares (como retificadores, inversores e acionamentos de motores) provoca o surgimento de componentes harmónicas na corrente consumida. Um medidor localizado no ponto de conexão com a rede de média tensão regista a corrente sob uma taxa de amostragem de **128 amostras por ciclo**. 

Como a frequência fundamental da rede é $60\text{ Hz}$, a frequência de amostragem ($f_s$) do sistema é:
$$f_s = 128 \times 60 = 7680\text{ Hz}$$

O objetivo deste trabalho (específico para o **Grupo B9**) consiste em projetar e implementar **Filtros IIR (Infinite Impulse Response)** aplicados ao sinal ruidoso, de forma a **minimizar o RMSE** (Root Mean Square Error) em relação ao sinal ideal (sem ruído), avaliando o custo computacional associado.

## 🛠️ Tecnologias e Dependências

O projeto foi desenvolvido em **Python 3**, utilizando as seguintes bibliotecas computacionais:
* `numpy` - Para manipulação de matrizes e computação matemática.
* `scipy` - Especificamente o módulo `scipy.signal` para o projeto e aplicação dos filtros IIR.
* `pandas` - Para a leitura e processamento dos arquivos de dados `.csv`.
* `matplotlib` - Para a visualização gráfica dos sinais no domínio do tempo.

## 📂 Estrutura do Repositório

```text
├── sinais/
│   ├── sinal_9_ruido.csv      # Sinal com ruído de medição (Entrada)
│   └── sinal_9_semruido.csv   # Sinal original de referência (Validação)
├── filtro_iir_b9.py           # Script principal com o algoritmo de filtragem e otimização
└── README.md                  # Este ficheiro de documentação
```

## Como Executar

```bash
pip install -r requirements.txt
python filtroIir.py
```