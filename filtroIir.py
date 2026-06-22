import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal
import time

def aplicar_filtro_iir(sinal_ruido, fs, f_corte, ordem, tipo_filtro='butter'):
    """
    Projeta um filtro IIR passa-baixas e o aplica ao sinal com ruído.

    Parâmetros:
    - sinal_ruido  : Array com o sinal ruidoso de entrada (qualquer tamanho).
    - fs           : Frequência de amostragem (Hz).
    - f_corte      : Frequência de corte desejada (Hz).
    - ordem        : Ordem do filtro IIR.
    - tipo_filtro  : 'butter' | 'cheby1' | 'cheby2' | 'ellip'

    Retorna:
    - sinal_filtrado: Array com o sinal após filtragem de fase zero (filtfilt).
    """
    nyq = 0.5 * fs
    Wn  = f_corte / nyq          # frequência normalizada (0 < Wn < 1)

    # Projeto dos coeficientes segundo a aproximação escolhida
    if tipo_filtro == 'butter':
        b, a = signal.butter(ordem, Wn, btype='low', analog=False)
    elif tipo_filtro == 'cheby1':
        # Ripple de 0,5 dB na banda de passagem — boa relação seletividade/ruído
        b, a = signal.cheby1(ordem, rp=0.5, Wn=Wn, btype='low', analog=False)
    elif tipo_filtro == 'cheby2':
        # Atenuação mínima de 40 dB na banda de rejeição
        b, a = signal.cheby2(ordem, rs=40, Wn=Wn, btype='low', analog=False)
    elif tipo_filtro == 'ellip':
        # Combina ripple de passagem (0,5 dB) e rejeição (40 dB) — maior seletividade
        b, a = signal.ellip(ordem, rp=0.5, rs=40, Wn=Wn, btype='low', analog=False)
    else:
        raise ValueError(f"Tipo de filtro não reconhecido: '{tipo_filtro}'")

    # filtfilt: filtragem de fase zero — ideal para processamento offline
    sinal_filtrado = signal.filtfilt(b, a, sinal_ruido)
    return sinal_filtrado


def calcular_rmse(sinal_original, sinal_filtrado):
    """Calcula o RMSE entre o sinal de referência e o sinal filtrado."""
    return np.sqrt(np.mean((sinal_original - sinal_filtrado) ** 2))


def medir_custo_computacional(sinal_ruido, fs, f_corte, ordem, tipo_filtro, n_repeticoes=20):
    """Mede o tempo médio de execução da filtragem (custo computacional)."""
    tempos = []
    for _ in range(n_repeticoes):
        t0 = time.perf_counter()
        aplicar_filtro_iir(sinal_ruido, fs, f_corte, ordem, tipo_filtro)
        tempos.append(time.perf_counter() - t0)
    return np.mean(tempos) * 1000  # retorna em ms


if __name__ == "__main__":

    fs = 7680
    try:
        df_sem_ruido = pd.read_csv('sinais/sinal_9_semruido.csv', header=None)
        df_com_ruido = pd.read_csv('sinais/sinal_9_ruido.csv',    header=None)
        sinal_sem_ruido = df_sem_ruido.values.flatten().astype(float)
        sinal_com_ruido = df_com_ruido.values.flatten().astype(float)
        print("Sinais carregados dos arquivos CSV.")
    except FileNotFoundError:
        print("Arquivos CSV não encontrados — gerando dados sintéticos para demonstração.")
        t = np.arange(0, 0.5, 1 / fs)
        sinal_sem_ruido = (10 * np.sin(2 * np.pi * 60  * t) +
                            3 * np.sin(2 * np.pi * 180 * t))
        sinal_com_ruido = sinal_sem_ruido + np.random.normal(0, 1.5, len(t))

    # --- Grids de varredura ---
    # Ordens baixas (2–6) equilibram seletividade e estabilidade numérica
    ordens = [2, 3, 4, 5, 6]
    # Frequências de corte cobrindo harmônicos típicos de sistemas de 60 Hz
    frequencias_corte  = [600, 900, 1200, 1500, 1800]
    tipos_aproximacao  = ['butter', 'cheby1', 'cheby2', 'ellip']

    melhores_por_tipo = {
        tipo: {'rmse': float('inf'), 'ordem': None, 'fc': None, 'sinal': None}
        for tipo in tipos_aproximacao
    }

    melhor_global_rmse   = float('inf')
    melhores_parametros  = {}

    print("\nIniciando varredura de parâmetros...")

    for tipo in tipos_aproximacao:
        for ordem in ordens:
            for fc in frequencias_corte:
                sinal_temp = aplicar_filtro_iir(sinal_com_ruido, fs, fc, ordem, tipo)
                rmse_temp  = calcular_rmse(sinal_sem_ruido, sinal_temp)

                # Melhor dentro do tipo
                if rmse_temp < melhores_por_tipo[tipo]['rmse']:
                    melhores_por_tipo[tipo].update({
                        'rmse' : rmse_temp,
                        'ordem': ordem,
                        'fc'   : fc,
                        'sinal': sinal_temp,
                    })

                # Melhor global
                if rmse_temp < melhor_global_rmse:
                    melhor_global_rmse  = rmse_temp
                    melhores_parametros = {'tipo': tipo, 'ordem': ordem, 'fc': fc}

    print("\n" + "=" * 50)
    print("  MELHOR CONFIGURAÇÃO POR TIPO DE APROXIMAÇÃO")
    print("=" * 50)
    for tipo, info in melhores_por_tipo.items():
        tempo_ms = medir_custo_computacional(
            sinal_com_ruido, fs, info['fc'], info['ordem'], tipo
        )
        print(f"\n[{tipo.upper():8s}]  Ordem: {info['ordem']}  |  fc: {info['fc']:5d} Hz  "
              f"|  RMSE: {info['rmse']:.4f}  |  Tempo médio: {tempo_ms:.3f} ms")

    print("\n" + "-" * 50)
    print(f"  MELHOR GLOBAL → {melhores_parametros['tipo'].upper()}, "
          f"Ordem {melhores_parametros['ordem']}, "
          f"fc = {melhores_parametros['fc']} Hz  |  RMSE = {melhor_global_rmse:.4f}")
    print("=" * 50)

    n_amostras = 400          # janela de exibição
    cores = {
        'butter': '#2196F3',   # azul
        'cheby1': '#E91E63',   # rosa
        'cheby2': '#4CAF50',   # verde
        'ellip' : '#FF9800',   # laranja
    }

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle("Filtragem IIR — Melhor Configuração por Aproximação\n(Grupo B9)",
                 fontsize=14, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35)

    posicoes = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for idx, tipo in enumerate(tipos_aproximacao):
        info = melhores_por_tipo[tipo]
        ax   = fig.add_subplot(gs[posicoes[idx]])

        ax.plot(sinal_com_ruido[:n_amostras],
                alpha=0.35, color='gray', lw=1, label='Com ruído')
        ax.plot(sinal_sem_ruido[:n_amostras],
                color='black', lw=1.5, label='Original')
        ax.plot(info['sinal'][:n_amostras],
                color=cores[tipo], lw=2, linestyle='--',
                label=f"Filtrado ({tipo.upper()})")

        ax.set_title(
            f"{tipo.upper()} — Ordem {info['ordem']} | fc = {info['fc']} Hz\n"
            f"RMSE = {info['rmse']:.4f}",
            fontsize=10
        )
        ax.set_xlabel("Amostras")
        ax.set_ylabel("Amplitude")
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.4)

    ax_comp = fig.add_subplot(gs[2, :])

    ax_comp.plot(sinal_com_ruido[:n_amostras],
                 alpha=0.25, color='gray', lw=1, label='Com ruído')
    ax_comp.plot(sinal_sem_ruido[:n_amostras],
                 color='black', lw=2, label='Original (referência)')

    for tipo in tipos_aproximacao:
        info = melhores_por_tipo[tipo]
        rmse = info['rmse']
        label = (f"{tipo.upper()} | Ordem {info['ordem']} | "
                 f"fc={info['fc']} Hz | RMSE={rmse:.4f}"
                 + (" ★ MELHOR GLOBAL" if tipo == melhores_parametros['tipo'] else ""))
        ax_comp.plot(info['sinal'][:n_amostras],
                     color=cores[tipo], lw=1.8, linestyle='--', label=label)

    ax_comp.set_title("Comparativo — Todos os Tipos (melhor configuração de cada um)",
                      fontsize=11, fontweight='bold')
    ax_comp.set_xlabel("Amostras")
    ax_comp.set_ylabel("Amplitude")
    ax_comp.legend(fontsize=8, loc='upper right')
    ax_comp.grid(True, alpha=0.4)

    plt.savefig("resultado_filtros_iir_b9.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("\nGráfico salvo em: resultado_filtros_iir_b9.png")

    tipo_glob  = melhores_parametros['tipo']
    info_glob  = melhores_por_tipo[tipo_glob]
    cor_glob   = cores[tipo_glob]

    fig2, ax2 = plt.subplots(figsize=(12, 5))

    ax2.plot(sinal_com_ruido[:n_amostras],
             alpha=0.4, color='gray', lw=1, label='Com ruído')
    ax2.plot(sinal_sem_ruido[:n_amostras],
             color='black', lw=2, label='Original (referência)')
    ax2.plot(info_glob['sinal'][:n_amostras],
             color=cor_glob, lw=2.5, linestyle='--',
             label=f"Melhor global: {tipo_glob.upper()} | "
                   f"Ordem {info_glob['ordem']} | "
                   f"fc = {info_glob['fc']} Hz | "
                   f"RMSE = {info_glob['rmse']:.4f}")

    ax2.set_title(
        f"★ Melhor Filtro IIR Global — {tipo_glob.upper()}  "
        f"(Ordem {info_glob['ordem']}, fc = {info_glob['fc']} Hz)\n"
        f"RMSE = {info_glob['rmse']:.4f}  |  Grupo B9",
        fontsize=12, fontweight='bold'
    )
    ax2.set_xlabel("Amostras")
    ax2.set_ylabel("Amplitude")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.4)

    fig2.tight_layout()
    plt.savefig("melhor_filtro_iir_b9.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Gráfico do melhor global salvo em: melhor_filtro_iir_b9.png")