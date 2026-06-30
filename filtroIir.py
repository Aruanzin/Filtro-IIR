"""
SEL0615 - Processamento Digital de Sinais (EESC-USP)
Projeto Final - Grupo B9: Filtros IIR

Rotina de filtragem digital IIR para atenuação do ruído de medição em um sinal
de corrente de planta industrial. O objetivo é minimizar o RMSE em relação ao
sinal de referência (sem ruído), avaliando o custo computacional associado.

Características desta implementação
----------------------------------
- Projeto via Seções de Segunda Ordem (SOS) -> estabilidade numérica superior
  à forma direta (b, a), especialmente em ordens altas e cortes baixos.
- Quatro aproximações: Butterworth, Chebyshev I, Chebyshev II e Elíptica.
- Comparação entre filtragem de FASE ZERO (sosfiltfilt) e CAUSAL (sosfilt),
  esta última realizável em tempo real.
- Busca em grade ampla (tipo x ordem x corte x ondulação/atenuação x método)
  com verificação de estabilidade (polos dentro do círculo unitário).
- Métricas: RMSE, melhoria de SNR (dB) e custo computacional (mediana do
  pipeline projeto+filtragem, robusta a jitter do SO).
- Análise no domínio da frequência (resposta freqz e espectro FFT) e no plano Z
  (diagrama de polos e zeros).
- Caracterização do sinal: detecção do instante de chaveamento das cargas não
  lineares (degrau da envoltória RMS) e análise harmônica/THD na janela
  pós-chaveamento (9 ciclos -> cada harmônica sobre um bin exato da FFT).
- Execução "zero-config": `python filtroIir.py` roda tudo e salva as figuras.

Uso
---
    python filtroIir.py            # análise completa + todas as figuras
    python filtroIir.py --rapido   # apenas a busca e o resumo no terminal
    python filtroIir.py --fs 7680  # sobrescreve a frequência de amostragem
"""

import argparse
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal

# Estilo visual consistente e de alto contraste, adequado para o relatório.
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 8,
    'figure.titlesize': 14,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

# Paleta por aproximação (reaproveitada em todas as figuras).
CORES = {
    'butter': '#2196F3',   # azul
    'cheby1': '#E91E63',   # rosa
    'cheby2': '#4CAF50',   # verde
    'ellip':  '#FF9800',   # laranja
}
NOMES = {
    'butter': 'Butterworth',
    'cheby1': 'Chebyshev I',
    'cheby2': 'Chebyshev II',
    'ellip':  'Elíptico',
}


# --------------------------------------------------------------------------- #
#  Projeto e aplicação dos filtros
# --------------------------------------------------------------------------- #
def projetar_sos(tipo, ordem, Wn, rp=0.5, rs=40.0):
    """
    Projeta um filtro IIR passa-baixas e retorna suas Seções de Segunda Ordem.

    Parâmetros:
    - tipo  : 'butter' | 'cheby1' | 'cheby2' | 'ellip'
    - ordem : ordem do filtro
    - Wn    : frequência de corte normalizada (0 < Wn < 1, sendo 1 = Nyquist)
    - rp    : ondulação máxima na banda de passagem em dB (cheby1, ellip)
    - rs    : atenuação mínima na banda de rejeição em dB (cheby2, ellip)

    A forma SOS (cascata de biquads) é numericamente muito mais estável que a
    forma direta (b, a) para ordens elevadas, evitando o "blow-up" de RMSE.
    """
    if tipo == 'butter':
        return signal.butter(ordem, Wn, btype='low', output='sos')
    if tipo == 'cheby1':
        return signal.cheby1(ordem, rp, Wn, btype='low', output='sos')
    if tipo == 'cheby2':
        return signal.cheby2(ordem, rs, Wn, btype='low', output='sos')
    if tipo == 'ellip':
        return signal.ellip(ordem, rp, rs, Wn, btype='low', output='sos')
    raise ValueError(f"Tipo de filtro não reconhecido: '{tipo}'")


def aplicar_filtro_iir(sinal_ruido, fs, f_corte, ordem, tipo='butter',
                       rp=0.5, rs=40.0, fase_zero=True):
    """
    Projeta e aplica um filtro IIR passa-baixas ao sinal.

    Retorna (sinal_filtrado, sos):
    - fase_zero=True  -> sosfiltfilt: filtragem de fase zero (offline).
    - fase_zero=False -> sosfilt:     filtragem causal (tempo real).
    """
    nyq = 0.5 * fs
    Wn = f_corte / nyq                       # frequência normalizada
    sos = projetar_sos(tipo, ordem, Wn, rp, rs)
    if fase_zero:
        filtrado = signal.sosfiltfilt(sos, sinal_ruido)
    else:
        filtrado = signal.sosfilt(sos, sinal_ruido)
    return filtrado, sos


def eh_estavel(sos):
    """Retorna True se todos os polos estiverem dentro do círculo unitário."""
    _, polos, _ = signal.sos2zpk(sos)
    return np.all(np.abs(polos) < 1.0)


# --------------------------------------------------------------------------- #
#  Métricas de qualidade e de custo
# --------------------------------------------------------------------------- #
def calcular_rmse(sinal_original, sinal_filtrado):
    """RMSE entre o sinal de referência e o sinal filtrado."""
    return np.sqrt(np.mean((sinal_original - sinal_filtrado) ** 2))


def melhoria_snr_db(referencia, ruidoso, filtrado):
    """
    Ganho de SNR (em dB) obtido pela filtragem.

    SNR = 10*log10( energia_do_sinal / energia_do_erro ). A melhoria é a
    diferença entre o SNR de saída (após o filtro) e o de entrada (sinal bruto).
    """
    pot_sinal = np.sum(referencia ** 2)
    erro_in = np.sum((ruidoso - referencia) ** 2)
    erro_out = np.sum((filtrado - referencia) ** 2)
    snr_in = 10.0 * np.log10(pot_sinal / erro_in)
    snr_out = 10.0 * np.log10(pot_sinal / erro_out)
    return snr_out - snr_in


def medir_custo_computacional(sinal_ruido, fs, f_corte, ordem, tipo,
                              rp=0.5, rs=40.0, fase_zero=True, n_repeticoes=50):
    """
    Mede o custo do pipeline completo (projeto dos coeficientes + filtragem).

    Cronometrar o pipeline inteiro reflete o custo prático de uso do filtro,
    que cresce com a ordem. Usa-se a MEDIANA de n_repeticoes execuções para
    reduzir a influência do escalonamento do sistema operacional. Retorna ms.
    """
    tempos = []
    for _ in range(n_repeticoes):
        t0 = time.perf_counter()
        aplicar_filtro_iir(sinal_ruido, fs, f_corte, ordem, tipo, rp, rs, fase_zero)
        tempos.append(time.perf_counter() - t0)
    return float(np.median(tempos)) * 1000.0  # ms


# --------------------------------------------------------------------------- #
#  Caracterização do sinal: chaveamento e conteúdo harmônico
# --------------------------------------------------------------------------- #
def detectar_instante_chaveamento(sinal, fs):
    """
    Localiza o instante de inserção das cargas não lineares.

    A entrada das cargas não lineares eleva abruptamente a envoltória da
    corrente. Avaliando o valor RMS ciclo a ciclo da fundamental (janela de
    ``fs/60`` amostras), a maior variação entre ciclos consecutivos marca o
    chaveamento. Retorna ``(indice, tempo_s)``.
    """
    ciclo = int(round(fs / 60.0))
    n_blocos = len(sinal) // ciclo
    rms = np.array([
        np.sqrt(np.mean(sinal[i * ciclo:(i + 1) * ciclo] ** 2))
        for i in range(n_blocos)
    ])
    bloco = int(np.argmax(np.abs(np.diff(rms)))) + 1
    idx = bloco * ciclo
    return idx, idx / fs


def analise_harmonica(sinal, fs, n_ciclos=9, hmax=13):
    """
    Conteúdo harmônico e THD em regime permanente pós-chaveamento.

    Mede os últimos ``n_ciclos`` ciclos completos da fundamental do registro
    (região estacionária após o chaveamento, evitando o transitório de
    inserção). Como a janela cobre um número inteiro de períodos de 60 Hz, cada
    harmônica recai sobre um bin exato da FFT, eliminando o espalhamento
    espectral; as amplitudes de pico saem de ``2|X[k]|/N``. Retorna
    ``(DataFrame, thd_pct)``, com ``THD = sqrt(sum_{h>=2} A_h^2) / A_1``.
    """
    ciclo = int(round(fs / 60.0))
    N = n_ciclos * ciclo
    inicio = max(0, len(sinal) - N)                  # últimos n_ciclos completos
    janela = sinal[inicio:inicio + N]
    X = np.abs(np.fft.rfft(janela)) * 2.0 / N        # amplitude de pico por bin
    A1 = X[n_ciclos]                                 # bin exato da fundamental
    linhas, soma_h2 = [], 0.0
    for h in range(1, hmax + 1):
        bin_h = h * n_ciclos
        if bin_h >= len(X):
            break
        Ah = float(X[bin_h])
        if h >= 2:
            soma_h2 += Ah ** 2
        linhas.append({'harmonica': h, 'freq_hz': h * 60,
                       'amplitude': Ah, 'pct_fundamental': 100.0 * Ah / A1})
    thd = 100.0 * np.sqrt(soma_h2) / A1
    return pd.DataFrame(linhas), float(thd)


# --------------------------------------------------------------------------- #
#  Buscas de parâmetros
# --------------------------------------------------------------------------- #
def busca_em_grade(sinal_sem_ruido, sinal_com_ruido, fs,
                   ordens=range(1, 9),
                   frequencias_corte=np.arange(300, 3301, 100),
                   tipos=('butter', 'cheby1', 'cheby2', 'ellip'),
                   metodos=(True, False),
                   rps=(0.1, 0.5, 1.0),
                   rss=(30.0, 40.0, 50.0, 60.0)):
    """
    Varre tipo x ordem x corte x ondulação/atenuação x método (fase zero /
    causal) e devolve um DataFrame com o RMSE de cada configuração estável.

    A ondulação de passagem ``rp`` só é varrida nas aproximações que a possuem
    (Chebyshev I e elíptica) e a atenuação de rejeição ``rs`` apenas onde ela é
    parâmetro de projeto (Chebyshev II e elíptica); para as demais o valor é
    irrelevante e mantém-se fixo, evitando configurações redundantes.

    Obs.: o "janelamento" pedido no enunciado é técnica de filtros FIR e não se
    aplica a IIR; o par (rp, rs) varrido aqui é o grau de liberdade análogo que
    molda o gabarito da aproximação.
    """
    linhas = []
    for tipo in tipos:
        lista_rp = rps if tipo in ('cheby1', 'ellip') else (0.5,)
        lista_rs = rss if tipo in ('cheby2', 'ellip') else (40.0,)
        for ordem in ordens:
            for fc in frequencias_corte:
                for fase_zero in metodos:
                    for rp in lista_rp:
                        for rs in lista_rs:
                            try:
                                filtrado, sos = aplicar_filtro_iir(
                                    sinal_com_ruido, fs, fc, ordem, tipo,
                                    rp=rp, rs=rs, fase_zero=fase_zero)
                                if not eh_estavel(sos):
                                    continue              # descarta instáveis
                                rmse = calcular_rmse(sinal_sem_ruido, filtrado)
                                if not np.isfinite(rmse):
                                    continue
                                linhas.append({
                                    'tipo': tipo, 'ordem': ordem, 'fc': int(fc),
                                    'fase_zero': fase_zero, 'rp': rp, 'rs': rs,
                                    'rmse': rmse,
                                })
                            except Exception:
                                continue
    return pd.DataFrame(linhas)


def analise_custo_vs_ordem(sinal_sem_ruido, sinal_com_ruido, fs,
                           tipo='butter', fc=1100.0, ordens=range(1, 9),
                           fase_zero=True, rp=0.5, rs=40.0):
    """
    Varre a ordem com tipo/corte fixos, reportando RMSE, custo computacional e
    o módulo do polo dominante. Subsidia a escolha da ordem considerando o custo.
    """
    linhas = []
    for ordem in ordens:
        try:
            filtrado, sos = aplicar_filtro_iir(
                sinal_com_ruido, fs, fc, ordem, tipo, rp=rp, rs=rs,
                fase_zero=fase_zero)
            if not eh_estavel(sos):
                continue
            rmse = calcular_rmse(sinal_sem_ruido, filtrado)
            _, polos, _ = signal.sos2zpk(sos)
            pmax = float(np.max(np.abs(polos)))
            custo = medir_custo_computacional(
                sinal_com_ruido, fs, fc, ordem, tipo, rp=rp, rs=rs,
                fase_zero=fase_zero)
            linhas.append({'ordem': ordem, 'rmse': rmse, 'tempo_ms': custo,
                           'pmax': pmax})
        except Exception:
            continue
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------- #
#  Figuras
# --------------------------------------------------------------------------- #
def plot_sinal_bruto(sem_ruido, com_ruido, fs, idx_sw,
                     arquivo='sinal_bruto_b9.pdf'):
    """
    Sinal de corrente completo (todas as amostras): ruidoso versus referência,
    com o instante de inserção das cargas não lineares destacado. Salvo em PDF
    vetorial. Subsidia a seção de caracterização/análise do sinal.
    """
    n = len(sem_ruido)
    t_ms = np.arange(n) / fs * 1000.0
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(t_ms, com_ruido, color='#C0392B', lw=0.7, alpha=0.55,
            label='Com ruído')
    ax.plot(t_ms, sem_ruido, color='black', lw=1.2,
            label='Referência (sem ruído)')
    ax.axvline(idx_sw / fs * 1000.0, color='#2C3E50', lw=1.3, linestyle='--',
               label='Inserção de cargas não lineares')
    ax.set_title("Sinal de Corrente Medido",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel("Tempo (ms)")
    ax.set_ylabel("Corrente (A)")
    ax.set_xlim(0, t_ms[-1])
    ax.legend(loc='upper left')
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(arquivo, bbox_inches='tight')
    plt.close(fig)
    print(f"  figura salva: {arquivo}")


def plot_comparativo_tempo(sem_ruido, com_ruido, melhores_por_tipo,
                           melhor_tipo, idx_sw=384, n_amostras=768,
                           arquivo='resultado_filtros_iir_b9.pdf'):
    """Painel por aproximação (registro inteiro) + erro residual das 4 sobrepostas.

    Os quatro painéis superiores mostram, para cada aproximação, com ruído /
    original / filtrado no registro inteiro. O painel inferior sobrepõe o erro
    residual (filtrado − referência) das quatro aproximações, cada uma em sua
    cor, para comparação direta da grandeza minimizada pelo RMSE.
    """
    tipos = ['butter', 'cheby1', 'cheby2', 'ellip']

    fig = plt.figure(figsize=(16, 13))
    fig.suptitle("Filtragem IIR — Melhor Configuração por Aproximação (Fase Zero)",
                 fontsize=14, fontweight='bold', y=0.99)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.22)
    posicoes = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for idx, tipo in enumerate(tipos):
        info = melhores_por_tipo[tipo]
        ax = fig.add_subplot(gs[posicoes[idx]])
        ax.plot(com_ruido[:n_amostras], alpha=0.55, color='#6E6E6E', lw=1.0,
                label='Com ruído')
        ax.plot(sem_ruido[:n_amostras], color='black', lw=1.5, label='Original')
        ax.plot(info['sinal'][:n_amostras], color=CORES[tipo], lw=2.0,
                linestyle='--', label=f"Filtrado ({NOMES[tipo]})")
        estrela = " ★" if tipo == melhor_tipo else ""
        ax.set_title(f"{NOMES[tipo]}{estrela} — Ordem {info['ordem']} | fc = {info['fc']} Hz\n"
                     f"RMSE = {info['rmse']:.4f} | ΔSNR = {info['snr']:.1f} dB",
                     fontsize=10)
        ax.set_xlabel("Amostras")
        ax.set_ylabel("Corrente (A)")
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.4)

    # Painel inferior: erro residual das 4 aproximações sobreposto, ampliado na
    # janela do chaveamento (onde as curvas mais se distinguem; fora dela o erro
    # é pequeno e quase idêntico). Zoom em x preserva o pico (não corta em y).
    rw0, rw1 = 300, 460
    xr = np.arange(rw0, rw1)
    ax_res = fig.add_subplot(gs[2, :])
    for tipo in tipos:
        info = melhores_por_tipo[tipo]
        residuo = info['sinal'][:n_amostras] - sem_ruido[:n_amostras]
        destaque = tipo == melhor_tipo
        ax_res.plot(xr, residuo[rw0:rw1], color=CORES[tipo],
                    lw=2.2 if destaque else 1.4, alpha=0.95 if destaque else 0.8,
                    label=f"{NOMES[tipo]}{' ★' if destaque else ''} "
                          f"(RMSE={info['rmse']:.4f})")
    ax_res.axhline(0, color='black', lw=0.6)
    ax_res.axvline(idx_sw, color='0.4', ls=':', lw=1.2,
                   label=f"Chaveamento (amostra {idx_sw})")
    ax_res.set_xlim(rw0, rw1)
    ax_res.set_title(f"Erro residual (filtrado − referência) — zoom no transitório "
                     f"do chaveamento (amostras {rw0}–{rw1})",
                     fontsize=11, fontweight='bold')
    ax_res.set_xlabel("Amostras")
    ax_res.set_ylabel("Erro (A)")
    ax_res.legend(loc='upper right', ncol=3, fontsize=9)
    ax_res.grid(True, alpha=0.4)

    fig.savefig(arquivo, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  figura salva: {arquivo}")


def plot_melhor_global(sem_ruido, com_ruido, info, tipo, n_amostras=768,
                       arquivo='melhor_filtro_iir_b9.pdf'):
    """Destaque do melhor filtro global no domínio do tempo."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(com_ruido[:n_amostras], alpha=0.4, color='gray', lw=1, label='Com ruído')
    ax.plot(sem_ruido[:n_amostras], color='black', lw=2, label='Original (referência)')
    ax.plot(info['sinal'][:n_amostras], color=CORES[tipo], lw=2.5, linestyle='--',
            label=(f"Melhor global: {NOMES[tipo]} | Ordem {info['ordem']} | "
                   f"fc = {info['fc']} Hz | RMSE = {info['rmse']:.4f}"))
    ax.set_title(f"★ Melhor Filtro IIR Global — {NOMES[tipo]} "
                 f"(Ordem {info['ordem']}, fc = {info['fc']} Hz)\n"
                 f"RMSE = {info['rmse']:.4f} | ΔSNR = {info['snr']:.1f} dB",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel("Amostras")
    ax.set_ylabel("Corrente (A)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(arquivo, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  figura salva: {arquivo}")


def plot_resposta_frequencia(melhores_por_tipo, fs,
                             arquivo='resposta_frequencia_b9.pdf'):
    """Resposta em magnitude (freqz) dos melhores filtros de cada tipo."""
    fig, ax = plt.subplots(figsize=(11, 5))
    for tipo in ['butter', 'cheby1', 'cheby2', 'ellip']:
        info = melhores_por_tipo[tipo]
        w, h = signal.sosfreqz(info['sos'], worN=4096, fs=fs)
        ax.plot(w, 20 * np.log10(np.abs(h) + 1e-12), color=CORES[tipo], lw=1.6,
                label=f"{NOMES[tipo]} (Ordem {info['ordem']}, fc={info['fc']} Hz)")
        ax.axvline(info['fc'], color=CORES[tipo], lw=0.8, alpha=0.3, linestyle=':')
    ax.set_title("Resposta em Frequência dos Melhores Filtros",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel("Frequência (Hz)")
    ax.set_ylabel("Ganho (dB)")
    ax.set_xlim(0, fs / 2)
    ax.set_ylim(-100, 5)
    ax.legend(loc='lower left')
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(arquivo, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  figura salva: {arquivo}")


def plot_espectro_fft(sem_ruido, com_ruido, info_global, fs,
                      arquivo='espectro_fft_b9.pdf'):
    """Espectro de magnitude (FFT) antes e depois da filtragem."""
    n = len(sem_ruido)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    esp_ruido = np.abs(np.fft.rfft(com_ruido)) / n
    esp_limpo = np.abs(np.fft.rfft(sem_ruido)) / n
    esp_filtrado = np.abs(np.fft.rfft(info_global['sinal'])) / n

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.semilogy(freqs, esp_ruido, color='#FF4B4B', alpha=0.4, lw=0.9,
                label='Espectro ruidoso')
    ax.semilogy(freqs, esp_limpo, color='black', lw=1.6, label='Espectro referência')
    ax.semilogy(freqs, esp_filtrado, color=CORES[info_global['tipo']], lw=1.4,
                linestyle='--', label=f"Filtrado ({NOMES[info_global['tipo']]})")
    ax.axvline(info_global['fc'], color='#00A86B', lw=1.0, linestyle=':',
               label=f"fc = {info_global['fc']} Hz")
    ax.set_title("Espectro de Frequência — Antes e Depois da Filtragem",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel("Frequência (Hz)")
    ax.set_ylabel("Magnitude (norm.)")
    ax.set_xlim(0, fs / 2)
    ax.legend(loc='upper right')
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(arquivo, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  figura salva: {arquivo}")


def plot_custo_vs_ordem(df_custo, tipo, fc, arquivo='custo_vs_ordem_b9.pdf'):
    """Eixo duplo: RMSE e custo computacional em função da ordem."""
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(df_custo['ordem'], df_custo['rmse'], marker='o', color='#004B87',
             lw=1.6, label='RMSE')
    # Escala log: a ordem 1 é um outlier (RMSE ~ 18) que, em escala linear,
    # achata as ordens 3-8 e esconde o mínimo na ordem 3 e a subida posterior.
    ax1.set_yscale('log')
    ordem_min = int(df_custo.loc[df_custo['rmse'].idxmin(), 'ordem'])
    rmse_min = float(df_custo['rmse'].min())
    ax1.annotate(f"mínimo (ordem {ordem_min})", xy=(ordem_min, rmse_min),
                 xytext=(ordem_min + 1.2, rmse_min * 2.2), fontsize=8,
                 color='#004B87',
                 arrowprops=dict(arrowstyle='->', color='#004B87', lw=1.0))
    ax1.set_xlabel("Ordem do Filtro")
    ax1.set_ylabel("RMSE (escala log)", color='#004B87')
    ax1.tick_params(axis='y', labelcolor='#004B87')
    ax1.grid(True, which='both')
    ax2 = ax1.twinx()
    ax2.plot(df_custo['ordem'], df_custo['tempo_ms'], marker='s', color='#FF4B4B',
             linestyle='--', lw=1.6, label='Custo computacional')
    ax2.set_ylabel("Tempo de execução (ms)", color='#FF4B4B')
    ax2.tick_params(axis='y', labelcolor='#FF4B4B')
    ax1.set_title(f"Custo Computacional vs. Ordem "
                  f"({NOMES[tipo]} Fase Zero, fc = {int(fc)} Hz)",
                  fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(arquivo, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  figura salva: {arquivo}")


def plot_polos_zeros(info_global, arquivo='polos_zeros_b9.pdf'):
    """Diagrama de polos e zeros no plano Z (estabilidade do melhor filtro)."""
    zeros, polos, _ = signal.sos2zpk(info_global['sos'])
    # O empacotamento SOS de uma ordem ímpar insere um par polo/zero trivial na
    # origem (z=0) que se cancela; remove-se para exibir os polos/zeros reais.
    zeros = zeros[np.abs(zeros) > 1e-8]
    polos = polos[np.abs(polos) > 1e-8]
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    teta = np.linspace(0, 2 * np.pi, 512)
    ax.plot(np.cos(teta), np.sin(teta), color='gray', lw=1.0, linestyle='--',
            label='Círculo unitário')
    ax.scatter(np.real(zeros), np.imag(zeros), s=70, facecolors='none',
               edgecolors='#2196F3', linewidths=1.6, label='Zeros')
    ax.scatter(np.real(polos), np.imag(polos), s=70, marker='x',
               color='#E91E63', linewidths=1.8, label='Polos')
    raio_max = np.max(np.abs(polos)) if len(polos) else 0.0
    ax.set_title(f"Polos e Zeros — {NOMES[info_global['tipo']]} "
                 f"(Ordem {info_global['ordem']})\n"
                 f"Polo de maior módulo: {raio_max:.4f} "
                 f"({'estável' if raio_max < 1 else 'INSTÁVEL'})",
                 fontsize=11, fontweight='bold')
    ax.set_xlabel("Parte Real")
    ax.set_ylabel("Parte Imaginária")
    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(0, color='black', lw=0.5)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(arquivo, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  figura salva: {arquivo}")


def plot_causal_vs_fase_zero(sem_ruido, com_ruido, fs, df_busca, n_amostras=768,
                             arquivo='causal_vs_fasezero_b9.pdf'):
    """
    Compara o melhor filtro de fase zero com o melhor filtro causal (realizável
    em tempo real). Evidencia o atraso de fase introduzido pela filtragem causal.
    """
    melhor_fz = df_busca[df_busca['fase_zero']].sort_values('rmse').iloc[0]
    melhor_ca = df_busca[~df_busca['fase_zero']].sort_values('rmse').iloc[0]

    filt_fz, _ = aplicar_filtro_iir(com_ruido, fs, melhor_fz['fc'],
                                    int(melhor_fz['ordem']), melhor_fz['tipo'],
                                    rp=float(melhor_fz['rp']),
                                    rs=float(melhor_fz['rs']), fase_zero=True)
    filt_ca, _ = aplicar_filtro_iir(com_ruido, fs, melhor_ca['fc'],
                                    int(melhor_ca['ordem']), melhor_ca['tipo'],
                                    rp=float(melhor_ca['rp']),
                                    rs=float(melhor_ca['rs']), fase_zero=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(com_ruido[:n_amostras], alpha=0.3, color='gray', lw=1, label='Com ruído')
    ax.plot(sem_ruido[:n_amostras], color='black', lw=2, label='Original (referência)')
    ax.plot(filt_fz[:n_amostras], color='#004B87', lw=1.8, linestyle='--',
            label=(f"Fase zero: {NOMES[melhor_fz['tipo']]} O{int(melhor_fz['ordem'])} "
                   f"fc={melhor_fz['fc']} Hz | RMSE={melhor_fz['rmse']:.4f}"))
    ax.plot(filt_ca[:n_amostras], color='#E69F00', lw=1.8, linestyle='-.',
            label=(f"Causal: {NOMES[melhor_ca['tipo']]} O{int(melhor_ca['ordem'])} "
                   f"fc={melhor_ca['fc']} Hz | RMSE={melhor_ca['rmse']:.4f}"))
    ax.set_title("Filtragem Causal (tempo real) vs. Fase Zero (offline)",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel("Amostras")
    ax.set_ylabel("Corrente (A)")
    ax.legend(loc='upper right')
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(arquivo, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  figura salva: {arquivo}")
    return melhor_fz, melhor_ca


# --------------------------------------------------------------------------- #
#  Carregamento dos sinais (com fallback sintético)
# --------------------------------------------------------------------------- #
def carregar_sinais(fs, caminho_limpo='sinais/sinal_9_semruido.csv',
                    caminho_ruido='sinais/sinal_9_ruido.csv'):
    """Carrega os CSVs; se ausentes, gera um sinal sintético de demonstração."""
    try:
        sem = pd.read_csv(caminho_limpo, header=None).values.flatten().astype(float)
        com = pd.read_csv(caminho_ruido, header=None).values.flatten().astype(float)
        print("Sinais carregados dos arquivos CSV.")
        return sem, com
    except FileNotFoundError:
        print("Arquivos CSV não encontrados — gerando dados sintéticos para demonstração.")
        t = np.arange(0, 0.2, 1 / fs)
        sem = (10 * np.sin(2 * np.pi * 60 * t) +
               3 * np.sin(2 * np.pi * 180 * t) +
               1.5 * np.sin(2 * np.pi * 300 * t))
        ruido = np.random.default_rng(9).normal(0, 1.5, len(t))
        return sem, sem + ruido


# --------------------------------------------------------------------------- #
#  Programa principal
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Filtros IIR — Projeto Final PDS, Grupo B9")
    parser.add_argument('--fs', type=int, default=7680,
                        help="Frequência de amostragem em Hz (padrão: 7680)")
    parser.add_argument('--rapido', action='store_true',
                        help="Apenas busca e resumo no terminal, sem figuras")
    args = parser.parse_args()

    fs = args.fs
    sem_ruido, com_ruido = carregar_sinais(fs)

    rmse_inicial = calcular_rmse(sem_ruido, com_ruido)
    print(f"RMSE inicial (sem filtragem): {rmse_inicial:.4f}\n")

    # ---- Caracterização do sinal (chaveamento e harmônicos) ------------- #
    idx_sw, t_sw = detectar_instante_chaveamento(sem_ruido, fs)
    df_harm, thd = analise_harmonica(sem_ruido, fs)
    print(f"Instante de chaveamento das cargas não lineares: "
          f"amostra {idx_sw} (t = {t_sw * 1000:.1f} ms)")
    print(f"Conteúdo harmônico pós-chaveamento (THD = {thd:.1f}%):")
    print(df_harm[df_harm['pct_fundamental'] >= 1.0].to_string(
        index=False, formatters={'amplitude': '{:.2f}'.format,
                                  'pct_fundamental': '{:.2f}'.format}))
    print()

    print("Iniciando busca em grade (tipo x ordem x corte x método)...")
    df = busca_em_grade(sem_ruido, com_ruido, fs)
    print(f"Configurações estáveis avaliadas: {len(df)}\n")

    # Melhor configuração de fase zero por tipo de aproximação.
    melhores_por_tipo = {}
    for tipo in ['butter', 'cheby1', 'cheby2', 'ellip']:
        sub = df[(df['tipo'] == tipo) & (df['fase_zero'])].sort_values('rmse')
        melhor = sub.iloc[0]
        rp_b, rs_b = float(melhor['rp']), float(melhor['rs'])
        filtrado, sos = aplicar_filtro_iir(
            com_ruido, fs, melhor['fc'], int(melhor['ordem']), tipo,
            rp=rp_b, rs=rs_b, fase_zero=True)
        custo = medir_custo_computacional(
            com_ruido, fs, melhor['fc'], int(melhor['ordem']), tipo,
            rp=rp_b, rs=rs_b, fase_zero=True)
        melhores_por_tipo[tipo] = {
            'tipo': tipo, 'ordem': int(melhor['ordem']), 'fc': int(melhor['fc']),
            'rp': rp_b, 'rs': rs_b, 'rmse': float(melhor['rmse']),
            'snr': melhoria_snr_db(sem_ruido, com_ruido, filtrado),
            'tempo_ms': custo, 'sinal': filtrado, 'sos': sos,
        }

    melhor_tipo = min(melhores_por_tipo,
                      key=lambda t: melhores_por_tipo[t]['rmse'])
    info_global = melhores_por_tipo[melhor_tipo]

    # ---- Resumo no terminal --------------------------------------------- #
    print("=" * 64)
    print("  MELHOR CONFIGURAÇÃO (FASE ZERO) POR TIPO DE APROXIMAÇÃO")
    print("=" * 64)
    for tipo in ['butter', 'cheby1', 'cheby2', 'ellip']:
        i = melhores_por_tipo[tipo]
        print(f"[{NOMES[tipo]:14s}] Ordem {i['ordem']} | fc {i['fc']:5d} Hz | "
              f"rp {i['rp']:.1f} dB | rs {i['rs']:.0f} dB | "
              f"RMSE {i['rmse']:.4f} | ganho SNR {i['snr']:5.1f} dB | {i['tempo_ms']:.3f} ms")

    print("-" * 64)
    print(f"  MELHOR GLOBAL -> {NOMES[melhor_tipo]}, Ordem {info_global['ordem']}, "
          f"fc = {info_global['fc']} Hz | RMSE = {info_global['rmse']:.4f}")
    print(f"  Redução de RMSE: {(1 - info_global['rmse']/rmse_inicial)*100:.1f}%  |  "
          f"Melhoria de SNR: {info_global['snr']:.1f} dB")
    print("=" * 64)

    # Top causais (realizáveis em tempo real).
    top_causal = df[~df['fase_zero']].sort_values('rmse').head(3)
    print("\n  TOP 3 FILTROS CAUSAIS (realizáveis em tempo real):")
    for _, r in top_causal.iterrows():
        print(f"    {NOMES[r['tipo']]:14s} Ordem {int(r['ordem'])} | "
              f"fc {int(r['fc']):5d} Hz | RMSE {r['rmse']:.4f}")

    if args.rapido:
        return

    # ---- Figuras -------------------------------------------------------- #
    print("\nGerando figuras...")
    plot_sinal_bruto(sem_ruido, com_ruido, fs, idx_sw)
    plot_comparativo_tempo(sem_ruido, com_ruido, melhores_por_tipo, melhor_tipo,
                           idx_sw=idx_sw)
    plot_melhor_global(sem_ruido, com_ruido, info_global, melhor_tipo)
    plot_resposta_frequencia(melhores_por_tipo, fs)
    plot_espectro_fft(sem_ruido, com_ruido, info_global, fs)
    plot_polos_zeros(info_global)
    plot_causal_vs_fase_zero(sem_ruido, com_ruido, fs, df)

    df_custo = analise_custo_vs_ordem(
        sem_ruido, com_ruido, fs, tipo=melhor_tipo, fc=info_global['fc'],
        rp=info_global['rp'], rs=info_global['rs'])
    if not df_custo.empty:
        print("\n  CUSTO COMPUTACIONAL vs. ORDEM "
              f"({NOMES[melhor_tipo]}, fc={info_global['fc']} Hz, "
              f"rs={info_global['rs']:.0f} dB, fase zero):")
        print(df_custo.to_string(index=False, formatters={
            'rmse': '{:.3f}'.format, 'tempo_ms': '{:.2f}'.format,
            'pmax': '{:.3f}'.format}))
        plot_custo_vs_ordem(df_custo, melhor_tipo, info_global['fc'])

    # Exporta as tabelas para reuso/relatório.
    df.sort_values('rmse').to_csv('resultados_busca_b9.csv', index=False)
    print("  tabela salva: resultados_busca_b9.csv")
    df_harm.to_csv('harmonicos_b9.csv', index=False)
    print("  tabela salva: harmonicos_b9.csv")
    print("\nConcluído.")


if __name__ == "__main__":
    main()
