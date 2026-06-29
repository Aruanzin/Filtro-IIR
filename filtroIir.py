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
- Busca em grade ampla (tipo x ordem x corte x método) com verificação de
  estabilidade (polos dentro do círculo unitário).
- Métricas: RMSE, melhoria de SNR (dB) e custo computacional (mediana do
  pipeline projeto+filtragem, robusta a jitter do SO).
- Análise no domínio da frequência (resposta freqz e espectro FFT) e no plano Z
  (diagrama de polos e zeros).
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
#  Buscas de parâmetros
# --------------------------------------------------------------------------- #
def busca_em_grade(sinal_sem_ruido, sinal_com_ruido, fs,
                   ordens=range(1, 9),
                   frequencias_corte=np.arange(300, 3301, 100),
                   tipos=('butter', 'cheby1', 'cheby2', 'ellip'),
                   metodos=(True, False)):
    """
    Varre tipo x ordem x corte x método (fase zero / causal) e devolve um
    DataFrame com o RMSE de cada configuração estável encontrada.
    """
    linhas = []
    for tipo in tipos:
        for ordem in ordens:
            for fc in frequencias_corte:
                for fase_zero in metodos:
                    try:
                        filtrado, sos = aplicar_filtro_iir(
                            sinal_com_ruido, fs, fc, ordem, tipo,
                            fase_zero=fase_zero)
                        if not eh_estavel(sos):
                            continue                      # descarta instáveis
                        rmse = calcular_rmse(sinal_sem_ruido, filtrado)
                        if not np.isfinite(rmse):
                            continue
                        linhas.append({
                            'tipo': tipo, 'ordem': ordem, 'fc': int(fc),
                            'fase_zero': fase_zero, 'rmse': rmse,
                        })
                    except Exception:
                        continue
    return pd.DataFrame(linhas)


def analise_custo_vs_ordem(sinal_sem_ruido, sinal_com_ruido, fs,
                           tipo='butter', fc=1100.0, ordens=range(1, 9),
                           fase_zero=True):
    """
    Varre a ordem com tipo/corte fixos, reportando RMSE e custo computacional.
    Subsidia diretamente a escolha da ordem considerando o custo.
    """
    linhas = []
    for ordem in ordens:
        try:
            filtrado, sos = aplicar_filtro_iir(
                sinal_com_ruido, fs, fc, ordem, tipo, fase_zero=fase_zero)
            if not eh_estavel(sos):
                continue
            rmse = calcular_rmse(sinal_sem_ruido, filtrado)
            custo = medir_custo_computacional(
                sinal_com_ruido, fs, fc, ordem, tipo, fase_zero=fase_zero)
            linhas.append({'ordem': ordem, 'rmse': rmse, 'tempo_ms': custo})
        except Exception:
            continue
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------- #
#  Figuras
# --------------------------------------------------------------------------- #
def plot_comparativo_tempo(sem_ruido, com_ruido, melhores_por_tipo,
                           melhor_tipo, n_amostras=400,
                           arquivo='resultado_filtros_iir_b9.png'):
    """Composição 3x2: um subplot por aproximação + um comparativo geral."""
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle("Filtragem IIR — Melhor Configuração por Aproximação (Fase Zero)\n"
                 "(Grupo B9)", fontsize=14, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.30)
    posicoes = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for idx, tipo in enumerate(['butter', 'cheby1', 'cheby2', 'ellip']):
        info = melhores_por_tipo[tipo]
        ax = fig.add_subplot(gs[posicoes[idx]])
        ax.plot(com_ruido[:n_amostras], alpha=0.35, color='gray', lw=1,
                label='Com ruído')
        ax.plot(sem_ruido[:n_amostras], color='black', lw=1.5, label='Original')
        ax.plot(info['sinal'][:n_amostras], color=CORES[tipo], lw=2,
                linestyle='--', label=f"Filtrado ({NOMES[tipo]})")
        ax.set_title(f"{NOMES[tipo]} — Ordem {info['ordem']} | fc = {info['fc']} Hz\n"
                     f"RMSE = {info['rmse']:.4f} | ΔSNR = {info['snr']:.1f} dB",
                     fontsize=10)
        ax.set_xlabel("Amostras")
        ax.set_ylabel("Corrente (A)")
        ax.legend(loc='upper right')
        ax.grid(True)

    ax_comp = fig.add_subplot(gs[2, :])
    ax_comp.plot(com_ruido[:n_amostras], alpha=0.25, color='gray', lw=1,
                 label='Com ruído')
    ax_comp.plot(sem_ruido[:n_amostras], color='black', lw=2,
                 label='Original (referência)')
    for tipo in ['butter', 'cheby1', 'cheby2', 'ellip']:
        info = melhores_por_tipo[tipo]
        estrela = " ★ MELHOR GLOBAL" if tipo == melhor_tipo else ""
        ax_comp.plot(info['sinal'][:n_amostras], color=CORES[tipo], lw=1.8,
                     linestyle='--',
                     label=(f"{NOMES[tipo]} | Ordem {info['ordem']} | "
                            f"fc={info['fc']} Hz | RMSE={info['rmse']:.4f}{estrela}"))
    ax_comp.set_title("Comparativo — Todos os Tipos (melhor configuração de cada um)",
                      fontsize=11, fontweight='bold')
    ax_comp.set_xlabel("Amostras")
    ax_comp.set_ylabel("Corrente (A)")
    ax_comp.legend(loc='upper right')
    ax_comp.grid(True)

    fig.savefig(arquivo, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  figura salva: {arquivo}")


def plot_melhor_global(sem_ruido, com_ruido, info, tipo, n_amostras=400,
                       arquivo='melhor_filtro_iir_b9.png'):
    """Destaque do melhor filtro global no domínio do tempo."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(com_ruido[:n_amostras], alpha=0.4, color='gray', lw=1, label='Com ruído')
    ax.plot(sem_ruido[:n_amostras], color='black', lw=2, label='Original (referência)')
    ax.plot(info['sinal'][:n_amostras], color=CORES[tipo], lw=2.5, linestyle='--',
            label=(f"Melhor global: {NOMES[tipo]} | Ordem {info['ordem']} | "
                   f"fc = {info['fc']} Hz | RMSE = {info['rmse']:.4f}"))
    ax.set_title(f"★ Melhor Filtro IIR Global — {NOMES[tipo]} "
                 f"(Ordem {info['ordem']}, fc = {info['fc']} Hz)\n"
                 f"RMSE = {info['rmse']:.4f} | ΔSNR = {info['snr']:.1f} dB | Grupo B9",
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
                             arquivo='resposta_frequencia_b9.png'):
    """Resposta em magnitude (freqz) dos melhores filtros de cada tipo."""
    fig, ax = plt.subplots(figsize=(11, 5))
    for tipo in ['butter', 'cheby1', 'cheby2', 'ellip']:
        info = melhores_por_tipo[tipo]
        w, h = signal.sosfreqz(info['sos'], worN=4096, fs=fs)
        ax.plot(w, 20 * np.log10(np.abs(h) + 1e-12), color=CORES[tipo], lw=1.6,
                label=f"{NOMES[tipo]} (Ordem {info['ordem']}, fc={info['fc']} Hz)")
        ax.axvline(info['fc'], color=CORES[tipo], lw=0.8, alpha=0.3, linestyle=':')
    ax.set_title("Resposta em Frequência dos Melhores Filtros (Grupo B9)",
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
                      arquivo='espectro_fft_b9.png'):
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
    ax.set_title("Espectro de Frequência — Antes e Depois da Filtragem (Grupo B9)",
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


def plot_custo_vs_ordem(df_custo, tipo, fc, arquivo='custo_vs_ordem_b9.png'):
    """Eixo duplo: RMSE e custo computacional em função da ordem."""
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(df_custo['ordem'], df_custo['rmse'], marker='o', color='#004B87',
             lw=1.6, label='RMSE')
    ax1.set_xlabel("Ordem do Filtro")
    ax1.set_ylabel("RMSE", color='#004B87')
    ax1.tick_params(axis='y', labelcolor='#004B87')
    ax1.grid(True)
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


def plot_polos_zeros(info_global, arquivo='polos_zeros_b9.png'):
    """Diagrama de polos e zeros no plano Z (estabilidade do melhor filtro)."""
    zeros, polos, _ = signal.sos2zpk(info_global['sos'])
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


def plot_causal_vs_fase_zero(sem_ruido, com_ruido, fs, df_busca, n_amostras=400,
                             arquivo='causal_vs_fasezero_b9.png'):
    """
    Compara o melhor filtro de fase zero com o melhor filtro causal (realizável
    em tempo real). Evidencia o atraso de fase introduzido pela filtragem causal.
    """
    melhor_fz = df_busca[df_busca['fase_zero']].sort_values('rmse').iloc[0]
    melhor_ca = df_busca[~df_busca['fase_zero']].sort_values('rmse').iloc[0]

    filt_fz, _ = aplicar_filtro_iir(com_ruido, fs, melhor_fz['fc'],
                                    int(melhor_fz['ordem']), melhor_fz['tipo'],
                                    fase_zero=True)
    filt_ca, _ = aplicar_filtro_iir(com_ruido, fs, melhor_ca['fc'],
                                    int(melhor_ca['ordem']), melhor_ca['tipo'],
                                    fase_zero=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(com_ruido[:n_amostras], alpha=0.3, color='gray', lw=1, label='Com ruído')
    ax.plot(sem_ruido[:n_amostras], color='black', lw=2, label='Original (referência)')
    ax.plot(filt_fz[:n_amostras], color='#004B87', lw=1.8, linestyle='--',
            label=(f"Fase zero: {NOMES[melhor_fz['tipo']]} O{int(melhor_fz['ordem'])} "
                   f"fc={melhor_fz['fc']} Hz | RMSE={melhor_fz['rmse']:.4f}"))
    ax.plot(filt_ca[:n_amostras], color='#E69F00', lw=1.8, linestyle='-.',
            label=(f"Causal: {NOMES[melhor_ca['tipo']]} O{int(melhor_ca['ordem'])} "
                   f"fc={melhor_ca['fc']} Hz | RMSE={melhor_ca['rmse']:.4f}"))
    ax.set_title("Filtragem Causal (tempo real) vs. Fase Zero (offline) — Grupo B9",
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

    print("Iniciando busca em grade (tipo x ordem x corte x método)...")
    df = busca_em_grade(sem_ruido, com_ruido, fs)
    print(f"Configurações estáveis avaliadas: {len(df)}\n")

    # Melhor configuração de fase zero por tipo de aproximação.
    melhores_por_tipo = {}
    for tipo in ['butter', 'cheby1', 'cheby2', 'ellip']:
        sub = df[(df['tipo'] == tipo) & (df['fase_zero'])].sort_values('rmse')
        melhor = sub.iloc[0]
        filtrado, sos = aplicar_filtro_iir(
            com_ruido, fs, melhor['fc'], int(melhor['ordem']), tipo, fase_zero=True)
        custo = medir_custo_computacional(
            com_ruido, fs, melhor['fc'], int(melhor['ordem']), tipo, fase_zero=True)
        melhores_por_tipo[tipo] = {
            'tipo': tipo, 'ordem': int(melhor['ordem']), 'fc': int(melhor['fc']),
            'rmse': float(melhor['rmse']),
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
    plot_comparativo_tempo(sem_ruido, com_ruido, melhores_por_tipo, melhor_tipo)
    plot_melhor_global(sem_ruido, com_ruido, info_global, melhor_tipo)
    plot_resposta_frequencia(melhores_por_tipo, fs)
    plot_espectro_fft(sem_ruido, com_ruido, info_global, fs)
    plot_polos_zeros(info_global)
    plot_causal_vs_fase_zero(sem_ruido, com_ruido, fs, df)

    df_custo = analise_custo_vs_ordem(
        sem_ruido, com_ruido, fs, tipo=melhor_tipo, fc=info_global['fc'])
    if not df_custo.empty:
        plot_custo_vs_ordem(df_custo, melhor_tipo, info_global['fc'])

    # Exporta a tabela completa de resultados para reuso/relatório.
    df.sort_values('rmse').to_csv('resultados_busca_b9.csv', index=False)
    print("  tabela salva: resultados_busca_b9.csv")
    print("\nConcluído.")


if __name__ == "__main__":
    main()
