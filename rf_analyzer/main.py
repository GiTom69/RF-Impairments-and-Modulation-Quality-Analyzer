# app.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go

from rf_analyzer.signal.qam import qam16_modulate
from rf_analyzer.impairments.pa_nonlinearity import PA_amplify
from rf_analyzer.impairments.phase_noise import apply_phase_noise   # when ready
from rf_analyzer.impairments.iq_imbalance import apply_iq_imbalance # when ready
from rf_analyzer.impairments.adc_quantization import quantize
from rf_analyzer.metrics.evm import compute_evm
from rf_analyzer.metrics.acpr import compute_acpr
from rf_analyzer.utils.fft_utils import compute_power_spectrum

st.set_page_config(layout="wide", page_title="RF Analyzer")
st.title("RF Impairments & Modulation Quality Analyzer")

# ── SIDEBAR CONTROLS ──────────────────────────
with st.sidebar:
    st.header("Tx Chain")
    mod_order = st.selectbox("Modulation", [4, 16, 64, 256], index=1,
                             format_func=lambda x: {4:"QPSK",16:"16-QAM",64:"64-QAM",256:"256-QAM"}[x])
    pa_ibo    = st.slider("PA IBO (dB)",      0.0, 20.0, 10.0, 0.5)
    ph_noise  = st.slider("Phase Noise (°)",  0.0, 12.0,  2.0, 0.5)
    iq_gain   = st.slider("IQ Gain Imb (dB)", 0.0,  4.0,  0.0, 0.1)
    iq_phase  = st.slider("IQ Phase Imb (°)", 0.0, 15.0,  0.0, 0.5)

    st.header("Channel")
    snr       = st.slider("SNR (dB)",          0, 50, 30)

    st.header("Rx Chain")
    adc_bits  = st.slider("ADC Bits",           4, 16, 12)
    noise_fig = st.slider("Noise Figure (dB)", 0.0, 20.0, 5.0, 0.5)

# ── DSP PIPELINE ──────────────────────────────
bits     = np.random.randint(0, 2, 1024)
symbols  = qam16_modulate(bits)          # swap based on mod_order
pa_out   = PA_amplify(symbols, gain_lin=10**(pa_ibo/20))
# ... apply remaining impairments

freqs, psd_pre  = compute_power_spectrum(symbols,  fs=1e6)
freqs, psd_post = compute_power_spectrum(pa_out,   fs=1e6)

evm  = compute_evm(pa_out,  symbols)
acpr = compute_acpr(psd_post, freqs)

# ── METRICS ROW ───────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("EVM",  f"{evm:.2f}%")
c2.metric("ACPR", f"{acpr:.1f} dBc")
c3.metric("Eff. SNR", f"{snr - noise_fig:.1f} dB")

# ── PLOTS GRID ────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    fig = go.Figure(go.Scatter(x=freqs, y=10*np.log10(psd_pre + 1e-12)))
    fig.update_layout(title="SA: Pre-PA", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = go.Figure(go.Scatter(x=freqs, y=10*np.log10(psd_post + 1e-12)))
    fig.update_layout(title="SA: Post-PA", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

with col3:
    fig = go.Figure(go.Scatter(x=pa_out.real, y=pa_out.imag, mode='markers'))
    fig.update_layout(title="IQ: Constellation", template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)