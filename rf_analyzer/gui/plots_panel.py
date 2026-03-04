import numpy as np
import plotly.graph_objects as go
import streamlit as st


def render_metrics(evm: float, acpr: float, eff_snr: float) -> None:
	c1, c2, c3 = st.columns(3)
	c1.metric("EVM", f"{evm:.2f}%")
	c2.metric("ACPR", f"{acpr:.1f} dBc")
	c3.metric("Eff. SNR", f"{eff_snr:.1f} dB")


def render_plots(freqs: np.ndarray, psd_pre: np.ndarray, psd_post: np.ndarray, signal: np.ndarray) -> None:
	col1, col2, col3 = st.columns(3)

	with col1:
		fig = go.Figure(go.Scatter(x=freqs, y=10 * np.log10(psd_pre + 1e-12)))
		fig.update_layout(title="SA: Pre-PA", template="plotly_dark")
		st.plotly_chart(fig, use_container_width=True)

	with col2:
		fig = go.Figure(go.Scatter(x=freqs, y=10 * np.log10(psd_post + 1e-12)))
		fig.update_layout(title="SA: Post-PA", template="plotly_dark")
		st.plotly_chart(fig, use_container_width=True)

	with col3:
		fig = go.Figure(go.Scatter(x=signal.real, y=signal.imag, mode="markers"))
		fig.update_layout(title="IQ: Constellation", template="plotly_dark")
		st.plotly_chart(fig, use_container_width=True)
