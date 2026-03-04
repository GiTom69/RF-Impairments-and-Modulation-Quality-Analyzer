import numpy as np
import plotly.graph_objects as go
import streamlit as st


def render_information_chain() -> None:
	st.subheader("Information Chain")
	nodes = [
		"Bit Generator",
		"QAM Modulator",
		"RF Impairments Block",
		"TX AMPLIFIER",
		"AWGN",
		"RX AMPLIFIER",
		"ADC Quantization",
		"Metrics Engine",
	]

	chain_items: list[tuple[str, str]] = []
	for index, node in enumerate(nodes):
		chain_items.append(("node", node))
		if index < len(nodes) - 1:
			chain_items.append(("arrow", "→"))

	widths = [3 if item_type == "node" else 1 for item_type, _ in chain_items]
	cols = st.columns(widths)

	for col, (item_type, text) in zip(cols, chain_items):
		with col:
			if item_type == "node":
				st.markdown(
					f"<div style='text-align:center; font-weight:600; white-space:nowrap;'>{text}</div>",
					unsafe_allow_html=True,
				)
			else:
				st.markdown(
					"<div style='text-align:center; font-size:1.2rem;'>→</div>",
					unsafe_allow_html=True,
				)


def render_metrics(evm: float, acpr: float, eff_snr: float) -> None:
	c1, c2, c3 = st.columns(3)
	c1.metric("EVM", f"{evm:.2f}%")
	c2.metric("ACPR", f"{acpr:.1f} dBc")
	c3.metric("Eff. SNR", f"{eff_snr:.1f} dB")

def render_plots(freqs: np.ndarray, psd_pre: np.ndarray, psd_post: np.ndarray, signal: np.ndarray) -> None:
	col1, col2, col3 = st.columns(3)

	with col1:
		fig = go.Figure(go.Scatter(x=freqs, y=10 * np.log10(psd_pre + 1e-12)))
		fig.update_layout(title="TX Spectrum: Pre-PA", template="plotly_dark")
		st.plotly_chart(fig, width="stretch")

	with col2:
		fig = go.Figure(go.Scatter(x=freqs, y=10 * np.log10(psd_post + 1e-12)))
		fig.update_layout(title="TX Spectrum: Post-PA", template="plotly_dark")
		st.plotly_chart(fig, width="stretch")

	with col3:
		fig = go.Figure(go.Scatter(x=signal.real, y=signal.imag, mode="markers"))
		fig.update_layout(title="TX IQ: Constellation", template="plotly_dark")
		st.plotly_chart(fig, width="stretch")
