import numpy as np
import plotly.graph_objects as go
import streamlit as st


_CONSTELLATION_SIZE_PX = 320


def _get_sorted_spectrum(stage: dict) -> tuple[np.ndarray, np.ndarray]:
	order = np.argsort(stage["freqs"])
	freqs = np.asarray(stage["freqs"])[order]
	psd_db = np.asarray(stage["psd_db"])[order]
	return freqs, psd_db


def _render_spectrum_figure(stage: dict, show_obw: bool, show_acpr: bool) -> go.Figure:
	freqs, psd_db = _get_sorted_spectrum(stage)
	fig = go.Figure(go.Scatter(x=freqs, y=psd_db, mode="lines", name="Spectrum"))
	fig.update_layout(
		title=dict(text="Spectrum", x=0.5, xanchor="center", y=0.98, yanchor="top"),
		xaxis_title="Frequency (Hz)",
		yaxis_title="Power (dB)",
		margin=dict(l=5, r=5, t=5, b=5),
		height=260,
	)
	if show_obw and stage["metrics"]["obw_hz"] > 0:
		fig.add_vrect(
			x0=-stage["metrics"]["obw_hz"] / 2,
			x1=stage["metrics"]["obw_hz"] / 2,
			fillcolor="LightGreen",
			opacity=0.2,
			line_width=0,
		)
	if show_acpr:
		fig.add_annotation(
			xref="paper",
			yref="paper",
			x=0.98,
			y=0.95,
			text=f"ACPR {stage['metrics']['acpr_db']:.2f} dB",
			showarrow=False,
		)
	return fig

def _render_constellation_figure(stage: dict, show_ideal: bool) -> go.Figure:
	fig = go.Figure()
	axis_limit = 1.1
	fig.add_trace(
		go.Scatter(
			x=stage["symbols"].real,
			y=stage["symbols"].imag,
			mode="markers",
			name="Received",
			showlegend=False,
			marker=dict(size=5),
		)
	)
	if show_ideal:
		fig.add_trace(
			go.Scatter(
				x=stage["ideal_symbols"].real,
				y=stage["ideal_symbols"].imag,
				mode="markers",
				name="Ideal",
				showlegend=False,
				marker=dict(size=5, symbol="x"),
			)
		)
	fig.update_layout(
		title=dict(text="Constellation", x=0.5, xanchor="center", y=0.98, yanchor="top", font=dict(size=12)),
		xaxis_title="I",
		yaxis_title="Q",
		margin=dict(l=35, r=35, t=30, b=35),
		showlegend=False,
		width=_CONSTELLATION_SIZE_PX,
		height=_CONSTELLATION_SIZE_PX,
		xaxis=dict(
			range=[-1.1, 1.1], 
			constrain="domain",
			automargin=False,
			autorange=False, 
			fixedrange=True,
			scaleanchor="y",
		),
		yaxis=dict(
			range=[-1.1, 1.1], 
			constrain="domain",
			scaleratio=1,
			automargin=False, 
			autorange=False, 
			fixedrange=True,
		),
	)
	return fig

def render_visualizations(stage_results: list[dict]) -> None:
	st.subheader("1. Visualizations")
	st.markdown(
		"""
		<style>
		div[data-testid="stPlotlyChart"] {
			border: 2px solid rgba(180, 180, 180, 0.8) !important;
			border-radius: 4px;
			padding: 2px;
			display: block;
			overflow: hidden !important;
			box-sizing: border-box;
			margin-bottom: 8px !important;
		}
		div[data-testid="stPlotlyChart"] > div {
			border-bottom: none !important;
			overflow: hidden !important;
		}
		div[data-testid="stPlotlyChart"] * {
			overflow: hidden !important;
		}
		</style>
		""",
		unsafe_allow_html=True,
	)

	toggle_cols = st.columns([2, 2, 2, 10])
	with toggle_cols[0]:
		show_obw = st.toggle("Show OBW", value=False)
	with toggle_cols[1]:
		show_acpr = st.toggle("Show ACPR", value=False)
	with toggle_cols[2]:
		show_ideal = st.toggle("Show ideal symbols", value=False)

	stage_layout_columns = st.columns([8, 1, 8, 1, 8, 1, 8])
	stage_columns = [stage_layout_columns[0], stage_layout_columns[2], stage_layout_columns[4], stage_layout_columns[6]]
	
	for index, (col, stage) in enumerate(zip(stage_columns, stage_results)):
		with col:
			st.markdown(f"**{stage['name']}**")
			st.plotly_chart(
				_render_spectrum_figure(stage, show_obw, show_acpr),
				width="stretch",
				config={"staticPlot": True},
				key=f"spectrum_{index}_{stage['name']}",
			)
			st.plotly_chart(
				_render_constellation_figure(stage, show_ideal),
				width=_CONSTELLATION_SIZE_PX,
				config={"staticPlot": True},
				key=f"constellation_{index}_{stage['name']}",
			)

def render_metrics(stage_results: list[dict]) -> None:
	st.subheader("2. Metrics")
	st.markdown(
		"""
		<style>
		div.metric-value {
			font-size: 1.60rem;
			line-height: 1.05;
			margin: 0 0 0.2rem 0;
		}
		</style>
		""",
		unsafe_allow_html=True,
	)

	stage_layout_columns = st.columns([4, 1, 4, 1, 4, 1, 4])
	stage_columns = [stage_layout_columns[0], stage_layout_columns[2], stage_layout_columns[4], stage_layout_columns[6]]
	
	for index, (col, stage) in enumerate(zip(stage_columns, stage_results)):
		with col:
			metrics = stage["metrics"]
			left_metrics = [
				("ACPR", f"{metrics['acpr_db']:.2f} dB"),
				("OBW", f"{metrics['obw_hz']:.1f} Hz"),
				("SNR", f"{metrics['snr_db']:.2f} dB"),
			]
			right_metrics = [
				("EVM %", f"{metrics['evm_percent']:.4f}"),
				("EVM (dB)", f"{metrics['evm_db']:.2f}"),
				("BER", f"{metrics['ber']:.6f}"),
			]

			metric_columns = st.columns(2)
			for metric_column, metric_group in zip(metric_columns, [left_metrics, right_metrics]):
				with metric_column:
					for label, value in metric_group:
						st.markdown(
							(
								"<div style='font-size:1.00rem; line-height:1.05; margin:0 0 0.2rem 0;'>"
								f"<span style='font-weight:600'>{label}:</span> {value}"
								"</div>"
							),
							unsafe_allow_html=True,
						)

def render_information_chain(config_summary: dict) -> None:
	st.subheader("3. Information Chain")
	items = [
		("node", "BITS SOURCE", config_summary["bits_source"]),
		("arrow", "→", ""),
		("node", "IQ MODULATOR", config_summary["iq_modulator"]),
		("arrow", "→", ""),
		("node", "IMPAIRMENTS", config_summary["impairments"]),
		("arrow", "→", ""),
		("node", "CHANNEL", config_summary["channel"]),
		("arrow", "→", ""),
		("node", "ADC QUANT", config_summary["adc"]),
		("arrow", "→", ""),
		("node", "BITS DEST", config_summary["bits_dest"]),
	]
	widths = [4 if kind == "node" else 1 for kind, _, _ in items]
	for col, (kind, title, subtitle) in zip(st.columns(widths), items):
		with col:
			if kind == "arrow":
				st.markdown("### →")
			else:
				with st.container(border=True):
					st.markdown(f"**{title}**")
					if subtitle:
						st.caption(subtitle)
