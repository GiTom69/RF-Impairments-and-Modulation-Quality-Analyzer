import numpy as np
import plotly.graph_objects as go
import streamlit as st


def _render_spectrum_figure(stage: dict, show_obw: bool, show_acpr: bool) -> go.Figure:
	fig = go.Figure(go.Scatter(x=stage["freqs"], y=stage["psd_db"], name="Spectrum"))
	fig.update_layout(
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
	all_points = np.concatenate(
		[
			stage["symbols"],
			stage["ideal_symbols"] if show_ideal else np.array([], dtype=np.complex128),
		]
	)
	max_abs = np.max(np.abs(np.concatenate([all_points.real, all_points.imag]))) if len(all_points) else 1.0
	axis_limit = float(max(max_abs * 1.1, 1.0))
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
		xaxis_title="I",
		yaxis_title="Q",
		margin=dict(l=5, r=5, t=5, b=5),
		showlegend=False,
	)
	fig.update_xaxes(range=[-axis_limit, axis_limit], constrain="range")
	fig.update_yaxes(range=[-axis_limit, axis_limit], scaleanchor="x", scaleratio=1, constrain="range")
	return fig


def render_visualizations(stage_results: list[dict]) -> None:
	st.subheader("1. Visualizations")

	toggle_cols = st.columns([2, 2, 2, 10])
	with toggle_cols[0]:
		show_obw = st.toggle("Show OBW", value=True)
	with toggle_cols[1]:
		show_acpr = st.toggle("Show ACPR", value=True)
	with toggle_cols[2]:
		show_ideal = st.toggle("Show ideal symbols", value=True)

	stage_layout_columns = st.columns([4, 1, 4, 1, 4, 1, 4])
	stage_columns = [stage_layout_columns[0], stage_layout_columns[2], stage_layout_columns[4], stage_layout_columns[6]]
	for index, (col, stage) in enumerate(zip(stage_columns, stage_results)):
		with col:
			st.markdown(f"**{stage['name']}**")
			st.caption("Spectrum")
			st.plotly_chart(
				_render_spectrum_figure(stage, show_obw, show_acpr),
				width="stretch",
				key=f"spectrum_{index}_{stage['name']}",
			)
			st.caption("Constellation")
			st.plotly_chart(
				_render_constellation_figure(stage, show_ideal),
				width="stretch",
				key=f"constellation_{index}_{stage['name']}",
			)


def render_metrics(stage_results: list[dict]) -> None:
	st.subheader("2. Metrics")
	metric_labels = ["EVM %", "EVM dB", "BER", "ACPR", "OBW", "SNR"]
	for col, stage in zip(st.columns(len(stage_results)), stage_results):
		with col:
			st.markdown(f"**{stage['name']}**")
			metrics = stage["metrics"]
			metric_values = [
				f"{metrics['evm_percent']:.4f}",
				f"{metrics['evm_db']:.2f}",
				f"{metrics['ber']:.6f}",
				f"{metrics['acpr_db']:.2f} dB",
				f"{metrics['obw_hz']:.1f} Hz",
				f"{metrics['snr_db']:.2f} dB",
			]
			for label, value in zip(metric_labels, metric_values):
				st.markdown(
					(
						"<div style='font-size:0.80rem; line-height:1.05; margin:0 0 0.2rem 0;'>"
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
