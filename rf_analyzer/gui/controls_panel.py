import streamlit as st


DEFAULT_CONTROLS = {
	"source_type": "Random",
	"source_input": "12345",
	"modulation_type": "16-QAM",
	"samples_per_symbol": 8,
	"iq_gain_mismatch_db": 0,
	"iq_phase_mismatch_deg": 0,
	"phase_noise_deg": 0,
	"pa_1dbcp_dbm": 0,
	"snr_db": 20,
	"adc_bits": 12,
}


def _begin_section(title: str):
	st.markdown(
		"""
		<style>
		.section-title-column {
			display: flex;
			justify-content: center;
			align-items: flex-start;
			height: 100%;
			min-height: 120px;
			overflow: visible;
			padding-top: 1.25rem;
		}
		.section-title-rotated {
			transform: rotate(90deg);
			transform-origin: center center;
			white-space: nowrap;
			font-size: 1.35rem;
			font-weight: 700;
			line-height: 1;
		}
		</style>
		""",
		unsafe_allow_html=True,
	)
	title_col, content_col = st.columns([1, 24])
	with title_col:
		st.markdown(
			f"<div class='section-title-column'><div class='section-title-rotated'>{title}</div></div>",
			unsafe_allow_html=True,
		)
	return content_col


def initialize_control_state() -> None:
	for key, value in DEFAULT_CONTROLS.items():
		if key not in st.session_state:
			st.session_state[key] = value


def get_current_controls() -> dict:
	initialize_control_state()
	return {key: st.session_state[key] for key in DEFAULT_CONTROLS}


def _source_input_label(source_type: str) -> str:
	if source_type == "Random":
		return "Seed"
	if source_type == "File":
		return "File path"
	return "Binary string (0/1)"


def render_controls() -> dict:
	content_col = _begin_section("Settings")
	with content_col:
		cols = st.columns(6)

		with cols[0]:
			st.markdown("**Bit Source**")
			st.selectbox(
				"Source Type",
				["Random", "File", "binary string"],
				key="source_type",
			)
			st.text_input(
				_source_input_label(st.session_state["source_type"]),
				key="source_input",
			)

		with cols[1]:
			st.markdown("**IQ Modulation**")
			st.selectbox(
				"Modulation type",
				["BPSK", "QPSK", "16-QAM", "64-QAM"],
				key="modulation_type",
			)
			st.slider(
				"Samples per symbol",
				min_value=1,
				max_value=100,
				step=1,
				key="samples_per_symbol",
			)

		with cols[2]:
			st.markdown("**RF Impairments**")
			st.slider(
				"Gain mismatch (dB)",
				min_value=-10,
				max_value=10,
				step=1,
				key="iq_gain_mismatch_db",
			)
			st.slider(
				"Phase mismatch (deg)",
				min_value=0,
				max_value=90,
				step=1,
				key="iq_phase_mismatch_deg",
			)
			st.slider(
				"Phase Noise (deg)",
				min_value=0,
				max_value=90,
				step=1,
				key="phase_noise_deg",
			)
			st.slider(
				"PA Nonlinearity (1dBcp, dBm)",
				min_value=-10,
				max_value=10,
				step=1,
				key="pa_1dbcp_dbm",
			)

		with cols[3]:
			st.markdown("**Channel**")
			st.slider(
				"SNR (dB)",
				min_value=-10,
				max_value=40,
				step=1,
				key="snr_db",
			)

		with cols[4]:
			st.markdown("**ADC**")
			st.slider(
				"Number of bits",
				min_value=4,
				max_value=32,
				step=1,
				key="adc_bits",
			)

		with cols[5]:
			st.markdown("&nbsp;", unsafe_allow_html=True)

	return get_current_controls()
