import streamlit as st


def render_controls() -> dict:
	with st.sidebar:
		st.header("Tx Chain")
		mod_order = st.selectbox(
			"Modulation",
			[4, 16, 64, 256],
			index=1,
			format_func=lambda x: {4: "QPSK", 16: "16-QAM", 64: "64-QAM", 256: "256-QAM"}[x],
		)
		pa_ibo = st.slider("PA IBO (dB)", 0.0, 20.0, 10.0, 0.5)
		ph_noise = st.slider("Phase Noise (°)", 0.0, 12.0, 2.0, 0.5)
		iq_gain = st.slider("IQ Gain Imb (dB)", 0.0, 4.0, 0.0, 0.1)
		iq_phase = st.slider("IQ Phase Imb (°)", 0.0, 15.0, 0.0, 0.5)

		st.header("Channel")
		snr = st.slider("SNR (dB)", 0, 50, 30)

		st.header("Rx Chain")
		adc_bits = st.slider("ADC Bits", 4, 16, 12)
		noise_fig = st.slider("Noise Figure (dB)", 0.0, 20.0, 5.0, 0.5)

	return {
		"mod_order": mod_order,
		"pa_ibo": pa_ibo,
		"ph_noise": ph_noise,
		"iq_gain": iq_gain,
		"iq_phase": iq_phase,
		"snr": snr,
		"adc_bits": adc_bits,
		"noise_fig": noise_fig,
	}
