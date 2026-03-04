import importlib

import numpy as np
import streamlit as st

from rf_analyzer.gui.controls_panel import render_controls
from rf_analyzer.gui.plots_panel import render_information_chain, render_metrics, render_plots
from rf_analyzer.impairments.adc_quantization import quantize
from rf_analyzer.impairments.pa_nonlinearity import PA_amplify
from rf_analyzer.signal.qam import qam16_modulate
from rf_analyzer.utils.fft_utils import compute_power_spectrum


def _optional_callable(module_path: str, function_name: str):
	try:
		module = importlib.import_module(module_path)
		return getattr(module, function_name, None)
	except Exception:
		return None


def _fallback_evm(rx: np.ndarray, ref: np.ndarray) -> float:
	error = rx - ref
	ref_power = np.mean(np.abs(ref) ** 2)
	if ref_power <= 0:
		return 0.0
	return float(np.sqrt(np.mean(np.abs(error) ** 2) / ref_power) * 100)


def _fallback_acpr(psd: np.ndarray, freqs: np.ndarray) -> float:
	if len(psd) < 9:
		return 0.0
	idx = np.argsort(freqs)
	p = np.asarray(psd)[idx]
	n = len(p)
	third = n // 3
	left = np.sum(p[:third])
	main = np.sum(p[third : 2 * third])
	right = np.sum(p[2 * third :])
	adj = max(left, right)
	if main <= 0:
		return 0.0
	return float(10 * np.log10((adj + 1e-15) / (main + 1e-15)))


def _compute_evm(rx: np.ndarray, ref: np.ndarray) -> float:
	compute_evm = _optional_callable("rf_analyzer.metrics.evm", "compute_evm")
	if callable(compute_evm):
		return float(compute_evm(rx, ref))
	return _fallback_evm(rx, ref)


def _compute_acpr(psd: np.ndarray, freqs: np.ndarray) -> float:
	compute_acpr = _optional_callable("rf_analyzer.metrics.acpr", "compute_acpr")
	if callable(compute_acpr):
		return float(compute_acpr(psd, freqs))
	return _fallback_acpr(psd, freqs)


def _apply_optional_impairments(signal: np.ndarray, controls: dict) -> np.ndarray:
	output = signal

	apply_phase_noise = _optional_callable("rf_analyzer.impairments.phase_noise", "apply_phase_noise")
	if callable(apply_phase_noise) and controls["ph_noise"] > 0:
		output = apply_phase_noise(output, controls["ph_noise"])

	apply_iq_imbalance = _optional_callable("rf_analyzer.impairments.iq_imbalance", "apply_iq_imbalance")
	if callable(apply_iq_imbalance) and (controls["iq_gain"] > 0 or controls["iq_phase"] > 0):
		output = apply_iq_imbalance(output, controls["iq_gain"], controls["iq_phase"])

	return output


def _quantize_complex_signal(signal: np.ndarray, sr_hz: float, bits: int) -> np.ndarray:
	quantize_r = np.vectorize(lambda v: quantize(float(v), sr_hz, bits))
	real = quantize_r(signal.real)
	imag = quantize_r(signal.imag)
	return real + 1j * imag


def run_app() -> None:
	st.set_page_config(layout="wide", page_title="RF Analyzer")
	st.title("RF Impairments & Modulation Quality Analyzer")

	controls = render_controls()

	bits = np.random.randint(0, 2, 1024)
	if controls["mod_order"] != 16:
		st.info("Current signal mapper supports 16-QAM; using 16-QAM for now.")
	symbols = qam16_modulate(bits)

	pa_out = PA_amplify(symbols, gain_lin=10 ** (controls["pa_ibo"] / 20))
	impaired = _apply_optional_impairments(pa_out, controls)
	rx = _quantize_complex_signal(impaired, sr_hz=1e6, bits=controls["adc_bits"])

	freqs, psd_pre = compute_power_spectrum(symbols, sampling_rate=1e6)
	freqs, psd_post = compute_power_spectrum(rx, sampling_rate=1e6)

	evm = _compute_evm(rx, symbols)
	acpr = _compute_acpr(psd_post, freqs)
	eff_snr = controls["snr"] - controls["noise_fig"]

	render_metrics(evm=evm, acpr=acpr, eff_snr=eff_snr)
	render_plots(freqs=freqs, psd_pre=psd_pre, psd_post=psd_post, signal=rx)
	render_information_chain()
