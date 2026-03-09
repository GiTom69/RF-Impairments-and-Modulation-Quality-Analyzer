import numpy as np
import streamlit as st

from rf_analyzer.gui.controls_panel import get_current_controls, initialize_control_state, render_controls
from rf_analyzer.gui.plots_panel import render_information_chain, render_metrics, render_visualizations
from rf_analyzer.impairments.adc_quantization import quantize
from rf_analyzer.utils.fft_utils import compute_power_spectrum


def _gray_code(value: int) -> int:
	return value ^ (value >> 1)


def _bits_per_symbol(modulation_type: str) -> int:
	return {"BPSK": 1, "QPSK": 2, "16-QAM": 4, "64-QAM": 6}[modulation_type]


def _int_to_bits(value: int, width: int) -> np.ndarray:
	return np.array([(value >> bit) & 1 for bit in range(width - 1, -1, -1)], dtype=np.uint8)


def _bits_to_int(bits: np.ndarray) -> int:
	result = 0
	for bit in bits:
		result = (result << 1) | int(bit)
	return result


def _pam_levels(bits_axis: np.ndarray, bits_per_axis: int) -> float:
	max_level = 2**bits_per_axis - 1
	binary = _bits_to_int(bits_axis)
	gray = _gray_code(binary)
	return float(2 * gray - max_level)


def _generate_symbols(bits: np.ndarray, modulation_type: str) -> tuple[np.ndarray, np.ndarray]:
	bps = _bits_per_symbol(modulation_type)
	padding = (-len(bits)) % bps
	if padding:
		bits = np.pad(bits, (0, padding), mode="constant")
	groups = bits.reshape(-1, bps)

	if modulation_type == "BPSK":
		symbols = (2 * groups[:, 0] - 1).astype(float).astype(np.complex128)
		return symbols, bits

	if modulation_type == "QPSK":
		i = 2 * groups[:, 0] - 1
		q = 2 * groups[:, 1] - 1
		symbols = (i + 1j * q) / np.sqrt(2)
		return symbols.astype(np.complex128), bits

	bits_per_axis = bps // 2
	levels = np.array([_pam_levels(group[:bits_per_axis], bits_per_axis) + 1j * _pam_levels(group[bits_per_axis:], bits_per_axis) for group in groups])
	avg_energy = np.mean(np.abs(levels) ** 2)
	return (levels / np.sqrt(avg_energy)).astype(np.complex128), bits


def _upsample_signal(signal: np.ndarray, samples_per_symbol: int) -> np.ndarray:
	if samples_per_symbol <= 1:
		return signal.astype(np.complex128, copy=True)
	return np.repeat(signal.astype(np.complex128), samples_per_symbol)


def _sample_symbols_from_waveform(signal: np.ndarray, samples_per_symbol: int) -> np.ndarray:
	if samples_per_symbol <= 1:
		return signal.astype(np.complex128, copy=True)
	return signal[::samples_per_symbol].astype(np.complex128, copy=False)


def _demodulate_bits(rx_symbols: np.ndarray, modulation_type: str) -> np.ndarray:
	bps = _bits_per_symbol(modulation_type)

	if modulation_type == "BPSK":
		return (rx_symbols.real >= 0).astype(np.uint8)

	if modulation_type == "QPSK":
		bits_i = (rx_symbols.real >= 0).astype(np.uint8)
		bits_q = (rx_symbols.imag >= 0).astype(np.uint8)
		return np.column_stack((bits_i, bits_q)).reshape(-1)

	bits_per_axis = bps // 2
	levels = np.arange(-(2**bits_per_axis - 1), 2**bits_per_axis, 2)
	gray_order = np.array([_gray_code(index) for index in range(2**bits_per_axis)])
	thresholds = (levels[:-1] + levels[1:]) / 2

	max_axis = np.max(np.abs(np.concatenate([rx_symbols.real, rx_symbols.imag])))
	if max_axis == 0:
		max_axis = 1.0
	scale = levels[-1] / max_axis
	scaled_i = rx_symbols.real * scale
	scaled_q = rx_symbols.imag * scale

	indices_i = np.digitize(scaled_i, thresholds)
	indices_q = np.digitize(scaled_q, thresholds)
	binary_i = np.array([gray_order[idx] for idx in indices_i], dtype=int)
	binary_q = np.array([gray_order[idx] for idx in indices_q], dtype=int)

	bits_i = np.vstack([_int_to_bits(value, bits_per_axis) for value in binary_i])
	bits_q = np.vstack([_int_to_bits(value, bits_per_axis) for value in binary_q])
	return np.hstack((bits_i, bits_q)).reshape(-1).astype(np.uint8)


def _parse_source_bits(source_type: str, source_input: str, count: int = 4096) -> np.ndarray:
	if source_type == "Random":
		seed = None
		text = source_input.strip()
		if text:
			try:
				seed = int(text)
			except ValueError:
				seed = abs(hash(text)) % (2**32)
		rng = np.random.default_rng(seed)
		return rng.integers(0, 2, size=count, dtype=np.uint8)

	if source_type == "File":
		try:
			with open(source_input, "r", encoding="utf-8") as file:
				text = file.read()
		except OSError:
			return np.zeros(count, dtype=np.uint8)
		filtered = [char for char in text if char in ("0", "1")]
		if not filtered:
			return np.zeros(count, dtype=np.uint8)
		bits = np.array([int(char) for char in filtered], dtype=np.uint8)
		if len(bits) >= count:
			return bits[:count]
		repeats = int(np.ceil(count / len(bits)))
		return np.tile(bits, repeats)[:count]

	filtered = [char for char in source_input if char in ("0", "1")]
	if not filtered:
		return np.zeros(count, dtype=np.uint8)
	bits = np.array([int(char) for char in filtered], dtype=np.uint8)
	repeats = int(np.ceil(count / len(bits)))
	return np.tile(bits, repeats)[:count]


def _apply_iq_imbalance(signal: np.ndarray, gain_db: float, phase_deg: float) -> np.ndarray:
	gain_lin = 10 ** (gain_db / 20)
	phi = np.deg2rad(phase_deg)
	i = signal.real * gain_lin
	q = signal.imag
	q_misaligned = q * np.cos(phi) + i * np.sin(phi)
	return i + 1j * q_misaligned


def _apply_phase_noise(signal: np.ndarray, phase_deg: float, rng: np.random.Generator) -> np.ndarray:
	if phase_deg <= 0:
		return signal
	sigma = np.deg2rad(phase_deg)
	noise_phase = rng.normal(0.0, sigma, size=len(signal))
	return signal * np.exp(1j * noise_phase)


def _apply_pa_nonlinearity(signal: np.ndarray, p1dbcp_dbm: float) -> np.ndarray:
	compression = 10 ** (p1dbcp_dbm / 20)
	if compression <= 0:
		compression = 1e-9
	soft_real = compression * np.tanh(signal.real / compression)
	soft_imag = compression * np.tanh(signal.imag / compression)
	return soft_real + 1j * soft_imag


def _normalize_or_clip_iq(signal: np.ndarray, clipping_enabled: bool) -> np.ndarray:
	if clipping_enabled:
		return np.clip(signal.real, -1.0, 1.0) + 1j * np.clip(signal.imag, -1.0, 1.0)

	max_mag = float(np.max(np.abs(signal))) if signal.size else 0.0
	scale = max(max_mag, 1.0)
	return signal / scale


def _add_awgn(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
	signal_power = np.mean(np.abs(signal) ** 2)
	if signal_power <= 0:
		return signal
	noise_power = signal_power / (10 ** (snr_db / 10))
	noise = (rng.normal(0, np.sqrt(noise_power / 2), len(signal)) + 1j * rng.normal(0, np.sqrt(noise_power / 2), len(signal)))
	return signal + noise


def _compute_evm_percent(rx: np.ndarray, ref: np.ndarray) -> float:
	error = rx - ref
	ref_power = np.mean(np.abs(ref) ** 2)
	if ref_power <= 0:
		return 0.0
	return float(np.sqrt(np.mean(np.abs(error) ** 2) / ref_power) * 100)


def _compute_acpr_db(psd: np.ndarray, freqs: np.ndarray) -> float:
	if len(psd) < 30:
		return 0.0
	freq_sorted_idx = np.argsort(freqs)
	p = np.asarray(psd)[freq_sorted_idx]
	n = len(p)
	bw = max(n // 6, 1)
	center = n // 2
	main_start = max(center - bw // 2, 0)
	main_end = min(main_start + bw, n)
	left_start = max(main_start - bw, 0)
	left_end = main_start
	right_start = main_end
	right_end = min(main_end + bw, n)
	left = np.sum(p[left_start:left_end])
	main = np.sum(p[main_start:main_end])
	right = np.sum(p[right_start:right_end])
	adj = max(left, right)
	if main <= 0:
		return 0.0
	return float(10 * np.log10((adj + 1e-15) / (main + 1e-15)))


def _compute_obw_hz(psd: np.ndarray, freqs: np.ndarray, power_fraction: float = 0.99) -> float:
	if len(psd) == 0:
		return 0.0
	order = np.argsort(freqs)
	f = np.asarray(freqs)[order]
	p = np.asarray(psd)[order]
	total = np.sum(p)
	if total <= 0:
		return 0.0
	cum = np.cumsum(p)
	lower = np.searchsorted(cum, (1 - power_fraction) * total / 2)
	upper = np.searchsorted(cum, (1 + power_fraction) * total / 2)
	upper = min(max(upper, lower), len(f) - 1)
	return float(abs(f[upper] - f[lower]))


def _compute_snr_db(reference: np.ndarray, observed: np.ndarray) -> float:
	noise = observed - reference
	noise_power = np.mean(np.abs(noise) ** 2)
	signal_power = np.mean(np.abs(reference) ** 2)
	if noise_power <= 0 or signal_power <= 0:
		return 0.0
	return float(10 * np.log10(signal_power / noise_power))


def _compute_stage_metrics(
	reference_symbols: np.ndarray,
	stage_symbols: np.ndarray,
	reference_bits: np.ndarray,
	modulation_type: str,
	spectrum_signal: np.ndarray | None = None,
	sampling_rate_hz: float = 1e6,
) -> dict:
	stage_bits = _demodulate_bits(stage_symbols, modulation_type)[: len(reference_bits)]
	bit_errors = int(np.sum(stage_bits != reference_bits))
	ber = bit_errors / len(reference_bits) if len(reference_bits) else 0.0
	fft_signal = stage_symbols if spectrum_signal is None else spectrum_signal
	freqs, psd = compute_power_spectrum(fft_signal, sampling_rate=sampling_rate_hz)
	ev_percent = _compute_evm_percent(stage_symbols, reference_symbols)
	ev_ratio = max(ev_percent / 100.0, 1e-12)
	acpr_db = _compute_acpr_db(psd, freqs)
	obw_hz = _compute_obw_hz(psd, freqs)
	return {
		"freqs": freqs,
		"psd_db": 10 * np.log10(psd + 1e-15),
		"acpr_db": acpr_db,
		"obw_hz": obw_hz,
		"metrics": {
			"evm_percent": ev_percent,
			"evm_db": float(20 * np.log10(ev_ratio)),
			"ber": ber,
			"snr_db": _compute_snr_db(reference_symbols, stage_symbols),
		},
	}


def _quantize_complex_signal(signal: np.ndarray, sr_hz: float, bits: int) -> np.ndarray:
	real = np.array([quantize(float(value), sr_hz, bits) for value in signal.real])
	imag = np.array([quantize(float(value), sr_hz, bits) for value in signal.imag])
	return real + 1j * imag


def run_app() -> None:
	st.set_page_config(layout="wide", page_title="RF Analyzer")
	st.title("RF Impairments & Modulation Quality Analyzer")
	initialize_control_state()
	controls = get_current_controls()
	rng = np.random.default_rng(1234)
	samples_per_symbol = max(int(controls["samples_per_symbol"]), 1)
	symbol_rate_hz = 1e6
	sampling_rate_hz = symbol_rate_hz * samples_per_symbol

	tx_bits = _parse_source_bits(controls["source_type"], controls["source_input"], count=4096)
	tx_symbols, padded_bits = _generate_symbols(tx_bits, controls["modulation_type"])
	tx_waveform = _upsample_signal(tx_symbols, samples_per_symbol)

	iq_modulator_waveform = tx_waveform.copy()
	impaired = _apply_iq_imbalance(
		iq_modulator_waveform.copy(),
		controls["iq_gain_mismatch_db"],
		controls["iq_phase_mismatch_deg"],
	)
	impaired = _apply_phase_noise(impaired, controls["phase_noise_deg"], rng)
	impairments_waveform = _apply_pa_nonlinearity(impaired, controls["pa_1dbcp_dbm"])
	impairments_waveform = _normalize_or_clip_iq(impairments_waveform, controls["clipping_enabled"])
	channel_out = _add_awgn(impairments_waveform, controls["snr_db"], rng)
	rx_waveform = _quantize_complex_signal(channel_out, sr_hz=sampling_rate_hz, bits=controls["adc_bits"])

	iq_modulator_symbols = _sample_symbols_from_waveform(iq_modulator_waveform, samples_per_symbol)
	impairments_symbols = _sample_symbols_from_waveform(impairments_waveform, samples_per_symbol)
	channel_symbols = _sample_symbols_from_waveform(channel_out, samples_per_symbol)
	rx_symbols = _sample_symbols_from_waveform(rx_waveform, samples_per_symbol)

	stage_results = [
		{
			"name": "IQ MODULATOR",
			"symbols": iq_modulator_symbols,
			"ideal_symbols": tx_symbols,
			**_compute_stage_metrics(
				tx_symbols,
				iq_modulator_symbols,
				padded_bits,
				controls["modulation_type"],
				spectrum_signal=iq_modulator_waveform,
				sampling_rate_hz=sampling_rate_hz,
			),
		},
		{
			"name": "IMPARMENTS",
			"symbols": impairments_symbols,
			"ideal_symbols": tx_symbols,
			**_compute_stage_metrics(
				tx_symbols,
				impairments_symbols,
				padded_bits,
				controls["modulation_type"],
				spectrum_signal=impairments_waveform,
				sampling_rate_hz=sampling_rate_hz,
			),
		},
		{
			"name": "CHANNEL",
			"symbols": channel_symbols,
			"ideal_symbols": tx_symbols,
			**_compute_stage_metrics(
				tx_symbols,
				channel_symbols,
				padded_bits,
				controls["modulation_type"],
				spectrum_signal=channel_out,
				sampling_rate_hz=sampling_rate_hz,
			),
		},
		{
			"name": "ADC",
			"symbols": rx_symbols,
			"ideal_symbols": tx_symbols,
			**_compute_stage_metrics(
				tx_symbols,
				rx_symbols,
				padded_bits,
				controls["modulation_type"],
				spectrum_signal=rx_waveform,
				sampling_rate_hz=sampling_rate_hz,
			),
		},
	]

	final_ber = stage_results[-1]["metrics"]["ber"]

	info_chain_summary = {
		"bits_source": f"{controls['source_type']}",
		"iq_modulator": f"{controls['modulation_type']}, SPS={controls['samples_per_symbol']}",
		"impairments": (
			f"IQ G={controls['iq_gain_mismatch_db']} dB, Phase={controls['iq_phase_mismatch_deg']}°, "
			f"PN={controls['phase_noise_deg']}°, PA={controls['pa_1dbcp_dbm']} dBm, "
			f"{'Clip' if controls['clipping_enabled'] else 'Normalize'}"
		),
		"channel": f"SNR={controls['snr_db']} dB",
		"adc": f"{controls['adc_bits']} bits",
		"bits_dest": f"BER={final_ber:.3e}",
	}

	render_visualizations(stage_results)
	render_metrics(stage_results)
	render_information_chain(info_chain_summary)
	render_controls()
