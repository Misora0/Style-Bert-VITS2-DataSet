"""スコア計算・設定読み込みの回帰テスト (合成音声で「真値」が分かるケースを固定する).

- SNR: 雑音床が -55 dBFS を超える収録でも真の SNR を返し、録音レベルに依存しない
- jitter/shimmer: Praat の local 定義と同じ値になり、有声区間の長さに依存しない
- text_overrides: 本文を直したら x_cer を新しい本文で引き直す
- 設定: --config の打ち間違い・未知キーを起動時に弾く
- residual_symbols: NFKC 後の ! ? ... を残留記号として数えない
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import scipy.signal
import soundfile as sf
import typer

from make_dataset import cli, itn
from make_dataset import stage6_output as s6
from make_dataset.config import Config
from make_dataset.manifest import Record
from make_dataset.overrides import apply_text_overrides
from make_dataset.stage2_score import score_segments
from make_dataset.stage3_asr import normalize_text

SR = 44100


def _voiced(dur: float, rng, jitter: float = 0.0, level_db: float = -20.0, f0: float = 200.0):
    """声門パルス列 + 3 フォルマント。周期揺らぎ jitter は相対 std。RMS を level_db に合わせる."""
    n = int(dur * SR)
    x = np.zeros(n)
    t = 0.0
    while t < dur:
        if int(t * SR) < n:
            x[int(t * SR)] = 1.0
        t += (1.0 / f0) * (1.0 + jitter * rng.standard_normal())
    for fc, bw in [(700, 130), (1200, 70), (2600, 160)]:
        r = np.exp(-np.pi * bw / SR)
        th = 2 * np.pi * fc / SR
        x = scipy.signal.lfilter([1 - r], [1, -2 * r * np.cos(th), r * r], x)
    x /= np.sqrt(np.mean(x**2)) + 1e-12
    return x * 10 ** (level_db / 20)


def _utterance(rng, noise_db: float, speech_db: float = -20.0, jitter: float = 0.0):
    """話声 0.8s / ポーズ 0.25s / 話声 0.8s (前後 50ms) + 白色雑音 (RMS = noise_db)."""
    sp = _voiced(0.8, rng, jitter=jitter, level_db=speech_db)
    z = np.zeros(int(0.05 * SR))
    x = np.concatenate([z, sp, np.zeros(int(0.25 * SR)), sp, z])
    return (x + rng.standard_normal(len(x)) * 10 ** (noise_db / 20)).astype(np.float32)


def _score(tmp_path: Path, xs: list[np.ndarray]) -> list[Record]:
    recs = []
    for i, x in enumerate(xs):
        sf.write(tmp_path / f"u{i}.flac", x, SR, format="FLAC", subtype="PCM_24")
        recs.append(Record(utt_id=f"u{i}", src="s", start=0.0, end=len(x) / SR))
    score_segments(recs, Config.load(None), tmp_path)
    return recs


def test_snr_tracks_true_snr_above_minus55_dbfs_noise(tmp_path):
    """旧実装は雑音床が -55 dBFS を超えると SNR が ~35 dB に張り付き、真の 5 dB でも 35.5 dB と出た."""
    rng = np.random.default_rng(0)
    noise_levels = [-70, -50, -40, -35, -30]
    recs = _score(tmp_path, [_utterance(rng, nd) for nd in noise_levels])
    for nd, r in zip(noise_levels, recs, strict=True):
        true_snr = -20 - nd
        assert r.scores["snr"] == pytest.approx(true_snr, abs=2.0), (nd, r.scores["snr"])
    # 真の SNR 5 dB は min_snr_db = 5 を通ってはいけない (旧実装は 35.5 dB で素通りしていた)
    (r5,) = _score(tmp_path, [_utterance(rng, -25)])
    assert r5.scores["snr"] < 5.0


def test_snr_is_independent_of_recording_level(tmp_path):
    """同じ真の SNR (15 dB) なら録音レベルが違っても同じ SNR (旧実装は 14.9〜44.3 dB に散った)."""
    rng = np.random.default_rng(1)
    recs = _score(tmp_path, [_utterance(rng, s - 15, speech_db=s) for s in (-10, -20, -30, -40)])
    snrs = [r.scores["snr"] for r in recs]
    assert max(snrs) - min(snrs) < 2.0, snrs
    assert all(r.scores["voiced_ratio"] < 0.95 for r in recs)  # ポーズを有声と数えない


def test_jitter_matches_praat_local_definition(tmp_path):
    """旧実装は分母に有声区間長が掛かり、周期揺らぎ 4% でも 0.0014 (Praat 0.035) に潰れていた."""
    import parselmouth
    from parselmouth.praat import call

    rng = np.random.default_rng(2)
    xs = [_utterance(rng, -70, jitter=j) for j in (0.0, 0.02, 0.04)]
    recs = _score(tmp_path, xs)
    praat = []
    for x in xs:
        x16 = scipy.signal.resample_poly(x, 160, 441)
        snd = parselmouth.Sound(x16.astype(np.float64), sampling_frequency=16000)
        pp = call(snd, "To PointProcess (periodic, cc)", 75.0, 600.0)
        praat.append(call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
    got = [r.scores["jitter_local"] for r in recs]
    for g, p in zip(got, praat, strict=True):
        assert g == pytest.approx(p, abs=0.002), (got, praat)
    assert got[0] < got[1] < got[2]  # 揺らぎが大きいほど大きい (単調)
    assert got[2] > 0.02  # 4% の揺らぎは既定ゲート max_jitter_local=0.02 を超える
    assert all(r.scores["shimmer_local"] is not None for r in recs)


def test_jitter_does_not_depend_on_voiced_length(tmp_path):
    rng = np.random.default_rng(3)
    xs = []
    for dur in (0.3, 0.8, 2.0, 4.0):
        z = np.zeros(int(0.1 * SR))
        x = np.concatenate([z, _voiced(dur, rng, jitter=0.01), z])
        xs.append((x + rng.standard_normal(len(x)) * 10 ** (-70 / 20)).astype(np.float32))
    js = [r.scores["jitter_local"] for r in _score(tmp_path, xs)]
    # 旧実装は 0.3s→0.0001、4s→0.0 と長さに反比例して潰れた。短い区間は周期数が少なく
    # 推定がばらつくので幅を持たせ、「同じ声なら同じ桁 (Praat の ~0.7%) に収まる」ことだけ見る
    assert all(0.004 < j < 0.015 for j in js), js


def _passing_scores() -> dict:
    return {
        "seg_dur": 2.0, "snr": 30.0, "hf_energy_ratio": 0.01, "reverb_score": 0.1,
        "jitter_local": 0.005, "speaker_sim": 0.9, "pesq": 3.0, "asr_conf": 0.9,
        "align_conf": 0.9, "clip_ratio": 0.0, "dc_offset_abs": 0.0,
        "digital_silence_ratio": 0.0, "voiced_ratio": 0.8, "token_prob_min": 0.5,
        "token_nll": 0.05, "tag_hit": None, "lm_ppl": 50.0, "mora_count": 4,
        "mora_rate": 4.0, "mora_rate_dev": 0.0, "g2p_ok": True, "unknown_tokens": 0,
        "text_tail_open": 0, "kana_ratio": 1.0, "char_count": 5, "filler_ratio": 0.0,
    }  # fmt: skip


def test_text_override_recomputes_x_cer(tmp_path):
    """主 ASR「米紙。」/ 第 2 系統「こめかみ。」を本文「こめかみ。」に直したら x_cer=0 で採用に戻る."""
    r = Record(utt_id="u1", src="s", start=0.0, end=2.0, text="米紙。")
    r.scores.update(_passing_scores())
    r.scores.update({"asr2_text": "こめかみ。", "x_cer": 1.0, "x_cer_surface": 1.0})
    cfg = Config({"thresholds": {"max_x_cer": 0.45}, "lm": {"enabled": False}})
    (tmp_path / "text_overrides.jsonl").write_text(
        json.dumps({"utt_id": "u1", "text": "こめかみ。"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    apply_text_overrides([r], tmp_path, cfg)
    assert r.scores["x_cer"] == 0.0
    s6.evaluate_checks([r], cfg)
    s6.finalize_statuses([r])
    assert r.status == "adopted", r.dropped_by


def test_config_load_rejects_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        Config.load(tmp_path / "filtres.toml")
    assert Config.load(None).thresholds == {}  # パス省略 (テスト用) は従来どおり DEFAULTS のみ


def test_cli_rejects_typo_config_and_unknown_keys(tmp_path):
    with pytest.raises(typer.Exit):
        cli._load_config(str(tmp_path / "filtres.toml"))
    bad = tmp_path / "filters.toml"
    bad.write_text("[thresholds]\nmin_snr_bd = 5.0\n[asr]\nmodle = 'tiny'\n", encoding="utf-8")
    with pytest.raises(typer.Exit):
        cli._load_config(str(bad))
    good = tmp_path / "ok.toml"
    good.write_text("[thresholds]\nmin_snr_db = 5.0\nmin_mora_rate = 2.0\n", encoding="utf-8")
    assert cli._load_config(str(good)).thresholds["min_snr_db"] == 5.0


def test_bundled_filters_toml_has_no_unknown_keys():
    """同梱 filters.toml は検証を通る (検証が厳しすぎて既定設定で起動できない、を防ぐ)."""
    path = Path(__file__).parents[1] / "filters.toml"
    cfg = cli._load_config(str(path))
    assert cfg.thresholds


@pytest.mark.parametrize("raw", ["本当！？", "ごめんね…", "え、ほんとに？", "すごい!!", "まって～"])
def test_residual_symbols_ignores_normalized_punctuation(raw):
    """NFKC 後の ! ? ... ~ を残留記号と数えない (数えると感嘆・疑問文が max_residual_symbols で全滅)."""
    assert itn.residual_symbols(normalize_text(raw)) == 0
    assert itn.residual_symbols(normalize_text("50#です")) == 1  # 本当の残留記号は数え続ける
