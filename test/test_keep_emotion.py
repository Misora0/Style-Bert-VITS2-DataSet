"""感情の手がかりを残す変更の回帰テスト (docs/DECISIONS.md 7.5).

- 文末の ? / ! を第 2 系統から補う (punct.py / stage3)
- 1 ファイル 1 セリフの素材を間 (ため) で分割しない (stage1 whole_file_max_s)
- 同じセリフの別テイクを残す (stage5 max_takes_per_text) と、多様性選別がテイク同士を間引かない
- クリッピングは「潰れた頭」だけ数え、フルスケールに触れただけの波形は数えない (stage2)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from make_dataset import stage3_asr as s3
from make_dataset import stage5_dedup as s5
from make_dataset.config import Config
from make_dataset.manifest import Record
from make_dataset.punct import asr2_terminal_mark, restore_terminal_punct
from make_dataset.stage2_score import clipped_ratio

# ---------------------------------------------------------------- 文末記号の補完


@pytest.mark.parametrize(
    ("asr2", "mark"),
    [
        ("そうなんですか?", "?"),
        ("先生!", "!"),
        ("それはー!", "!"),
        ("え、本当?!", "?!"),
        ("すごい!!!", "!"),
        ("どういうこと？」", "?"),
        ("そうなんですか", None),
        ("", None),
        (None, None),
    ],
)
def test_asr2_terminal_mark(asr2, mark):
    assert asr2_terminal_mark(asr2) == mark


def test_restore_terminal_punct_only_when_readings_agree():
    assert (
        restore_terminal_punct("そうなんですか。", "そうなんですか?", 0.0, 0.15)
        == "そうなんですか?"
    )
    assert restore_terminal_punct("先生", "先生!", 0.0, 0.15) == "先生!"  # 文末記号なしにも付ける
    # 読みが食い違う (「いいですか」/「良いですか?」) ときは記号も借りない
    assert restore_terminal_punct("いいですか。", "良いですか?", 0.2, 0.15) is None
    assert restore_terminal_punct("いいですか。", "いいですか?", None, 0.15) is None  # x_cer 不明
    assert restore_terminal_punct("本当?", "本当!", 0.0, 0.15) is None  # 既に記号があれば触らない
    assert restore_terminal_punct("はい。", "はい", 0.0, 0.15) is None  # 第 2 系統にも記号なし


def test_transcribe_restores_terminal_punct(monkeypatch):
    import make_dataset.stage0_ingest as s0

    monkeypatch.setattr(s0, "load_cached", lambda r, c, sr: np.zeros(16000))

    def fake(audio, cfg, prefix="asr"):
        return (
            ("信用して本当にいいんですか。", 0.9, {})
            if prefix == "asr"
            else ("信用して本当にいいんですか?", 0.8, {})
        )

    monkeypatch.setattr(s3, "_asr_whisper", fake)
    cfg = Config(
        {
            "asr": {"backend": "faster-whisper", "model": "tiny"},
            "asr2": {"enabled": True, "backend": "faster-whisper", "model": "small",
                     "restore_punct": True, "restore_punct_max_x_cer": 0.15},
        }
    )  # fmt: skip
    r = Record(utt_id="u0", src="s", start=0.0, end=1.5)
    r.scores["seg_dur"] = 1.5
    s3.transcribe_segments([r], cfg, Path("."))
    assert r.text == "信用して本当にいいんですか?"
    assert r.scores["punct_restored"] == "?"
    assert r.scores["x_cer"] == 0.0 and r.scores["x_cer_surface"] == 0.0
    assert r.scores["text_tail_open"] == 0  # ? も文として閉じている

    # 既定 (restore_punct=False) では本文を変えない (従来動作)
    cfg.data["asr2"]["restore_punct"] = False
    r2 = Record(utt_id="u1", src="s", start=0.0, end=1.5)
    r2.scores["seg_dur"] = 1.5
    s3.transcribe_segments([r2], cfg, Path("."))
    assert r2.text == "信用して本当にいいんですか。" and "punct_restored" not in r2.scores


# ---------------------------------------------------------------- 1 ファイル 1 セリフは分割しない


def _segment_with_fake_vad(tmp_path: Path, monkeypatch, whole_file_max_s: float, dur: float):
    """「話声 / 0.4s の間 / 話声」のファイルを、VAD が 2 発話に分けた状況で Stage 1 に通す."""
    import silero_vad

    from make_dataset.stage0_ingest import Source
    from make_dataset.stage1_segment import segment_sources

    sr = 44100
    t = np.arange(int(dur * sr)) / sr
    x = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    gap = (t > dur / 2 - 0.2) & (t < dur / 2 + 0.2)
    x[gap] = 0.0
    x[: int(0.1 * sr)] = 0.0
    x[-int(0.1 * sr) :] = 0.0
    path = tmp_path / "line.wav"
    sf.write(path, x, sr)

    n16 = int(dur * 16000)
    fake_ts = [
        {"start": int(0.1 * 16000), "end": int((dur / 2 - 0.2) * 16000)},
        {"start": int((dur / 2 + 0.2) * 16000), "end": n16 - int(0.1 * 16000)},
    ]
    monkeypatch.setattr(silero_vad, "load_silero_vad", lambda: None)
    monkeypatch.setattr(silero_vad, "get_speech_timestamps", lambda *a, **k: list(fake_ts))
    cfg = Config({"stage1": {"whole_file_max_s": whole_file_max_s, "decode_chunk_min": 0}})
    return segment_sources([Source(src_hash="h", path=path)], cfg, tmp_path / "cache")


def test_short_file_is_not_split_at_pause(tmp_path, monkeypatch):
    recs = _segment_with_fake_vad(tmp_path, monkeypatch, whole_file_max_s=12.0, dur=3.0)
    assert len(recs) == 1
    r = recs[0]
    assert r.scores.get("whole_file") is True
    assert r.start < 0.15 and r.end > 2.85  # 最初の発話の頭〜最後の発話の終わり (中の間は残す)
    assert 2.6 < r.scores["seg_dur"] < 3.0


def test_split_unchanged_when_disabled_or_long(tmp_path, monkeypatch):
    # 無効 (既定 0) なら従来どおり間で 2 つに切る
    assert len(_segment_with_fake_vad(tmp_path, monkeypatch, whole_file_max_s=0.0, dur=3.0)) == 2
    # 上限より長いファイル (長尺収録) も従来どおり
    assert len(_segment_with_fake_vad(tmp_path, monkeypatch, whole_file_max_s=2.0, dur=3.0)) == 2


# ---------------------------------------------------------------- 別テイクを残す


def _take(uid: str, text: str, f0: float, lufs: float, fp_seed: int, q: float = 3.0) -> Record:
    r = Record(utt_id=uid, src=f"{uid}.wav", start=0.0, end=1.0, text=text)
    rng = np.random.default_rng(fp_seed)
    r.scores.update(
        {
            "failed_checks": [],
            "quality_score": q,
            "f0_median_hz": f0,
            "lufs": lufs,
            "fp_sig": [int(v) for v in rng.integers(0, 2**40, size=s5.FP_NUM_PERM)],
        }
    )
    return r


def _dedup_cfg(k: int) -> Config:
    return Config({"dedup": {"max_takes_per_text": k, "take_min_diff": 1.0}})


def test_alt_takes_are_kept_up_to_k():
    """同じ「はい。」でも別の録音 (指紋が別) で演技が違うものは k 本まで残す."""
    recs = [
        _take("a", "はい。", 300, -12, 1, q=3.5),  # 代表 (品質最大)
        _take("b", "はい。", 302, -12.2, 2),  # 代表とほぼ同じ演技 → 残さない
        _take("c", "はい。", 400, -9, 3),  # 高く大きい → 残す
        _take("d", "はい。", 240, -18, 4),  # 低く小さい → 残す
        _take("e", "はい。", 390, -9.5, 5),  # c とほぼ同じ → k=3 なので残らない
    ]
    s5.dedup(recs, _dedup_cfg(3))
    kept = {r.utt_id for r in recs if not r.dup_reason}
    assert kept == {"a", "c", "d"}
    assert all(r.scores.get("alt_take") for r in recs if r.utt_id in {"c", "d"})
    assert {r.dup_reason for r in recs if r.utt_id in {"b", "e"}} == {"exact"}


def test_same_recording_is_still_collapsed():
    """同じ録音のズレ切り出し (指紋一致) は、演技の違いが見えても 1 本にまとめる."""
    a = _take("a", "はい。", 300, -12, 1, q=3.5)
    b = _take("b", "はい。", 400, -6, 1)  # 指紋が a と同一
    s5.dedup([a, b], _dedup_cfg(3))
    assert not a.dup_reason and b.dup_reason == "exact" and not b.scores.get("alt_take")


def test_k1_keeps_legacy_behaviour():
    recs = [_take("a", "はい。", 300, -12, 1, q=3.5), _take("c", "はい。", 400, -9, 3)]
    s5.dedup(recs, _dedup_cfg(1))
    assert [r.dup_reason for r in recs] == [None, "exact"]


def test_phoneme_fuzzy_is_labeled_fuzzy_phoneme():
    """音声を比べていない音素ファジィは fuzzy_phoneme (旧版は fuzzy_audio と誤表示)."""
    a = _take("a", "そうなんだ!", 300, -12, 1, q=3.5)
    b = _take("b", "そうなんだ。", 300, -12, 2)
    s5.dedup([a, b], _dedup_cfg(1))
    assert b.dup_reason == "fuzzy_phoneme"


def test_diversity_does_not_prune_kept_takes():
    recs = [
        _take("a", "はい。", 300, -12, 1, q=3.5),
        _take("c", "はい。", 400, -9, 3),
        _take("d", "はい。", 240, -18, 4),
    ]
    cfg = Config(
        {
            "dedup": {"max_takes_per_text": 3, "take_min_diff": 1.0},
            "diversity": {"enabled": True, "metric": "phoneme", "sim_threshold": 0.6},
        }
    )
    s5.dedup(recs, cfg)
    s5.diversity_select(recs, cfg)
    assert all(not r.dup_reason and not r.diversity_reason for r in recs)


# ---------------------------------------------------------------- クリッピング


def _voice(seconds: float = 2.0, f0: float = 300.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.arange(int(seconds * 44100)) / 44100
    v = sum(np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6)) / k for k in range(1, 15))
    return v / np.abs(v).max()


def test_loud_but_clean_audio_is_not_clipping():
    """フルスケールに触れるだけの無傷の波形 (大音量マスタリング) は 0. 旧計算では 0.002 を超えていた."""
    x = _voice(f0=400)
    assert np.mean(np.abs(x) >= 0.999) > 0.002  # 旧 clip_ratio なら除外されていた
    assert clipped_ratio(x) == 0.0
    assert clipped_ratio(np.round(x * 0.999 * 32767) / 32767) == 0.0  # 16bit 量子化しても同じ


def test_real_clipping_is_detected_even_below_full_scale():
    x = _voice()
    hard = np.clip(x * 1.5, -1, 1)
    assert clipped_ratio(hard) > 0.1
    assert clipped_ratio(np.clip(x * 1.05, -1, 1)) > 0.02  # 軽いクリップも閾値 0.002 を大きく超える
    assert clipped_ratio(hard * 0.5) == pytest.approx(
        clipped_ratio(hard)
    )  # クリップ後に音量を下げた素材
