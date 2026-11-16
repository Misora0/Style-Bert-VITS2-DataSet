"""Stage 2: 発話スコアリング. speaker_sim / rolloff / jitter / shimmer / F0 / snr / lufs を manifest に積み上げる.

- 埋め込み統計はコーパス全体で次元ごとに std へスケーリング (CMVN の分散側) してから cosine を取る。
  c0 (エネルギー) が cosine を支配して speaker_sim が「音量類似度」になるのを防ぐ。正規化後のベクトルを
  scores["emb"] に保存するので、rescore は再計算なしに同じ指紋で比較できる。
- voiced/unvoiced は騒音床 (下限 -55 dBFS) で分離する (中央値分割だと voiced_ratio が常に ≒0.5 で無情報)。
- jitter/shimmer は Praat の PointProcess (periodic, cc) から Get jitter/shimmer (local) で取る
  (Praat の定義そのもの。ピッチ軌跡の差分による近似は使わない)。
- LUFS ゲートはメータのブロック長 (0.4s) ベース。計算不能・非有限なら None を記録し、
  x * -inf で nan を撒き散らさない。
"""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

from .config import Config
from .manifest import Record


def _finite(v) -> float | None:
    """Praat の undefined (nan) や非有限値を None にする (検証不能として扱う)."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def score_segments(records: list[Record], cfg: Config, cache_dir) -> None:
    """各 Record.scores にスコアを追記する (除外可否は Stage 6 の閾値で決める)."""
    import pyloudnorm as pyln
    import scipy.signal

    from .stage0_ingest import load_cached, resample_to

    meter = pyln.Meter(44100)
    encoder = _encoder()
    raw_embs: list[np.ndarray] = []
    emb_rows: list[Record] = []

    for r in tqdm(records, desc="stage2 score", unit="utt"):
        try:  # 1 本の異常セグメントで数時間のランを全損させない
            x = load_cached(r, cache_dir, 44100)
            x16 = resample_to(x, 44100, 16000)
            sr = 44100

            # クリッピング / デジタル無音 / DC オフセット (学習後に修復不可能なので投入時点で弾く)
            clip_ratio = float(np.mean(np.abs(x) >= 0.999))
            dc_offset = float(abs(np.mean(x)))
            digital_silence = float(np.mean(x == 0.0))

            # 帯域ロールオフ / 高域エネルギー比: 16k/24k アップコンの「見せかけ高品質」検出
            freqs, psd = scipy.signal.welch(x, fs=sr, nperseg=min(2048, max(len(x), 2)))
            cum = np.cumsum(psd)
            rolloff = float(freqs[min(int(np.searchsorted(cum, 0.99 * cum[-1])), len(freqs) - 1)])
            hf_ratio = float(psd[freqs > 7000].sum() / (psd.sum() + 1e-12))

            # フレームエネルギーを騒音床で分割して SNR と voiced_ratio を作る。
            #   雑音レベル = フレーム RMS の p10 (発話内の最も静かな 1 割)。
            #   有声判定の床 = max(雑音レベル + 6 dB, -55 dBFS)。-55 dBFS は「床の下限」:
            #   旧実装は min(p20×3, -55 dBFS) で床が常に -55 dBFS 以下に抑えられ、雑音床が -55 dBFS を
            #   超える収録 (家庭録音・配信など大半) では全フレームが床超えになって、SNR が
            #   「話声レベル - (-55)」= 録音レベルの指標に化けていた (真の SNR 5 dB が 35 dB と出た)。
            #   p20×3 (+9.5 dB) はポーズが 2 割未満のきつい切り出しで床が話声を超えるため p10+6 dB にした。
            #   ポーズがほぼ無い発話では p10 が「静かな話声」になり、SNR は控えめ (下振れ) に出る。
            hop = sr // 100  # 10ms
            fe = _frame_rms(x, sr, hop)
            noise = float(np.percentile(fe, 10))
            floor = max(noise * 2.0, 10 ** (-55 / 20))
            vmask = fe > floor
            if (~vmask).any() and vmask.any():
                snr = float(
                    10 * np.log10((fe[vmask] ** 2).mean() / ((fe[~vmask] ** 2).mean() + 1e-12))
                )
            elif vmask.any():
                # 床を下回るフレームが 1 本もない (ポーズ無しの切り出し) → 雑音推定 (p10) 比で算出
                snr = float(10 * np.log10((fe[vmask] ** 2).mean() / (noise**2 + 1e-12)))
            else:
                snr = None  # ほぼ無音のセグメント (検証不能としてゲート側で不合格扱い)
            voiced_ratio = float(vmask.mean())

            # LUFS はメータのブロック長以上ある時だけ計算。不能/非有限は None (偽値 -120 を撒かない)
            if len(x) >= meter.block_size * sr:
                v = meter.integrated_loudness(x)
                lufs = float(v) if np.isfinite(v) else None
            else:
                lufs = None

            # jitter / shimmer は Praat の定義そのもの (声門周期の PointProcess 上の local 摂動) で測る。
            #   旧実装は 10ms 刻みのピッチ軌跡の差分を使い、しかも分母にだけ有声区間長を掛けていたため
            #   値が「真の jitter ÷ 有声フレーム数」に潰れていた (周期揺らぎ 4% で 0.0014、Praat では 0.035)。
            #   Praat: jitter(local) = 連続周期差の絶対値平均 / 平均周期 (Voice 2. Jitter)。
            #   周期が数えられない (無声・短すぎ) ときは Praat が undefined (nan) を返す → None。
            import parselmouth
            from parselmouth.praat import call

            seg = parselmouth.Sound(
                np.ascontiguousarray(x16, dtype=np.float64), sampling_frequency=16000
            )
            st = seg.to_pitch_ac(time_step=0.01)
            f = st.selected_array["frequency"]
            valid = ~np.isnan(f) & (f > 0)
            jitter = shimmer = None
            if valid.any():
                try:
                    pp = call(seg, "To PointProcess (periodic, cc)", 75.0, 600.0)
                    jitter = _finite(call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
                    shimmer = _finite(
                        call([seg, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
                    )
                except parselmouth.PraatError:  # 周期点が取れない等: 検証不能 (None) として記録
                    jitter = shimmer = None
            f0_median = float(np.median(f[valid])) if valid.any() else None

            r.scores.update(
                {
                    "overlap_flag": None,  # 既定は未接続 (None 記録)。[overlap] enabled で pyannote が上書き
                    "reverb_score": round(_tail_ratio(x), 3),
                    "rolloff_hz": round(rolloff),
                    "hf_energy_ratio": round(hf_ratio, 5),
                    "jitter_local": round(float(jitter), 4) if jitter is not None else None,
                    "shimmer_local": round(float(shimmer), 4) if shimmer is not None else None,
                    "f0_median_hz": round(f0_median, 1) if f0_median is not None else None,
                    "snr": round(snr, 1) if snr is not None else None,
                    "lufs": round(lufs, 1) if lufs is not None else None,
                    "voiced_ratio": round(voiced_ratio, 2),
                    "clip_ratio": round(clip_ratio, 5),
                    "dc_offset_abs": round(dc_offset, 5),
                    "digital_silence_ratio": round(digital_silence, 4),
                }
            )
            if not np.all(np.isfinite(x)):
                r.scores.setdefault("errors", []).append("stage2: non-finite audio samples")
            frames = _mfcc_frames(
                encoder, x16
            )  # MFCC は 1 回だけ計算し emb と内容指紋で共有 (E: 二重計算回避)
            raw_embs.append(_embed_from_frames(frames))
            try:  # 指紋計算の失敗で本体スコアを捨てない (dedup が弱まるだけ)
                r.scores["fp_sig"] = _content_fingerprint_from(
                    frames
                )  # 内容指紋 (話者ベクトルではない)
            except Exception as e:  # noqa: BLE001
                r.scores["fp_sig"] = []
                r.scores.setdefault("fp_error", []).append(repr(e))
            emb_rows.append(r)
        except Exception as e:  # noqa: BLE001
            r.scores.setdefault("errors", []).append(f"stage2: {e!r}")

    _cmvn_store(raw_embs, emb_rows)

    if cfg.get("overlap.enabled"):  # pyannote segmentation-3.0 (有効時のみ。None 記録を上書きする)
        from .overlap import compute_overlap

        compute_overlap(records, cfg, cache_dir)


def _frame_rms(audio: np.ndarray, sr: int, hop: int) -> np.ndarray:
    n = len(audio) // hop * hop
    if n == 0:
        return np.zeros(1, dtype=np.float32)
    frames = audio[:n].reshape(-1, hop)
    return np.sqrt((frames**2).mean(axis=1))


def _tail_ratio(x: np.ndarray) -> float:
    """残響プロキシ: 発話ピーク区間に対する末尾減衰エネルギー比 (O(N) の移動平均)."""
    import scipy.ndimage

    env = np.abs(x)
    if len(env) < 4410:
        return float(env[-1] / (env.max() + 1e-9))
    smooth = scipy.ndimage.uniform_filter1d(
        env, size=4410, mode="nearest"
    )  # sliding_window_view(4410).mean の O(N×W) 回避
    peak = float(smooth.max())
    tail = float(smooth[-1])  # 末尾 100ms 移動平均 = 元の env[-4410:].mean() と同値を O(N) で
    return float(tail / (peak + 1e-9))


_ENCODER = None


def _encoder():
    """話者埋め込みエンコーダ。現状は MFCC 統計のプレースホルダ (WavLM系/ERes2Net/TitaNet への差し替えは未着手、docs/DESIGN.md 参照)."""
    global _ENCODER
    if _ENCODER is None:
        import torchaudio

        _ENCODER = torchaudio.transforms.MFCC(
            sample_rate=16000, n_mfcc=40, melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 40}
        )
        _ENCODER.eval()
    return _ENCODER


def _mfcc_frames(encoder, audio16k: np.ndarray):
    """MFCC フレーム列 (40, T) を 1 回だけ計算する (埋め込みと内容指紋で共有して二重計算を回避)."""
    import torch

    with torch.inference_mode():
        return encoder(torch.from_numpy(np.ascontiguousarray(audio16k)).unsqueeze(0)).squeeze(0)


def _embed_from_frames(f) -> np.ndarray:
    import torch

    # std が非有限/0 でも次元数を 80 のまま保つ (フォールバックで 40 次元になり torch.stack が落ちるのを防ぐ)
    return torch.cat([torch.nan_to_num(f.mean(dim=1)), torch.nan_to_num(f.std(dim=1))]).numpy()


def _content_fingerprint_from(f) -> list[int]:
    """内容指紋 — MFCC フレームの2値化コード → 3フレーム gram → MinHash 署名.

    DESIGN.md:77 の注意 (話者埋め込みは話者ベクトルなので内容照合に使わない) に従い、
    時系列順を持つフレーム列から作る。ズレ切り出し重複で high、別発話で low になる。
    署名は manifest に保存するので rescore は再計算なしで dedup を再実行できる。
    """
    from .stage5_dedup import FP_NUM_PERM, minhash_signature

    if f.shape[1] < 3:
        return []
    codes = (
        (f > f.mean(dim=1, keepdim=True)).T.numpy().astype(np.uint8)
    )  # (T,40) 発話平均に対する2値化
    frames = [codes[i].tobytes() for i in range(len(codes))]
    grams = {frames[i] + frames[i + 1] + frames[i + 2] for i in range(len(frames) - 2)}
    return minhash_signature(grams, num_perm=FP_NUM_PERM)


def _cmvn_store(raw_embs: list[np.ndarray], rows: list[Record]) -> None:
    """次元ごとに std でスケーリング (CMVN の分散側) した埋め込みを scores["emb"] に保存する.

    c0 (エネルギー) が分散の大きさで cosine を支配し「音量類似度」になるのを防ぐ。平均引き算は
    小コーパスで pairwise cosine に -1/(N-1) の偏りを作るため行わない (スケーリングのみ)。
    speaker_sim (主話者 centroid とのコサイン) は全スコアが揃った Stage 6 直前の
    recompute_speaker_sim が頑健サブセットから計算する (不良発話で centroid が汚染されないため)。
    """
    if not raw_embs:
        return
    dim = min(e.shape[0] for e in raw_embs)
    M = np.stack([e[:dim] for e in raw_embs]).astype(np.float64)
    Z = M / (M.std(axis=0) + 1e-9)  # 次元正規化: c0 (エネルギー) が cosine を支配しない
    for z, r in zip(Z, rows, strict=True):  # raw_embs と emb_rows は必ず対で append される
        r.scores["emb"] = [round(float(v), 4) for v in z]


def recompute_speaker_sim(records: list[Record]) -> None:
    """主話者 centroid とのコサイン speaker_sim を頑健サブセットから 2 パスで再計算する (centroid 汚染回避).

    全件中央値だと無音・音楽・別人の多いコーパスで centroid がズレ min_speaker_sim が機能不全になる。
    (1) 全 emb 中央値で暫定 centroid → (2) 暫定コサインが中央値以上 (外側を切った頑健群) で引き直す。
    scores["emb"] から計算するので run/rescore どちらでも最新の採用候補に対して最新化される。
    """
    rows = [r for r in records if r.scores.get("emb")]
    if (
        len(rows) < 3
    ):  # 母集団が小さすぎると centroid が無意味 → 判定不能 (None) にしてゲート側で扱う
        for r in records:
            r.scores["speaker_sim"] = None
        return
    Z = np.array([r.scores["emb"] for r in rows], dtype=np.float64)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
    c0 = np.median(Z, axis=0)
    c0 /= np.linalg.norm(c0) + 1e-12
    sims0 = Z @ c0
    robust = Z[
        sims0 >= np.median(sims0)
    ]  # 暫定 centroid から遠い外れ値 (別人/ノイズ) を落として引き直す
    centroid = np.median(robust, axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-12
    for z, r in zip(Z, rows, strict=True):
        r.scores["speaker_sim"] = round(float(z @ centroid), 3)
