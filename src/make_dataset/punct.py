"""文末の疑問符・感嘆符を第 2 系統 ASR から補う.

主 ASR (Qwen3-ASR) は日本語のほぼ全文を「。」で終える (実データ 4,994 本中 4,989 本、! は 0 本)。
一方、第 2 系統 (Whisper 系) は同じ音声に ? / ! を付けることが多い (同データで ? 324 本 / ! 161 本)。
SBV2 は本文の句読点を音素列と BERT の両方で使うので、? / ! が全部「。」に潰れると
疑問・感嘆の言い方を本文と結び付けて学習できず、推論時に ? / ! を書いても学習分布の外になる。

ここでは「両系統の読みが一致している (x_cer が小さい)」発話に限り、主 ASR の末尾「。」を
第 2 系統の文末記号に置き換える。読みが違う発話は、記号だけ借りると誤りを持ち込むので触らない。
"""

from __future__ import annotations

import re

# 第 2 系統の文末: 「…ー!」「…!?」「…? 」など。末尾の空白・長音・閉じ括弧を挟んでもよい
_TAIL = re.compile(r"([!?！？]+)[\s」』)）]*$")


def asr2_terminal_mark(asr2_text: str | None) -> str | None:
    """第 2 系統の文末記号 (半角に正規化, 最大 2 文字) を返す。無ければ None."""
    if not asr2_text:
        return None
    m = _TAIL.search(asr2_text.strip())
    if not m:
        return None
    mark = m.group(1).replace("！", "!").replace("？", "?")
    # 「!!!」「???」のような連打は 1 つに、「?!」「!?」は保つ
    mark = "".join(dict.fromkeys(mark))[:2]
    return mark


def restore_terminal_punct(
    text: str, asr2_text: str | None, x_cer: float | None, max_x_cer: float
) -> str | None:
    """主 ASR 本文の末尾「。」を第 2 系統の ? / ! に置き換えた本文を返す。対象外なら None.

    - 主本文が「。」で終わる (または文末記号が無い) ときだけ置き換える (既に ? / ! があれば触らない)
    - 両系統の読みの CER (x_cer) が max_x_cer 以下のときだけ (読みが違えば記号も信用しない)
    """
    if not text or x_cer is None or x_cer > max_x_cer:
        return None
    mark = asr2_terminal_mark(asr2_text)
    if mark is None:
        return None
    body = text.rstrip()
    if body.endswith(("!", "?", "！", "？")):
        return None
    if body.endswith("。"):
        body = body[:-1]
    if not body:
        return None
    return body + mark
