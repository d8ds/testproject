This is test

def load_titech_lexicon(path: str) -> dict[str, float]:
    """
    東工大・単語感情極性対応表
    格式: 語(終止形):読み:品詞:スコア
    编码: Shift-JIS (CP932)
    """
    result = {}
    with open(path, encoding="cp932", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(":")
            if len(parts) < 4:
                continue
            word  = parts[0]   # 終止形
            score = float(parts[3])
            # 同一词存在多个读み → 取平均
            if word in result:
                result[word] = (result[word] + score) / 2
            else:
                result[word] = score
    return result


def load_tohoku_lexicon(noun_path: str, verb_path: str) -> dict[str, float]:
    """
    東北大・日本語評価極性辞書（名詞編 + 用言編）
    格式: 単語\tp/e/n\t動詞句  (tab分隔)
    商用利用可
    """
    label_map = {"p": 1.0, "e": 0.0, "n": -1.0}
    result = {}
    for path in [noun_path, verb_path]:
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                word  = parts[0]
                label = parts[1]
                result[word] = label_map.get(label, 0.0)
    return result


def load_lexicon(
    titech_path: str  = "pn_ja.dic",
    noun_path:   str  = "pn.csv.m3.120408.trim",
    verb_path:   str  = "wago.121808.pn",
    prefer:      str  = "titech",   # "titech" or "tohoku"
) -> dict[str, float]:
    """
    合并两个辞书 + domain vocab
    prefer 参数决定重叠词时哪个辞书优先
    """
    titech  = load_titech_lexicon(titech_path)  if Path(titech_path).exists()  else {}
    tohoku  = load_tohoku_lexicon(noun_path, verb_path) \
              if Path(noun_path).exists() and Path(verb_path).exists() else {}

    if prefer == "titech":
        merged = {**tohoku, **titech}   # titech覆盖tohoku
    else:
        merged = {**titech, **tohoku}   # tohoku覆盖titech

    merged.update(DOMAIN_VOCAB)         # 领域词最高优先级
    return merged
