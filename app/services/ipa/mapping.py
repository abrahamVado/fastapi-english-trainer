# app/services/ipa/mapping.py
from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import re

try:
    import eng_to_ipa as engipa
except Exception:
    engipa = None

WORD_RE = re.compile(r"[A-Za-z']+")

def tokenize(text: str) -> List[str]:
    return WORD_RE.findall(text or "")

def en_to_ipa(word: str) -> str:
    if engipa is None:
        return word
    s = engipa.convert(word)
    return s.replace(" ", "") if isinstance(s, str) else word

# ---- English IPA -> Latin-American friendly IPA ----

VOWEL_RULES = [
    ("iː","i"),("i","i"),("ɪ","i"),
    ("uː","u"),("u","u"),("ʊ","u"),
    ("eɪ","ei"),("oʊ","ou"),("əʊ","ou"),
    ("aɪ","ai"),("aʊ","au"),("ɔɪ","oi"),
    ("ɑː","a"),("ɒ","o"),("ɔː","o"),
    ("æ","a"),("ʌ","a"),
]

def _apply_vowels(s: str) -> str:
    for a,b in VOWEL_RULES: s = s.replace(a,b)
    return s

def map_to_latam(ipa: str, *, theta="t", mode="strict", r="tap", schwa="e") -> str:
    s = ipa.replace("ː","").replace("ˈ","").replace("ˌ","")
    s = s.replace("ð","d").replace("θ","s" if theta=="s" else "t")
    s = s.replace("ɚ","er").replace("ɝ","er")
    if mode == "strict":
        s = s.replace("ʃ","t͡ʃ").replace("ʒ","ʝ").replace("d͡ʒ","ʝ")
    s = s.replace("ɹ", "ɾ" if r=="tap" else "r")
    s = re.sub(r"ŋ(?=$)", "n", s)  # final ŋ -> n
    s = _apply_vowels(s).replace("ə", "a" if schwa=="a" else "e")
    return re.sub(r"\s+","", s)

def respell(latam_ipa: str) -> str:
    return (latam_ipa.replace("t͡ʃ","ch").replace("ʝ","y")
            .replace("ɾ","r").replace("ʃ","sh").replace("ʒ","y")
            .replace("ŋ","ng").replace("j","y"))

# ---- Pronunciation scoring (edit distance on phones) ----

PHONES = ["t͡ʃ","d͡ʒ","aʊ","aɪ","eɪ","oʊ","əʊ","ɔɪ","iː","uː","ɔː","ɑː",
          "æ","ʌ","ɪ","ʊ","ŋ","ʃ","ʒ","ɹ","ð","θ","ə","ɑ","ɛ","i","u","o","a","ɔ","e",
          "p","b","t","d","k","ɡ","f","v","s","z","h","m","n","l","w","j","r","ɾ","ʝ"]
PHONES_RE = re.compile("|".join(map(re.escape, PHONES)))

def split_ipa(ipa: str) -> List[str]:
    s = ipa.replace("ˈ","").replace("ˌ","").replace("ː","")
    toks = PHONES_RE.findall(s)
    return toks if toks else list(s)

def edit_ops(ref: List[str], hyp: List[str]) -> Tuple[int, List[str]]:
    n,m=len(ref),len(hyp)
    dp=[[0]*(m+1) for _ in range(n+1)]
    bt=[[None]*(m+1) for _ in range(n+1)]
    for i in range(1,n+1): dp[i][0]=i; bt[i][0]=("del", ref[i-1])
    for j in range(1,m+1): dp[0][j]=j; bt[0][j]=("ins", hyp[j-1])
    for i in range(1,n+1):
        for j in range(1,m+1):
            cost=0 if ref[i-1]==hyp[j-1] else 1
            cand=[(dp[i-1][j]+1,("del",ref[i-1])),
                  (dp[i][j-1]+1,("ins",hyp[j-1])),
                  (dp[i-1][j-1]+cost,("keep" if cost==0 else "sub",(ref[i-1],hyp[j-1])))]
            dp[i][j],bt[i][j]=min(cand,key=lambda x:x[0])
    i,j=n,m; ops=[]
    while i>0 or j>0:
        op=bt[i][j]
        if op[0]=="del": ops.append(f"del:{op[1]}"); i-=1
        elif op[0]=="ins": ops.append(f"ins:{op[1]}"); j-=1
        else:
            kind,val=op
            if kind=="keep": ops.append(f"keep:{val}")
            else: ops.append(f"sub:{val[0]}->{val[1]}")
            i-=1;j-=1
    ops.reverse()
    return dp[n][m], ops

def score_pronunciation(ref_text: str, hyp_text: str,
                        theta="t", mode="strict", r="tap", schwa="e") -> Dict[str, object]:
    ref_tokens, hyp_tokens = tokenize(ref_text), tokenize(hyp_text)
    ref_lat = [map_to_latam(en_to_ipa(w.lower()), theta=theta, mode=mode, r=r, schwa=schwa) for w in ref_tokens]
    hyp_lat = [map_to_latam(en_to_ipa(w.lower()), theta=theta, mode=mode, r=r, schwa=schwa) for w in hyp_tokens]
    L = max(len(ref_lat), len(hyp_lat))
    words_out=[]; tot=0.0; cnt=0
    for i in range(L):
        ref = ref_lat[i] if i<len(ref_lat) else ""
        hyp = hyp_lat[i] if i<len(hyp_lat) else ""
        if ref and hyp:
            dist, ops = edit_ops(split_ipa(ref), split_ipa(hyp))
            acc = 1.0 - (dist / max(len(split_ipa(ref)),1))
            tot += acc; cnt += 1
        else:
            acc, ops = 0.0, ["skip"]
        words_out.append({"idx": i, "expected_ipa": ref, "heard_ipa": hyp, "ops": ops, "score": round(acc,3)})
    overall = (tot/cnt) if cnt else 0.0
    return {
        "overall": {"score_0_100": round(overall*100), "phoneme_accuracy": round(overall,3)},
        "words": words_out
    }
