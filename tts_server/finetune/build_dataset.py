#!/usr/bin/env python3
"""Build a single-speaker TTS fine-tuning dataset for Yuumi (base skin, League of Legends).

Source: https://wiki.leagueoflegends.com/en-us/Yuumi/Audio  (MediaWiki page with one
<audio> element per voice line).  The live wiki sits behind a JS challenge, so this
script reads the page HTML from a local cache (raw/_page.html, saved from the same
page) and falls back to the official API / the identical Fandom mirror of the page
when the cache is absent.  Audio bytes are fetched from the wiki's file CDN.

Outputs (under OUT):
  raw/<original file>          downloaded original ogg files
  wavs/yuumi_0001.wav ...     24 kHz mono 16-bit PCM, silence-trimmed, peak-normalised
  train_raw.jsonl              {"audio":..., "text":..., "ref_audio":...}
  ref.wav                      reference clip (same ref_audio for every row)
  metadata.csv                 file,duration_seconds,section,text,source_url
  excluded.csv                 text,reason,source_url
"""

from __future__ import annotations

import argparse
import collections
import csv
import html as htmllib
import json
import os
import re
import shutil
import sys
import time
import unicodedata
from urllib.parse import unquote, urlsplit

import numpy as np
import requests
import soundfile as sf
from scipy.signal import resample_poly

# --------------------------------------------------------------------------- paths
OUT = r"C:\Users\saval\voice_data\yuumi"
RAW = os.path.join(OUT, "raw")
WAVS = os.path.join(OUT, "wavs")
PAGE_CACHE = os.path.join(RAW, "_page.html")
TRAIN_JSONL = os.path.join(OUT, "train_raw.jsonl")
META_CSV = os.path.join(OUT, "metadata.csv")
EXCL_CSV = os.path.join(OUT, "excluded.csv")
REF_WAV = os.path.join(OUT, "ref.wav")

PAGE_URL = "https://wiki.leagueoflegends.com/en-us/Yuumi/Audio"
OFFICIAL_API = "https://wiki.leagueoflegends.com/en-us/api.php"
FANDOM_API = "https://leagueoflegends.fandom.com/api.php"
WAYBACK_CDX = "http://web.archive.org/cdx/search/cdx"

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
HEADERS = {"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"}

SR = 24000
PAD = 0.1            # seconds of silence kept at each end
PEAK_DBFS = -1.0
DOWNLOAD_DELAY = 0.2

# Chosen reference line (base/original Yuumi, clean ordinary delivery, 4-10 s):
# "In case you're wondering what that smell is, it's Book. Smells like old trees."
# (Movement / Long Move, 4.99 s, no exclamations, no sound effects).
# None -> automatic selection (closest to 6 s among calm 4-10 s lines).
REF_FILENAME = "Yuumi_Original_MoveLong_Attached_6.ogg"

# --------------------------------------------------------------------------- text

ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff"), None)
CURLY = {"\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
         "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"'}
NONVERBAL_PARENS = re.compile(r"\((?:whispering|puffing|purring|meowing|hissing)\)", re.I)
# Sentences the wiki writes outside quotes to describe the delivery, e.g.
#   "Today we are destroyers!" Yuumi giggles.
# Only applied to text outside double quotes, so dialogue is never touched.
NARRATION = re.compile(
    r"(?:^|(?<=[.!?] ))(?:Yuumi|Book|She)\s+"
    r"(?:giggles|yawns|snores[^.!?\"]*|gasps|splutters|coughs|hisses|sniffs[^.!?\"]*|"
    r"grunts|yelps|yowls|laughs[^.!?\"]*)\.", re.I)


def strip_markup(fragment: str) -> str:
    """HTML fragment -> plain text, dropping player chrome (Link, play icon, refs)."""
    s = re.sub(r"<audio\b.*?</audio>", " ", fragment, flags=re.S | re.I)
    s = re.sub(r"<sup\b.*?</sup>", " ", s, flags=re.S)
    s = re.sub(r"<a\b[^>]*class=\"[^\"]*ext-audiobutton[^\"]*\"[^>]*>.*?</a>", " ", s, flags=re.S)
    s = re.sub(r"<[^>]+>", " ", s)
    s = htmllib.unescape(s)
    s = s.translate(ZERO_WIDTH)
    s = s.replace("\u00a0", " ").replace("\u266a", " ").replace("\u25b6\ufe0f", " ")
    s = s.replace("\u2026", "...")
    for a, b in CURLY.items():
        s = s.replace(a, b)
    s = unicodedata.normalize("NFC", s)
    return re.sub(r"\s+", " ", s).strip()


def clean_transcript(raw_text: str) -> str:
    """Turn a wiki table entry into the spoken transcript, or '' if non-speech."""
    s = strip_markup(raw_text).strip()
    if not s:
        return ""
    s = NONVERBAL_PARENS.sub(" ", s)

    # Drop narrating sentences, but only in the unquoted parts of the entry
    # (split on '"' -> odd indices are inside quotes).
    parts = s.split('"')
    for i in range(0, len(parts), 2):
        parts[i] = NARRATION.sub(" ", parts[i].strip())
    s = " ".join(parts).replace('"', " ")

    # Protect runs of dots so the space-before-punctuation pass cannot eat the
    # space in "friend? ...Book".
    dots = "\u0001"
    s = re.sub(r"\.{2,}", lambda m: dots * len(m.group(0)), s)

    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+([,!?;:])", r"\1", s)                   # "none , Book" -> "none, Book"
    s = re.sub(r"(\w)\s+'(\s*[,.;:!?])", r"\1'\2", s)       # "'cat Aspect '" -> "'cat Aspect'"
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"[~\u223c]+$", "", s).strip()
    s = re.sub(r"[\s,]*\bPurr$", "", s).strip()             # trailing purr marker
    s = s.replace(dots, ".")
    s = re.sub(r"'(\s*[!?.,]+)$", r"\1", s)                 # "'Cat-tack'!" -> "Cat-tack!"
    s = re.sub(r"([!?])\1+", r"\1", s)                      # "magic!!" -> "magic!"
    s = re.sub(r"^[\s\"'\-]+|[\s\"]+$", "", s).strip()
    if not s or re.fullmatch(r"(?i)(sound effect|sfx)\.?", s):
        return ""
    if re.fullmatch(r"(?i)yuumi\s+\w+(?:\s+\w+){0,4}\.", s):
        # Defensive: a stage direction that slipped through.
        return ""
    return s


# --------------------------------------------------------------------------- page

def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def _wayback_latest(session: requests.Session) -> str | None:
    try:
        r = session.get(WAYBACK_CDX, params={"url": PAGE_URL, "output": "json",
                                            "filter": "statuscode:200", "limit": "-3"},
                        timeout=60)
        rows = r.json()
        if len(rows) > 1:
            ts = rows[-1][1]
            return f"https://web.archive.org/web/{ts}id_/{PAGE_URL}"
    except Exception as exc:  # noqa: BLE001 - best effort fallback
        print(f"  wayback lookup failed: {exc}")
    return None


def fetch_page_html(force: bool = False) -> str:
    """Return the page HTML, using raw/_page.html as cache."""
    if os.path.exists(PAGE_CACHE) and not force:
        print(f"page: using cache {PAGE_CACHE}")
        return open(PAGE_CACHE, encoding="utf-8").read()

    session = _session()
    html = None
    sources = [("official API", OFFICIAL_API,
                {"action": "parse", "page": "Yuumi/Audio", "prop": "text", "format": "json"})]
    for name, url, params in sources:
        try:
            r = session.get(url, params=params, timeout=60)
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/json"):
                html = r.json()["parse"]["text"]["*"]
                print(f"page: fetched from {name}")
                break
            print(f"page: {name} returned {r.status_code}")
        except Exception as exc:  # noqa: BLE001
            print(f"page: {name} failed: {exc}")

    if html is None:
        wb = _wayback_latest(session)
        if wb:
            try:
                r = session.get(wb, timeout=90)
                r.raise_for_status()
                html = r.text
                print(f"page: fetched Wayback snapshot {wb}")
            except Exception as exc:  # noqa: BLE001
                print(f"page: wayback fetch failed: {exc}")

    if html is None:
        r = session.get(FANDOM_API, params={"action": "parse", "page": "Yuumi/LoL/Audio",
                                            "prop": "text", "format": "json"}, timeout=60)
        r.raise_for_status()
        html = r.json()["parse"]["text"]["*"]
        print("page: fetched from Fandom mirror (same content)")

    os.makedirs(RAW, exist_ok=True)
    with open(PAGE_CACHE, "w", encoding="utf-8") as fh:
        fh.write(html)
    return html


def parse_entries(html_text: str) -> list[dict]:
    """One dict per <li> that contains an audio element, with section + filename."""
    entries: list[dict] = []
    h2 = h3 = ""
    for m in re.finditer(r"<h([23])\b.*?</h\1>|<li\b.*?</li>", html_text, flags=re.S):
        tok = m.group(0)
        if tok.startswith("<h"):
            title = strip_markup(re.sub(r'<span class="mw-editsection".*', "", tok, flags=re.S))
            if tok[2] == "2":
                h2, h3 = title, ""
            else:
                h3 = title
            continue
        if "<audio" not in tok:
            continue
        srcs = re.findall(r'<source\b[^>]*\bsrc="([^"]+)"', tok)
        if not srcs:
            continue
        # Official wiki: "/en-us/images/Yuumi_Select.ogg?711ef"
        # Fandom CDN:   ".../images/2/2a/Yuumi_Select.ogg/revision/latest?cb=..."
        path = urlsplit(srcs[0]).path
        if "/revision/" in path:
            path = path.split("/revision/")[0]
        filename = unquote(path.rsplit("/", 1)[-1])
        entries.append({
            "section": f"{h2} / {h3}" if h3 else h2,
            "filename": filename,
            "raw_text": strip_markup(tok),
            "page_src": srcs[0],
        })
    return entries


# --------------------------------------------------------------------------- files

def resolve_file_urls(session: requests.Session, filenames: list[str]) -> dict[str, str]:
    urls: dict[str, str] = {}
    for i in range(0, len(filenames), 50):
        batch = filenames[i:i + 50]
        r = session.get(FANDOM_API, params={
            "action": "query", "titles": "|".join("File:" + f for f in batch),
            "prop": "imageinfo", "iiprop": "url|size|mime", "format": "json"}, timeout=60)
        r.raise_for_status()
        for page in r.json()["query"]["pages"].values():
            if "missing" in page or "imageinfo" not in page:
                continue
            key = page["title"][5:].replace(" ", "_")
            urls[key] = page["imageinfo"][0]["url"]
    return urls


def download(session: requests.Session, filename: str, url: str) -> str:
    dest = os.path.join(RAW, filename)
    if os.path.exists(dest) and os.path.getsize(dest) > 1024:
        return dest
    r = session.get(url, headers={"Referer": "https://leagueoflegends.fandom.com/"}, timeout=90)
    r.raise_for_status()
    if not r.content.startswith(b"OggS"):
        raise RuntimeError(f"{filename}: not an Ogg file ({r.content[:8]!r})")
    tmp = dest + ".part"
    with open(tmp, "wb") as fh:
        fh.write(r.content)
    os.replace(tmp, dest)
    time.sleep(DOWNLOAD_DELAY)
    return dest


# --------------------------------------------------------------------------- audio

def load_mono(path: str) -> tuple[np.ndarray, int]:
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    x = data.mean(axis=1)
    if not np.all(np.isfinite(x)):
        x = np.nan_to_num(x)
    if sr != SR:
        g = np.gcd(int(sr), SR)
        x = resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
        sr = SR
    x = x - float(np.mean(x))
    return x, sr


def trim_silence(x: np.ndarray, sr: int, pad: float = PAD,
                 rel: float = 0.005, floor: float = 1e-3) -> np.ndarray:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= 0.0:
        return x
    thr = max(floor, peak * rel)
    idx = np.flatnonzero(np.abs(x) > thr)
    if idx.size == 0:
        return x
    start = max(0, int(idx[0]) - int(pad * sr))
    end = min(len(x), int(idx[-1]) + int(pad * sr) + 1)
    return x[start:end]


def process(src: str, dest: str) -> dict:
    x, sr = load_mono(src)
    x = trim_silence(x, sr)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak <= 0.0:
        raise RuntimeError(f"{src}: silent clip")
    x = (x * (10 ** (PEAK_DBFS / 20.0) / peak)).astype(np.float32)
    sf.write(dest, x, sr, subtype="PCM_16")
    info = sf.info(dest)
    return {
        "duration": info.frames / info.samplerate,
        "samplerate": info.samplerate,
        "channels": info.channels,
        "subtype": info.subtype,
        "rms_db": 20 * np.log10(max(float(np.sqrt(np.mean(x ** 2))), 1e-9)),
    }


# --------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="parse and report, download nothing")
    ap.add_argument("--refresh-page", action="store_true", help="ignore raw/_page.html cache")
    args = ap.parse_args()

    os.makedirs(RAW, exist_ok=True)
    os.makedirs(WAVS, exist_ok=True)

    html_text = fetch_page_html(force=args.refresh_page)
    entries = parse_entries(html_text)
    print(f"page entries with audio: {len(entries)}")

    kept: list[dict] = []
    excluded: list[dict] = []
    seen_files: set[str] = set()

    for idx, e in enumerate(entries, 1):
        text = clean_transcript(e["raw_text"])
        item = {**e, "index": idx, "text": text}
        if e["section"].startswith("Trivia"):
            excluded.append({**item, "reason": "Trivia section: quote used as an explanatory reference, not a voice line"})
            continue
        if e["raw_text"].strip().lower() in {"sound effect", "sfx", "sound effect."}:
            excluded.append({**item, "reason": "sound-effect entry, no spoken words"})
            continue
        if not text:
            excluded.append({**item, "reason": "stage direction / non-speech only (no spoken words)"})
            continue
        if e["filename"] in seen_files:
            excluded.append({**item, "reason": f"duplicate audio file ({e['filename']}) already used for an earlier line"})
            continue
        seen_files.add(e["filename"])
        kept.append(item)

    print(f"kept {len(kept)}, excluded {len(excluded)}")

    if args.dry_run:
        for it in kept:
            print(f"K {it['index']:3d} [{it['section']}] {it['text']}")
        for it in excluded:
            print(f"X {it['index']:3d} [{it['section']}] {it['raw_text']}  <-- {it['reason']}")
        return 0

    session = _session()
    urls = resolve_file_urls(session, [it["filename"] for it in kept])
    missing = [it["filename"] for it in kept if it["filename"] not in urls]
    if missing:
        raise SystemExit(f"no file URL for: {missing}")

    for n, it in enumerate(kept, 1):
        src = download(session, it["filename"], urls[it["filename"]])
        wav_name = f"yuumi_{n:04d}.wav"
        dest = os.path.join(WAVS, wav_name)
        info = process(src, dest)
        it.update({"wav": wav_name, "wav_path": os.path.abspath(dest),
                   "audio_url": urls[it["filename"]], **info})
        print(f"  [{n:3d}/{len(kept)}] {wav_name} {info['duration']:5.2f}s  {it['text'][:70]}")

    # --- reference clip -----------------------------------------------------
    ref_item = None
    if REF_FILENAME:
        ref_item = next((it for it in kept if it["filename"] == REF_FILENAME), None)
    if ref_item is None:
        cands = [it for it in kept
                 if 4.0 <= it["duration"] <= 10.0
                 and "!" not in it["text"] and "?" not in it["text"]
                 and len(it["text"].split()) >= 5
                 and not re.search(r"\b[A-Z]{4,}\b", it["text"])]
        if not cands:
            cands = [it for it in kept if 4.0 <= it["duration"] <= 10.0]
        if not cands:
            cands = sorted(kept, key=lambda it: abs(it["duration"] - 6.0))[:5]
        ref_item = min(cands, key=lambda it: abs(it["duration"] - 6.0))
    shutil.copyfile(ref_item["wav_path"], REF_WAV)
    print(f"reference: {ref_item['wav']} {ref_item['duration']:.2f}s "
          f'[{ref_item["section"]}] "{ref_item["text"]}"')

    # --- write jsonl --------------------------------------------------------
    with open(TRAIN_JSONL, "w", encoding="utf-8") as fh:
        for it in kept:
            fh.write(json.dumps({"audio": it["wav_path"], "text": it["text"],
                                 "ref_audio": os.path.abspath(REF_WAV)}, ensure_ascii=False) + "\n")

    # --- write metadata -----------------------------------------------------
    with open(META_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "duration_seconds", "section", "text", "source_url"])
        for it in kept:
            w.writerow([it["wav"], f"{it['duration']:.3f}", it["section"], it["text"], it["audio_url"]])

    with open(EXCL_CSV, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["text", "reason", "source_url"])
        for it in excluded:
            src = it["page_src"]
            if src.startswith("/"):
                src = "https://wiki.leagueoflegends.com" + src
            w.writerow([re.sub(r"\s+([,.;:!?])", r"\1", it["raw_text"]), it["reason"], src])

    # --- summary ------------------------------------------------------------
    durs = np.array([it["duration"] for it in kept])
    print("\n=== summary ===")
    print(f"kept lines      : {len(kept)}")
    print(f"excluded entries: {len(excluded)}")
    print(f"total duration  : {durs.sum() / 60:.2f} min ({durs.sum():.1f} s)")
    print(f"min/median/max  : {durs.min():.2f}s / {np.median(durs):.2f}s / {durs.max():.2f}s")
    print(f"exclusion reasons: {dict(collections.Counter(e['reason'] for e in excluded))}")
    print(f"train_raw.jsonl : {TRAIN_JSONL}")
    print(f"metadata.csv    : {META_CSV}")
    print(f"excluded.csv    : {EXCL_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
