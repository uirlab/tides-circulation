"""Helper functions extracted verbatim from src/prep_xfer.py (full, canon, xfer, cells) for src/prep_xfer_rb.py."""
import zipfile

import pandas as pd

R = "data/raw/youbike/"


def full(month):
    with zipfile.ZipFile(f"{R}2023{month:02d}_youbike.zip") as z:
        cols = ["rent_time", "rent_station", "return_time", "return_station", "rent", "infodate"]
        has_header = z.open(z.namelist()[0]).read(9) == b"rent_time"  # June 2023 file ships without a header row
        df = pd.read_csv(z.open(z.namelist()[0]), dtype=str, header=0 if has_header else None, names=None if has_header else cols,
                         usecols=["rent_time", "rent_station", "return_station", "rent"],
                         encoding="utf-8-sig", encoding_errors="replace")  # some names are byte-truncated mid-character
    df = df.rename(columns={"rent_time": "t", "rent_station": "s", "return_station": "e", "rent": "dur"})
    df = df.dropna(subset=["t", "s", "e"])
    for c in ("s", "e"):
        df[c] = df[c].str.rstrip("\ufffd")
    return df


def canon(xdf, names):
    """Map transfer-file station names onto (possibly truncated) full-file names: exact match, else the longest full-file
    name that is a prefix of the transfer name."""
    ns = sorted((n for n in names if isinstance(n, str)), key=len, reverse=True); m = {}
    for n in set(xdf.s) | set(xdf.e):
        if n in names:
            m[n] = n
        else:
            hit = next((c for c in ns if len(c) >= 4 and n.startswith(c)), None)
            m[n] = hit if hit is not None else n
    xdf = xdf.copy(); xdf["s"] = xdf.s.map(m); xdf["e"] = xdf.e.map(m)
    miss = [n for n, v in m.items() if v not in names]
    print("transfer station names:", len(m), "unmatched:", len(miss), miss[:10])
    return xdf


def xfer(month):
    df = pd.read_csv(f"{R}xfer_2023{month:02d}.csv", dtype=str, encoding="utf-8-sig")
    df = df.rename(columns={"借車時間": "t", "借車站": "s", "還車站": "e", "租借時數": "dur"})[["t", "s", "e", "dur"]]
    df = df.dropna(subset=["t", "s", "e"])
    df["t"] = pd.to_datetime(df.t.str[:19])  # ISO with +08:00 offset -> naive local
    return df


def cells(df):
    t = pd.to_datetime(df.t)
    return pd.DataFrame(dict(h=t.dt.floor("h"), s=df.s, e=df.e)).value_counts().rename("n")
