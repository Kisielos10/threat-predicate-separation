"""ECML/PKDD 2007 Discovery Challenge HTTP corpus: loader and parser.

Second web corpus, used so that no claim in the paper rests on a single dataset. It is a useful
companion to CSIC-2010 because it differs in the ways that matter here: the traffic is synthesised
around randomised hosts and paths rather than one shop application, the attack classes are curated
by the challenge organisers rather than derived heuristically, and the normal traffic is far less
homogeneous.

Record format in the distributed files is a flat text stream rather than XML despite the file
names:

    Start - Id: 396
    class: Valid
    GET /path?query HTTP/1.1
    Header: value
    ...
    <blank line>
    <optional body>
    End - Id: 396

Classes are 'Valid' plus seven attack families. The two distributed files differ in what they
contain, which is easy to get wrong: `xml_train.txt` holds 24,504 records that are ALL Valid (it
is the normal-behaviour learning set), while `xml_test.txt` holds the labelled mixture used here,
10,502 Valid against 15,110 attacks across OsCommanding, PathTransversal, LdapInjection,
XPathInjection, SqlInjection, SSI and XSS. The default path is therefore the test file; the train
file is useful as an additional pool of normal traffic.

Unlike CSIC-2010, where the attack families in this project are derived heuristically from payload
patterns, these class labels come from the challenge organisers. Per-family claims on this corpus
therefore rest on curated labels.
"""

from __future__ import annotations

import os
import re

import pandas as pd

_START = re.compile(r"^Start - Id:\s*(\d+)", re.M)
_CLASS = re.compile(r"^class:\s*(.+)$", re.M)


def _records(text: str):
    """Split the stream into (id, class, raw_request) triples."""
    starts = [(m.start(), m.group(1)) for m in _START.finditer(text)]
    for i, (pos, rid) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        block = text[pos:end]
        cm = _CLASS.search(block)
        if not cm:
            continue
        cls = cm.group(1).strip()
        # the request begins after the class line and ends before any trailing "End - Id" marker
        body = block[cm.end():]
        body = re.split(r"^End - Id:.*$", body, flags=re.M)[0]
        yield rid, cls, body.strip("\n")


def load_ecml(path: str = "data/raw_ecml/xml_test.txt", n_normal: int | None = None,
              n_anomalous: int | None = None, seed: int = 17) -> pd.DataFrame:
    """Return a frame with the same shape as `load_csic`: columns `requests` and `label`.

    label 0 = Valid, 1 = any attack class. The original class name is kept in `attack_class` so
    per-family analysis stays possible.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; see data/raw_ecml/README.md")
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()

    rows = [{"requests": raw, "attack_class": cls,
             "label": 0 if cls.lower() == "valid" else 1}
            for _, cls, raw in _records(text) if raw]
    df = pd.DataFrame(rows)

    rng = pd.Series(range(len(df))).sample(frac=1.0, random_state=seed).to_numpy()
    df = df.iloc[rng].reset_index(drop=True)
    if n_normal is not None:
        norm = df[df["label"] == 0].head(n_normal)
        atk = df[df["label"] == 1].head(n_anomalous if n_anomalous is not None else len(df))
        df = pd.concat([norm, atk]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df
