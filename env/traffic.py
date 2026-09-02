"""Traffic generator for the mock environment.

Sends a mix of benign browsing and web attacks to the target service. Each request carries an
X-Case-Id header; the ground truth is written to logs/ground_truth.json. The label lives ONLY in
that side file and in the header - never in the method/path/query/body the agents actually see.

    python env/traffic.py [base_url] [n_benign] [n_attack]
"""

from __future__ import annotations

import json
import os
import random
import sys
import urllib.error
import urllib.parse
import urllib.request

BENIGN = [
    ("GET", "/", ""), ("GET", "/products", ""), ("GET", "/login", ""),
    ("GET", "/search", "q=kawa"), ("GET", "/search", "q=herbata+zielona"),
    ("GET", "/products", "kategoria=napoje&strona=2"),
    ("POST", "/login", "user=anna&pass=Wiosna2024"),
    ("POST", "/login", "user=piotr&pass=Haslo123"),
    ("GET", "/search", "q=ekspres+do+kawy&sort=cena"),
]

ATTACKS = [
    ("sqli", "GET", "/search", "q=' OR '1'='1"),
    ("sqli", "GET", "/products", "id=1 UNION SELECT username,password FROM users--"),
    ("sqli", "POST", "/login", "user=admin&pass=' OR 1=1--"),
    ("sqli", "GET", "/products", "id=1; WAITFOR DELAY '0:0:15'--"),
    ("xss", "GET", "/search", "q=<script>alert(document.cookie)</script>"),
    ("xss", "GET", "/search", "q=<img src=x onerror=alert(1)>"),
    ("path_traversal", "GET", "/products", "file=../../../../etc/passwd"),
    ("path_traversal", "GET", "/products", "file=..%2f..%2f..%2fboot.ini"),
    ("command", "GET", "/search", "q=kawa;cat /etc/passwd"),
]


def send(base: str, method: str, path: str, query: str, case_id: str, timeout: float = 5.0) -> bool:
    url = f"{base}{path}" + (f"?{urllib.parse.quote(query, safe='=&')}" if query and method == "GET" else "")
    data = query.encode() if method == "POST" else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"X-Case-Id": case_id, "User-Agent": "zdv-traffic/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except urllib.error.HTTPError:
        return True          # 404 is still a logged request
    except Exception as e:
        print(f"  ! {case_id} failed: {e}")
        return False


def main(base: str = "http://127.0.0.1:8081", n_benign: int = 12, n_attack: int = 12,
         seed: int = 17) -> None:
    rng = random.Random(seed)
    plan = ([("benign", None, *rng.choice(BENIGN)) for _ in range(n_benign)]
            + [("attack", *rng.choice(ATTACKS)) for _ in range(n_attack)])
    rng.shuffle(plan)

    truth, sent = [], 0
    for i, (label, family, method, path, query) in enumerate(plan):
        cid = f"live-{i:03d}"
        if send(base, method, path, query, cid):
            sent += 1
            truth.append({"case_id": cid, "label": label, "family": family})
    out = os.path.join(os.path.dirname(__file__), "logs", "ground_truth.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(truth, fh, indent=2, ensure_ascii=False)
    print(f"sent {sent}/{len(plan)} requests ({n_benign} benign, {n_attack} attack) -> {out}")


if __name__ == "__main__":
    b = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8081"
    nb = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    na = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    main(b, nb, na)
