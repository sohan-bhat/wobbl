"""PubChem's public REST service, for names and formulas the offline table does not know.

Requests go to pubchem.ncbi.nlm.nih.gov/rest/pug, a few at a time, and every answer (found or not) is kept in
~/.wobbl/cache/pubchem so the same name never asks twice. WOBBL_OFFLINE=1 turns PubChem off, and any network
failure counts as offline, so resolution falls back to the table and to what the text itself says.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

from ..solvers.locate import wobbl_home

BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
PROPS = "Title,MolecularFormula,MolecularWeight,SMILES,ConnectivitySMILES,IUPACName,Charge,InChI,InChIKey"


class Offline(RuntimeError):
    pass


def online() -> bool:
    return os.environ.get("WOBBL_OFFLINE", "").strip().lower() not in ("1", "yes", "true", "on")


def _cache(key: str):
    d = wobbl_home() / "cache" / "pubchem"
    return d / (hashlib.sha1(key.encode()).hexdigest()[:20] + ".json")


def _get(path: str, raw: bool = False):
    """The JSON at a PubChem path, None when PubChem says it has no such record. Raises Offline."""
    if not online():
        raise Offline("PubChem is turned off with WOBBL_OFFLINE")
    c = _cache(path)
    if c.exists():
        d = json.loads(c.read_text())
        return d.get("v")
    url = f"{BASE}/{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "wobbl (github.com/sohan-bhat/wobbl)"})
    from ..setup.installer import _ssl_context
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=15, context=_ssl_context()) as r:
                body = r.read().decode()
            v = body if raw else json.loads(body)
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                v = None
                break
            if e.code in (503, 429) and attempt < 2:      # PubChem asks clients to slow down
                time.sleep(1.5 * (attempt + 1))
                continue
            raise Offline(f"PubChem answered {e.code}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise Offline(f"PubChem could not be reached ({e})") from None
    c.parent.mkdir(parents=True, exist_ok=True)
    c.write_text(json.dumps({"v": v, "url": url}))
    return v


def _props(kind: str, value: str) -> list[dict]:
    q = urllib.parse.quote(value, safe="")
    d = _get(f"compound/{kind}/{q}/property/{PROPS}/JSON")
    return (d or {}).get("PropertyTable", {}).get("Properties", [])


def by_name(name: str) -> dict | None:
    hits = _props("name", name)
    return hits[0] if hits else None


def by_smiles(smiles: str) -> dict | None:
    hits = _props("smiles", smiles)
    return hits[0] if hits and hits[0].get("CID") else None


def by_inchi(inchi: str) -> dict | None:
    d = _get("compound/inchi/property/" + PROPS + "/JSON?inchi=" + urllib.parse.quote(inchi, safe=""))
    hits = (d or {}).get("PropertyTable", {}).get("Properties", [])
    return hits[0] if hits else None


def by_formula(formula: str, limit: int = 5) -> list[dict]:
    """Compounds with exactly this formula, most referenced first as PubChem orders them."""
    d = _get(f"compound/fastformula/{urllib.parse.quote(formula)}/cids/JSON?MaxRecords={limit}")
    cids = (d or {}).get("IdentifierList", {}).get("CID", [])[:limit]
    if not cids:
        return []
    return _props("cid", ",".join(str(c) for c in cids))


def sdf3d(cid: int) -> str | None:
    """PubChem's computed 3D structure of a compound as SDF text, when it has one."""
    return _get(f"compound/cid/{cid}/SDF?record_type=3d", raw=True)
