import awkward as ak

# eta definitions can be used later
ETA_RANGES = {
    "barrel": (0.0, 1.4),
    "endcap": (1.5, 2.4),
    "endcapNoTk": (2.5, 3.0),
    "hf": (3.0, 5.0),
}

def build_objects(events):
    """
    Central mapping of collections.
    Works for both:
      - PF/PUPPI/GenCands files
      - GenEl/TkEleL2-only files

    Also normalizes TkEleL2:
      - tkele.pt     := tkele.ptCorr
      - tkele.ptRaw  := original tkele.pt  (piero regression)
    """
    obj = {}

    if hasattr(events, "L1PFCands"):
        obj["pf"] = events.L1PFCands
    if hasattr(events, "L1PuppiCands"):
        obj["puppi"] = events.L1PuppiCands
    if hasattr(events, "GenCands"):
        obj["gen"] = events.GenCands

    if hasattr(events, "GenEl"):
        obj["genel"] = events.GenEl

    if hasattr(events, "TkEleL2"):
        tkele = events.TkEleL2

        # Normalize pt: use ptCorr as "pt" everywhere downstream if exists (piero regression)
        if hasattr(tkele, "ptCorr"):
            # preserve raw pt too (handy)
            if hasattr(tkele, "pt"):
                tkele = ak.with_field(tkele, tkele.pt, "ptRaw")
            tkele = ak.with_field(tkele, tkele.ptCorr, "pt")

        obj["tkele"] = tkele

    return obj


def cut_range(events, obj, key, var, vmin=None, vmax=None, doAbs=False, *, skip_if_missing=True):
    """
    Apply an object-level cut:
      obj[key] = obj[key][mask]

    - key: collection name in obj (e.g. 'tkele', 'genel', 'pf')
    - var: field name (e.g. 'pt', 'eta', 'charge')
    - vmin/vmax: numeric bounds; either can be None
    - doAbs: apply abs() before bounds (useful for eta)
    """
    if key not in obj:
        if skip_if_missing:
            return obj
        raise KeyError(f"cut_range: key '{key}' not in obj. Available: {list(obj.keys())}")

    coll = obj[key]
    if not hasattr(coll, var):
        if skip_if_missing:
            return obj
        raise AttributeError(f"cut_range: collection '{key}' has no field '{var}'")

    vals = getattr(coll, var)
    if doAbs:
        vals = abs(vals)

    mask = ak.ones_like(vals, dtype=bool)
    if vmin is not None:
        mask = mask & (vals >= vmin)
    if vmax is not None:
        mask = mask & (vals < vmax)

    out = dict(obj)
    out[key] = coll[mask]
    return out


def cut_equal(events, obj, key, var, value, doAbs=False, *, skip_if_missing=True):
    if key not in obj:
        if skip_if_missing:
            return obj
        raise KeyError(f"cut_equal: key '{key}' not in obj")

    coll = obj[key]
    if not hasattr(coll, var):
        if skip_if_missing:
            return obj
        raise AttributeError(f"cut_equal: '{key}' has no '{var}'")

    vals = getattr(coll, var)
    if doAbs:
        vals = abs(vals)

    mask = (vals == value)
    out = dict(obj)
    out[key] = coll[mask]
    return out


def apply_evt_mask(obj, evt_mask):
    return {k: coll[evt_mask] for k, coll in obj.items()}
