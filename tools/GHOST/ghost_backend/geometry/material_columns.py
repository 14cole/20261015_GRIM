"""Explicit material CSV column mapping, also shipped in the companion tool.

Keep this small module identical in FREDDY and GHOST so each can run standalone.
Interoperability tests exercise both readers against the same input files.
"""
import re
import unicodedata


def _key(label):
    text = unicodedata.normalize("NFKC", str(label)).casefold().strip()
    text = text.replace("ε", "eps").replace("μ", "mu").replace("ω", "ohm")
    return re.sub(r"[\s_.\-\[\]()/]+", "", text)


def material_column_order(columns, expected):
    """Return canonical column indices and the explicitly declared Hz scale."""
    aliases = {}
    for quantity, names in (("eps", ("eps", "epsr", "epsilonr", "relativepermittivity")),
                            ("mu", ("mu", "mur", "relativepermeability"))):
        for component in ("real", "imag"):
            for name in names:
                aliases[name + component] = quantity + "_" + component
    for quantity in ("resistance", "reactance"):
        for unit in ("ohm", "ohms"):
            aliases[quantity + unit] = quantity + "_ohm"
    canonical, scales = [], []
    for column in columns:
        # Preserve the one case distinction that changes a supported unit's
        # physical meaning: mHz is millihertz, whereas MHz is megahertz.
        unit_label = re.sub(r"[\s_.\-\[\]()/]+", "", str(column))
        if re.fullmatch(r"(?i:frequency|freq|f)mHz", unit_label):
            raise ValueError("Material CSV header uses mHz (millihertz); use MHz for megahertz or convert to Hz.")
        key = _key(column)
        match = re.fullmatch(r"(?:frequency|freq|f)(hz|khz|mhz|ghz)", key)
        if match:
            canonical.append("frequency_hz")
            scales.append({"hz": 1., "khz": 1.e3, "mhz": 1.e6, "ghz": 1.e9}[match.group(1)])
        else:
            canonical.append(aliases.get(key))
    if len(columns) != len(expected) or len(scales) != 1 or set(canonical) != set(expected):
        unknown = [str(column) for column, name in zip(columns, canonical) if name not in expected]
        missing = [name for name in expected if name not in canonical]
        repeated = [name for name in expected if canonical.count(name) > 1]
        details = []
        if unknown:
            details.append("unrecognized columns: " + ", ".join(unknown))
        if missing:
            details.append("missing: " + ", ".join(missing))
        if repeated:
            details.append("duplicate: " + ", ".join(repeated))
        raise ValueError("Material CSV header must identify exactly " + ",".join(expected)
                         + ". Column order, case, spaces and brackets may vary; frequency must "
                         "declare Hz, kHz, MHz or GHz, and impedance must declare ohms. "
                         + "; ".join(details))
    return [canonical.index(name) for name in expected], scales[0]
