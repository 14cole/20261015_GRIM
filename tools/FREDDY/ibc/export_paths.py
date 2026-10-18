"""Consistent names for exports from captured polarization results."""
from pathlib import Path


def polarization_paths(path, polarizations):
    path = Path(path)
    pols = [str(p).upper() for p in polarizations]
    if not pols or len(set(pols)) != len(pols) or any(p not in ('TE', 'TM') for p in pols):
        raise ValueError('Export requires unique TE/TM polarizations.')
    return {p: path.with_name(f'{path.stem}_{p}{path.suffix}') if len(pols) > 1 else path for p in pols}
