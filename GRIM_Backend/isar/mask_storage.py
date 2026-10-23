"""Preflight and stream Mask Cut arrays into managed result storage."""
import json
import zipfile

import numpy as np

from GRIM_Backend.datasets.result_storage import allocate_result, operation_result_storage
from GRIM_Backend.isar.artifact import _read_npy_member_header
from GRIM_Backend.plotting.requests import PlotCancelled


def read_mask_archive(path, *, cancel=None, maximum_bytes=16 * 1024**3):
    def checkpoint():
        if cancel is not None and cancel():
            raise PlotCancelled()

    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if len(entries) > 64 or len(set(names)) != len(names):
            raise ValueError("Mask image has too many or duplicate archive members.")
        if not {"manifest.npy", "x.npy", "y.npy", "magnitude.npy"} <= set(names):
            raise ValueError("Mask image is missing required arrays or metadata.")
        headers, total = {}, 0
        for entry in entries:
            checkpoint()
            name = entry.filename
            if (not name.endswith('.npy') or '/' in name or '\\' in name
                    or name[:-4] not in {'manifest', 'x', 'y', 'magnitude', 'complex_image'}
                    and not name.startswith('inverse_')):
                raise ValueError("Invalid Mask Cut archive member.")
            with archive.open(entry) as stream:
                shape, fortran, dtype, cells, size, offset = _read_npy_member_header(stream, name)
            if offset + size != entry.file_size:
                raise ValueError("Mask image array size disagrees with its header.")
            if name == 'manifest.npy':
                if shape != () or dtype.kind not in 'US' or size > 16 * 1024**2:
                    raise ValueError("Invalid or oversized Mask Cut metadata.")
            else:
                if dtype.kind not in 'biufc' or not 1 <= len(shape) <= 2 or not cells:
                    raise ValueError("Mask image arrays must be nonempty numeric vectors or matrices.")
                total += size
                if total > maximum_bytes:
                    raise ValueError("Mask image exceeds the numerical payload limit.")
            headers[name[:-4]] = (shape, fortran, dtype, offset)

        with archive.open('manifest.npy') as stream:
            manifest = json.loads(str(np.lib.format.read_array(stream, allow_pickle=False)))
        if (not isinstance(manifest, dict) or manifest.get('format') != 'GRIM-mask-image'
                or manifest.get('version') != 1 or not isinstance(manifest.get('metadata'), dict)
                or not isinstance(manifest.get('name'), str)
                or manifest.get('inverse') is not None and not isinstance(manifest['inverse'], dict)):
            raise ValueError("This is not a supported GRIM Mask Cut image dataset.")

        # Validate the principal shapes before any scientific arrays are allocated.
        xshape, yshape, mshape = (headers[key][0] for key in ('x', 'y', 'magnitude'))
        if (len(xshape) != 1 or len(yshape) != 1 or min(xshape[0], yshape[0]) < 2
                or mshape != (xshape[0], yshape[0])
                or any(headers[key][2].kind not in 'iuf' for key in ('x', 'y', 'magnitude'))):
            raise ValueError("Invalid image axes or magnitude shape.")
        if 'complex_image' in headers and (headers['complex_image'][0] != mshape
                                           or headers['complex_image'][2].kind != 'c'):
            raise ValueError("Invalid complex image shape or type.")

        arrays = {}
        with operation_result_storage('auto'):
            for key, (shape, fortran, dtype, offset) in headers.items():
                if key == 'manifest':
                    continue
                checkpoint()
                result = allocate_result(shape, dtype, order='F' if fortran else 'C')
                raw = memoryview(result.ravel(order='K').view(np.uint8))
                with archive.open(key + '.npy') as stream:
                    stream.read(offset)
                    position = 0
                    while position < len(raw):
                        checkpoint()
                        stop = min(position + 8 * 1024**2, len(raw))
                        count = stream.readinto(raw[position:stop])
                        if not count:
                            raise ValueError("Truncated Mask Cut array.")
                        position += count
                arrays[key] = result
        return manifest, arrays
