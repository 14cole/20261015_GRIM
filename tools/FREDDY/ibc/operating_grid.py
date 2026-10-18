"""Explicit grid copy actions; never silently couple independent studies."""
from PySide6.QtWidgets import QInputDialog, QPushButton


def copy_button(host, destination):
    button = QPushButton('Copy operating grid from…')
    button.setToolTip('Copy frequencies and, where supported, incidence angles from another workflow. Copying Inverse Design to Thickness also copies polarization and reflection limit. Existing results retain their captured settings.')
    def choose():
        if host.job_is_running(): return
        choices = [name for name in ('Impedance / IBC Batch', 'Off Angle', 'Thickness', 'Inverse Design', 'Material Mix') if name != destination]
        source, ok = QInputDialog.getItem(host, 'Copy operating grid', 'Source workflow', choices, 0, False)
        if ok:
            from .ui_controls import messagebox
            try: copy_grid(host, source, destination)
            except ValueError as exc: messagebox.showerror('Copy operating grid', str(exc))
    button.clicked.connect(choose)
    return button


def copy_grid(host, source, destination):
    def fields(name):
        if name == 'Thickness':
            f = host.tolerance_workspace.fields
            return ([f[k] for k in ('f_start','f_stop','f_step')], [f[k] for k in ('a_start','a_stop','a_step')], None, None)
        prefixes = {'Impedance / IBC Batch': ('f_', None, None), 'Off Angle': ('angle_f_', 'angle_', None),
                    'Inverse Design': ('inv_target_', 'inv_angle_', 'inv'), 'Material Mix': ('mix_target_', 'mix_perf_angle_', 'mix')}
        fp, ap, mode = prefixes[name]
        return ([getattr(host, fp + k + '_var') for k in ('start','stop','step')],
                [getattr(host, ap + k + '_var') for k in ('start','stop','step')] if ap else None,
                getattr(host, mode + '_freq_mode_var') if mode else None,
                getattr(host, mode + '_freq_list_var') if mode else None)
    def get(field): return field.get() if hasattr(field, 'get') else field.text()
    def put(field, text): (field.set if hasattr(field, 'set') else field.setText)(text)
    sf, sa, sm, sl = fields(source)
    df, da, dm, dl = fields(destination)
    discrete = sm is not None and get(sm).lower().startswith('discrete')
    if discrete and dm is None:
        raise ValueError('This workflow accepts a regular minimum/maximum/step grid. A discrete list cannot be copied without changing its defined frequencies.')
    if source == 'Inverse Design' and destination == 'Thickness':
        host.tolerance_workspace.use_inverse_setup()
        return
    if discrete:
        put(dl, get(sl)); put(dm, 'Discrete list')
    else:
        for src, dst in zip(sf, df): put(dst, get(src))
        if dm is not None: put(dm, 'Band sweep')
    if sa is not None and da is not None:
        for src, dst in zip(sa, da): put(dst, get(src))
