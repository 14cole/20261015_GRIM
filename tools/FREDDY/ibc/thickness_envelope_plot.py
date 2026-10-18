"""Plot captured thickness envelopes; no solver work in the display path."""
import math
import numpy as np
from .plot import style_axis


def draw_envelopes(figure,result,panel_index,pol_index,angle_index,colors=None):
    figure.clear()
    pi=max(0,pol_index)
    panels=result['panels'] if panel_index==0 else [result['panels'][panel_index-1]]
    n=len(panels);columns=2 if n>1 else 1;rows=math.ceil(n/columns)
    freqs=result['frequencies_ghz']
    nominal=result['nominal_reflection_db'][pi]
    pooled=angle_index==0
    ai=max(0,angle_index-1)
    if pooled:
        nom_lo,nom_hi=nominal.min(axis=0),nominal.max(axis=0)
    else: nom_lo=nom_hi=nominal[ai]
    angle_text=(f'all {len(result["angles_deg"])} incidence angles' if pooled and len(result['angles_deg'])>1 else
                f'{result["angles_deg"][ai]:g}° incidence')
    figure.suptitle(f'Thickness tolerance · {result["polarizations"][pi].upper()} · {angle_text}',
                   fontsize=11,color=colors['plot_text'] if colors else 'black')
    for index,panel in enumerate(panels,1):
        ax=figure.add_subplot(rows,columns,index)
        if colors: style_axis(ax,colors)
        low=panel['min_db'][pi].min(axis=0) if pooled else panel['min_db'][pi,ai]
        high=panel['max_db'][pi].max(axis=0) if pooled else panel['max_db'][pi,ai]
        ax.fill_between(freqs,low,high,color='#529ce8',alpha=.22)
        marker='o' if len(freqs)==1 else None
        ax.plot(freqs,low,color='#59c6a1',lw=1.2,marker=marker,label='Minimum reflection')
        ax.plot(freqs,high,color='#f4ad61',lw=1.3,marker=marker,label='Maximum reflection')
        nc=colors['plot_text'] if colors else '#444444'
        if np.array_equal(nom_lo,nom_hi): ax.plot(freqs,nom_hi,'--',color=nc,lw=1,marker=marker,label='Nominal')
        else:
            ax.fill_between(freqs,nom_lo,nom_hi,color=nc,alpha=.10,label='Nominal angle range')
            ax.plot(freqs,nom_lo,'--',color=nc,lw=.8);ax.plot(freqs,nom_hi,'--',color=nc,lw=.8)
        ax.axhline(result['target_db'],color=nc,ls=':',lw=.8,label='Reflection Target')
        label=panel['label']
        if len(label)>64: label=label[:61]+'…'
        thickness=panel.get('thickness_values_in')
        if thickness:
            detail=f'{thickness[0]:.5g}–{thickness[-1]:.5g} in · {panel["cases"]} cases'
        else: detail=f'{panel["cases"]:,} combinations'
        if not panel.get('nominal_in_grid', True): detail+=' · nominal is a separate reference'
        ax.set(title=label+'\n'+detail,xlabel='Frequency (GHz)',ylabel='PEC reflection (dB)')
        ax.grid(True,alpha=.2)
        ax.legend(fontsize=7,loc='best')
    if colors: figure.set_facecolor(colors['plot_bg'])
