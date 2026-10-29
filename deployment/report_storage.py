"""Render the completed storage measurements as a reusable scientific figure."""
import argparse
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.runtime/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_report(out,m):
    original=sum(x['logical_bytes'] for x in m['original'].values())
    kinds=['full','encoded','on_demand']
    labels=['Original\nfiles','HDF5\narrays','HDF5\nPNG bytes','On-demand\nHDF5']
    sizes=[original]+[m['archives'][k]['logical_bytes'] for k in kinds]
    colors=['#75889b','#df9b55','#4d8ab0','#30897e']
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    ax=axes[0]
    ax.bar(labels,np.array(sizes)/2**30,color=colors,width=.65)
    for i,b in enumerate(sizes):
        change='' if i==0 else f'\n{(b/original-1)*100:+.1f}%'
        ax.text(i,b/2**30+.05,f'{b/2**30:.3f} GiB'+change,ha='center',va='bottom',fontsize=10)
    ax.set_ylim(0,max(sizes)/2**30*1.24)
    ax.set_ylabel('Logical file size (GiB)')
    ax.set_title('Results storage per slide',loc='left',fontweight='bold')
    ax=axes[1]
    data=m['retrieval']['workloads']['random']
    vals=[data[k]['median_ms_per_cell'] for k in ['png','hdf5','encoded','on-demand']]
    ax.bar(labels,vals,color=colors,width=.65)
    for i,v in enumerate(vals):
        ax.text(i,v+max(vals)*.025,f'{v:.3f}',ha='center',va='bottom',fontsize=10)
    ax.set_ylim(0,max(vals)*1.24)
    ax.set_ylabel('Milliseconds per cell (RGB + mask)')
    ax.set_title('Random cell retrieval · warm cache',loc='left',fontweight='bold')
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
    fig.suptitle(f'{m["cells"]:,} cells · {m["patches"]:,} patches · exact recovery verified',fontsize=15,fontweight='bold')
    fig.supxlabel(f'On-demand requires the original slide ({m["source_slide_bytes"]/2**20:.1f} MiB extra). '
                  f'Retrieval: {m["retrieval"]["repetitions"]} repetitions × {m["retrieval"]["cells_per_workload"]:,} cells; opening/checksum excluded.',fontsize=9)
    fig.savefig(out/'storage_comparison.png',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('output_dir',type=Path)
    args=p.parse_args()
    metrics=json.loads((args.output_dir/'storage_metrics.json').read_text())
    plot_report(args.output_dir,metrics)
