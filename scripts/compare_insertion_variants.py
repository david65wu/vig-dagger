"""Compare dense and MoE success rates on the same physical cases."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plot_insertion_analysis import KINDS,LABELS,EPOCHS


def compare(dense,moe):
    selection=dense/'analysis_selection.json'
    if selection.exists():
        selected=json.loads(selection.read_text())
        assert selected['complete']
        dense=Path(selected['analysis_root'])
        assert (dense/'complete.json').exists()
    assert json.loads((dense/'cases.json').read_text())==json.loads((moe/'cases.json').read_text())
    tables={}
    for variant,root in [('Dense',dense),('MoE7',moe)]:
        rows=json.loads((root/'summary.json').read_text());assert len(rows)==90
        tables[variant]={(r['demos'],r['epoch'],r['case']):r for r in rows}
    fig,axes=plt.subplots(3,5,figsize=(21,11),sharex=True,sharey=True)
    for row,count in enumerate([50,200,500]):
        for col,(kind,label) in enumerate(zip(KINDS,LABELS)):
            ax=axes[row,col]
            for variant,color,marker,style in [('Dense','#2474b5','o','-'),('MoE7','#d85a32','s','--')]:
                records=[tables[variant][(count,epoch,kind)] for epoch in EPOCHS]
                assert all(r['rollouts']==10 for r in records)
                ax.plot(EPOCHS,[100*r['successes']/r['rollouts'] for r in records],
                    color=color,marker=marker,linestyle=style,linewidth=1.8,markersize=4,label=variant)
            if row==0:ax.set_title(label)
            if col==0:ax.set_ylabel(f'{count} demos\nSuccess rate (%)')
            if row==2:ax.set_xlabel('Training epoch')
            ax.set_ylim(-3,103);ax.set_yticks([0,20,40,60,80,100]);ax.set_xticks([10,50,100,200,300]);ax.grid(alpha=.2)
    fig.legend(*axes[0,0].get_legend_handles_labels(),ncol=2,loc='upper center',bbox_to_anchor=(.5,.94),frameon=False)
    fig.suptitle('Dense vs 7-level MoE: full-resolution plug insertion\nSame demonstration subsets, training schedule, and physical evaluation starts',fontsize=16)
    fig.text(.5,.015,'10 rollouts per condition and checkpoint. MoE training samples seven resolution levels; both policies are evaluated at full resolution.',ha='center')
    fig.tight_layout(rect=(0,.04,1,.89))
    for ext in ['png','pdf']:fig.savefig(moe/f'dense_vs_moe_success.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)
    combined=[dict(variant=variant,**row) for variant,table in tables.items() for row in table.values()]
    (moe/'dense_vs_moe_sources.json').write_text(json.dumps(dict(dense=str(dense),moe=str(moe)),indent=2))
    with (moe/'dense_vs_moe_summary.csv').open('w') as f:
        fields=list(dict.fromkeys(key for row in combined for key in row))
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(combined)
    page=moe/'index.html';html=page.read_text()
    block='<h2>Dense vs MoE</h2><p><a href="dense_vs_moe_success.png"><img src="dense_vs_moe_success.png" style="max-width:100%"></a></p><p><a href="dense_vs_moe_summary.csv">Comparison CSV</a></p>'
    if 'dense_vs_moe_success.png' not in html:page.write_text(html.replace('</body>',block+'</body>'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dense',type=Path,required=True);p.add_argument('--moe',type=Path,required=True)
    args=p.parse_args();compare(args.dense,args.moe)
