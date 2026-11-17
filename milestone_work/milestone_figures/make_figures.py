"""Two compact, Computer Modern figures from the copied, completed experiment records."""
import hashlib
import json
import os
from pathlib import Path

HERE=Path(__file__).resolve().parent
WORKSPACE=HERE.parent
os.environ.setdefault('MPLCONFIGDIR',str(WORKSPACE/'.cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME',str(WORKSPACE/'.cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np

COLORS={64:'#3575A9',128:'#C47B3C',256:'#458567'}
INK='#34383C'
GRID='#DEDEDE'


def load_inputs():
    paths={'validation':HERE/'inputs/validation_comparison.json',
           'test':HERE/'inputs/test_comparison.json'}
    values={k:json.loads(p.read_text()) for k,p in paths.items()}
    assert values['validation']['complete'] and values['test']['complete']
    assert len(values['validation']['per_seed'])==9
    return paths,values


def style_axes(ax,xticks,labels,ylim,yticks):
    for side in ['left','right','top','bottom']:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(.75)
        ax.spines[side].set_color(INK)
    ax.tick_params(axis='both',width=.75,length=3,color=INK,direction='out')
    ax.set_xticks(xticks,labels)
    ax.set_ylim(*ylim);ax.set_yticks(yticks)
    ax.grid(axis='y',color=GRID,linewidth=.4)
    ax.grid(axis='x',visible=False)
    ax.set_axisbelow(True)


def panel_figure(title,subtitle=''):
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.7),sharey=True)
    fig.subplots_adjust(left=.10,right=.98,bottom=.22,top=.69,wspace=.16)
    fig.suptitle(title,x=.54,y=.98,fontsize=15)
    if subtitle:
        fig.text(.54,.88,subtitle,ha='center',fontsize=10,color=INK)
    for ax,metric in zip(axes,['IC','RankIC']):
        ax.set_title(metric,fontsize=12,pad=9)
    axes[0].set_ylabel('Correlation',labelpad=7)
    return fig,axes


def main():
    paths,values=load_inputs()
    # Matplotlib ships the actual Computer Modern Roman font; no external TeX
    # installation or font substitution is needed for reproducible rendering.
    cm=Path(matplotlib.get_data_path())/'fonts/ttf/cmr10.ttf'
    if not cm.exists():raise FileNotFoundError('Computer Modern cmr10.ttf unavailable')
    font_manager.fontManager.addfont(str(cm))
    family=font_manager.FontProperties(fname=str(cm)).get_name()
    plt.rcParams.update({'font.family':family,'font.size':11,'mathtext.fontset':'cm',
       'axes.formatter.use_mathtext':True,'axes.unicode_minus':False,
       'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
       'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})
    assert Path(font_manager.findfont(font_manager.FontProperties(family=family))).name=='cmr10.ttf'
    output_values={}
    # Figure 1: nine validation-selected observations, not nine test results.
    fig,axes=panel_figure('Three-seed model comparison')
    plotted=[]
    for ax,metric in zip(axes,['IC','RankIC']):
        for x,width in enumerate([64,128,256]):
            records=sorted([r for r in values['validation']['per_seed'] if r['width']==width],key=lambda r:r['seed'])
            assert [r['seed'] for r in records]==[0,1,2]
            y=np.array([r['best_validation'][metric] for r in records])
            assert np.isfinite(y).all()
            ax.scatter(x+np.array([-.075,0,.075]),y,s=32,color=COLORS[width],zorder=4,edgecolors='white',linewidths=.35)
            ax.plot([x-.16,x+.16],[y.mean(),y.mean()],color=INK,lw=1.5,zorder=5)
            plotted.append({'width':width,'metric':metric,'seed_values':y.tolist(),'mean':float(y.mean())})
        style_axes(ax,[0,1,2],['64','128','256'],(.075,.13),[.08,.10,.12])
        ax.set_xlim(-.4,2.4);ax.set_xlabel('Hidden width',labelpad=7)
    handles=[Line2D([],[],linestyle='none',marker='o',color='#777777',markersize=5,label='Individual seed'),
             Line2D([],[],color=INK,lw=1.5,label='Mean of three')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,.85),ncol=2,frameon=False,fontsize=10)
    fig.savefig(HERE/'01_three_seed_validation.png',dpi=400)
    plt.close(fig)
    output_values['validation']=plotted
    # Figure 2: the matched evaluation of one supplied author model and our
    # three threshold-selected width-256 models, sharing one x position.
    fig,axes=plt.subplots(1,2,figsize=(4.6,3.1),sharey=True)
    fig.subplots_adjust(left=.16,right=.97,bottom=.12,top=.76,wspace=.20)
    axes[0].set_ylabel('Correlation',labelpad=6)
    for ax,metric in zip(axes,['IC','RankIC']):
        ax.set_title(metric,fontsize=12,pad=8)
    test=values['test']
    reference=next(r for r in test['models'] if r['kind']=='upstream')
    ours=sorted([r for r in test['models'] if r['kind']=='local_threshold'],key=lambda r:r['seed'])
    assert [r['seed'] for r in ours]==[0,1,2]
    plotted=[]
    for ax,metric in zip(axes,['IC','RankIC']):
        measured=reference['metrics'][metric]
        y=np.array([r['metrics'][metric] for r in ours])
        ax.plot([-.23,.23],[measured,measured],color='#0072B2',lw=1.6,ls='--',zorder=3)
        ax.scatter(np.array([-.10,0,.10]),y,s=29,color='#D55E00',edgecolors='white',linewidths=.4,zorder=5)
        ax.plot([-.23,.23],[y.mean(),y.mean()],color='#D55E00',lw=1.5,zorder=4)
        style_axes(ax,[],[],(0,.1),[0,.025,.05,.075,.1])
        ax.set_xlim(-.4,.4)
        plotted.append({'metric':metric,'supplied_checkpoint':measured,
                        'ours_seed_values':y.tolist(),'ours_mean':float(y.mean())})
    handles=[Line2D([],[],color='#0072B2',lw=1.6,ls='--',label="Authors' checkpoint"),
             Line2D([],[],linestyle='none',marker='o',color='#D55E00',markersize=5,label='Our seeds'),
             Line2D([],[],color='#D55E00',lw=1.5,label='Our mean')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.55,.98),ncol=3,frameon=False,fontsize=12,
               columnspacing=.6,handlelength=1.0)
    fig.savefig(HERE/'02_matched_test_reproduction.png',dpi=400)
    plt.close(fig)
    output_values['test']=plotted
    manifest={'font':family,'font_file':cm.name,'dpi':400,'border_width_points':.75,
        'seed_point_order':'left to right = seeds 0, 1, 2; horizontal line = arithmetic mean',
        'source_files':{k:{'path':str(p.relative_to(WORKSPACE)),
                           'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in paths.items()},
        'plotted_values':output_values,
        'limits':'Validation-selected first figure; threshold-selected held-out test second figure. No test scores for widths64/128 are implied.'}
    (HERE/'figure_sources.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Created exactly two images in',HERE)


if __name__=='__main__':main()
