"""Render predictions alone, without video, tracking, or ground truth."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

parser = argparse.ArgumentParser()
parser.add_argument('--csv', type=Path, default=Path('submission.csv'))
parser.add_argument('--output', type=Path, default=Path('audit/current/submission_overview.png'))
args = parser.parse_args()
labels = pd.read_csv(args.csv)['label'].to_numpy()
colors = {'H':'#39b8ff','A':'#ffb453','D':'#566679','X':'#28313d'}
names = {'H':'Barcelona (H)','A':'Athletic Club (A)','D':'Dead ball (D)'}
plt.rcParams.update({'font.family':'DejaVu Sans','text.color':'#eef3fa','axes.labelcolor':'#cbd7e5','xtick.color':'#aebed0','ytick.color':'#aebed0','axes.edgecolor':'#354253','axes.facecolor':'#14202e','figure.facecolor':'#0b1420','font.size':11})
fig = plt.figure(figsize=(16,10))
grid = fig.add_gridspec(4,2,height_ratios=[0.12,0.6,0.6,1.5],hspace=.65,wspace=.3)
fig.text(.07,.94,'PREDICTED POSSESSION',fontsize=25,weight='bold')
fig.text(.07,.905,'CSV-only view • no video or ground-truth comparison • 25 frames per second',fontsize=12,color='#aebed0')
durations=[]
graded=[]
for row,(half,start,stop) in enumerate([(1,11185,90652),(2,110808,190198)],1):
    ax=fig.add_subplot(grid[row,:]); w=labels[start:stop]; graded.extend(w)
    edges=np.r_[0,np.flatnonzero(w[1:]!=w[:-1])+1,len(w)]
    for a,b in zip(edges[:-1],edges[1:]):
        label=str(w[a]); ax.broken_barh([((a/1500),(b-a)/1500)],(0,1),facecolors=colors[label],edgecolors='none')
        if label in ('H','A'): durations.append((b-a)/25)
    ax.set_xlim(0,len(w)/1500);ax.set_ylim(0,1);ax.set_yticks([])
    ax.set_title(f'Half {half} · elapsed minutes within published graded block',loc='left',fontsize=12,color='#eef3fa',pad=9)
    ax.set_xlabel('Minutes');ax.spines[['top','right','left']].set_visible(False)
graded=np.array(graded)
ax=fig.add_subplot(grid[3,0]); counts=[int((graded==x).sum()) for x in ('H','A','D')]
ax.pie(counts,colors=[colors[x] for x in ('H','A','D')],startangle=90,wedgeprops={'width':.32,'edgecolor':'#0b1420'},autopct='%1.1f%%',pctdistance=.82,textprops={'color':'white','fontsize':11})
ax.text(0,0,'GRADED\nFRAMES',ha='center',va='center',fontsize=12,color='#cbd7e5')
ax.set_title('Predicted share of graded time',loc='left',color='#eef3fa',pad=12)
ax=fig.add_subplot(grid[3,1]);ax.hist(durations,bins=[0,2,5,10,15,20,30,45,60,90,120,180],color='#39b8ff',edgecolor='#14202e')
ax.set_xlabel('Possession duration (seconds)');ax.set_ylabel('Predicted spells');ax.set_title('Possession duration distribution',loc='left',color='#eef3fa',pad=12)
ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.12)
fig.legend(handles=[Patch(color=colors[x],label=names[x]) for x in ('H','A','D')],loc='upper right',bbox_to_anchor=(.94,.95),frameon=False)
fig.text(.07,.055,f'{len(labels):,} total rows  |  {len(graded):,} graded frames  |  {len(durations):,} predicted possessions  |  {np.mean(durations):.1f}s mean duration',fontsize=12,color='#dbe6f3')
fig.text(.07,.025,'Descriptive output, not accuracy. X marks the 40,462 ungraded frames and is omitted from the shares above.',fontsize=10,color='#91a4bb')
args.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(args.output,dpi=150,bbox_inches='tight');plt.close(fig)
print(args.output.resolve());print('matplotlib',matplotlib.__version__)
