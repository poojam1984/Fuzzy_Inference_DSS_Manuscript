import sys; sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__file__), '..'))
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from src.membership_functions import INPUT_MF, OUTPUT_MF, trimf
from src.fdss import evaluate, load_cases
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11})
C=['#1D4851','#1F8090','#A84B2F']
# Figure 1
panels=[('safety','Safety Performance (rating 1–10)',(1,10),['poor','average','good'],['Poor','Average','Good']),
('ease','Ease of Installation (rating 1–10)',(1,10),['difficult','moderate','easy'],['Difficult','Moderate','Easy']),
('height','Height Capability Ratio ρ = H$_{cap}$/H$_{req}$',(0,3),['insufficient','sufficient','optimal'],['Insufficient','Sufficient','Optimal']),
('reuse','Reusability (cycles)',(0,260),['low','moderate','high'],['Low','Moderate','High']),
('cost','Operational Cost (USD/hr)',(0,130),['low','moderate','high'],['Low','Moderate','High'])]
fig,axs=plt.subplots(3,2,figsize=(11,10.4),dpi=200)
for ax,(v,t,(lo,hi),sets,labs) in zip(axs.flat,panels):
    x=np.linspace(lo,hi,2001)
    for s,l,c in zip(sets,labs,C):
        y=trimf(x,INPUT_MF[v][s]); ax.plot(x,y,color=c,lw=2.4,label=l); ax.fill_between(x,y,color=c,alpha=0.15)
    ax.set_title(t,fontweight='bold'); ax.set_ylabel('μ',fontweight='bold'); ax.set_ylim(0,1.1); ax.set_xlim(lo-(hi-lo)*0.03,hi+(hi-lo)*0.03)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.legend(loc='upper center',bbox_to_anchor=(0.5,-0.13),ncol=3,frameon=False,fontsize=10)
axs.flat[-1].axis('off')
fig.suptitle('Triangular Membership Functions for Five Linguistic Input Variables',fontweight='bold',fontsize=14)
fig.tight_layout(rect=(0,0,1,0.97)); fig.savefig('fig1_input_mfs.png'); plt.close()
# Figure 5 surface
S=np.linspace(1,10,46); H=np.linspace(0,3,46)
Zs=np.array([[evaluate({'safety':s,'ease':5.5,'height':h,'reuse':100,'cost':60}) for s in S] for h in H])
print('surface nan:',np.isnan(Zs).sum(),'min',np.nanmin(Zs),'max',np.nanmax(Zs))

SS,HH=np.meshgrid(S,H)
fig=plt.figure(figsize=(10.83,9.0),dpi=200); ax=fig.add_subplot(111,projection='3d')
sf=ax.plot_surface(SS,HH,Zs,cmap='viridis',edgecolor='white',linewidth=0.2,alpha=0.95,vmin=0,vmax=100)
ax.set_xlabel('Safety (1–10)',fontweight='bold',labelpad=12); ax.set_ylabel('Height ratio ρ',fontweight='bold',labelpad=12)
ax.set_zlabel('Suitability Score',fontweight='bold',labelpad=10); ax.set_zlim(0,100); ax.view_init(elev=24,azim=-58)
cb=fig.colorbar(sf,shrink=0.55,pad=0.1); cb.set_label('Suitability score',fontweight='bold')
ax.set_title('Defuzzified Suitability Surface: Safety × Height Ratio',fontweight='bold',fontsize=15)
fig.tight_layout(); fig.savefig('fig5_surface.png'); plt.close()
# Figure 6 bars
rows=load_cases(); sysorder=['Timber','Steel','Aluminium','Plastic','Fabric','Engineered']
lab={'Timber':'Timber/\nPlywood','Engineered':'Engineered\nSystems','Aluminium':'Aluminium'}
cases=[('A_10_storey','10-storey case','#FFC555'),('B_45_storey','45-storey case','#1F8090'),('C_90_storey','90-storey case','#1D4851')]
sc={(r['case'],r['system']):evaluate(r) for r in rows}
fig,ax=plt.subplots(figsize=(10.2,6.245),dpi=200); x=np.arange(6); w=0.27
for i,(c,l,col) in enumerate(cases):
    v=[sc[(c,s)] for s in sysorder]
    b=ax.bar(x+(i-1)*w,v,w*0.95,color=col,label=l,edgecolor='white')
    for xi,vi in zip(x+(i-1)*w,v): ax.text(xi,vi+1.2,f'{vi:.1f}',ha='center',fontsize=8)
ax.axhline(50,ls='--',color='#A84B2F',lw=1.4,label='Lower bound of Suitable band (50)')
ax.set_xticks(x); ax.set_xticklabels([lab.get(s,s) for s in sysorder]); ax.set_ylim(0,100)
ax.set_ylabel('Defuzzified Suitability Score (0–100)',fontweight='bold')
ax.set_title('Fuzzy Inference Output Across Three Project Cases (illustrative inputs)',fontweight='bold',fontsize=13)
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.yaxis.grid(True,color='#e5e5e5'); ax.set_axisbelow(True)
ax.legend(loc='upper center',bbox_to_anchor=(0.5,-0.13),ncol=4,frameon=False,fontsize=9.5)
fig.tight_layout(); fig.savefig('fig6_case_scores.png'); plt.close()
for k,v in sc.items(): print(k, round(v,1))
