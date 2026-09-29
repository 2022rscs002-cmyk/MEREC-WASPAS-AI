from common import *
import numpy as np, pandas as pd, json, squarify
from scipy.stats import kruskal, spearmanr
import seaborn as sns
df=pd.read_csv('../data/china_smart_infrastructure_corpus.csv'); meta=json.load(open('meta.json')); crit=meta['crit']; short=meta['short']
df['season']=df.quarter.map({1:'Winter',2:'Spring',3:'Summer',4:'Autumn'})
_w=pd.read_csv('../data/weights_full_corpus.csv').set_index('criterion'); W=_w.loc[crit,'composite_weight'].values
cities=['Shanghai','Beijing','Shenzhen','Guangzhou','Chengdu','Wuhan','Hangzhou','Tianjin','Nanjing',"Xi'an"]
cls=['Inefficient','Moderate','Optimized']; cc=['#C0392B','#E6A117','#2E8B57']
stats={}
# ---------- Figure 4
fig=plt.figure(figsize=(7.2,8.4)); gs=fig.add_gridspec(3,3,hspace=0.62,wspace=0.55)
ax=fig.add_subplot(gs[0,0]); ax.hist(df.WASPAS_score,bins=40,color='#2F6DB5',edgecolor='white',lw=0.3)
ax.axvline(df.WASPAS_score.mean(),color='#C0392B',ls='--',lw=1,label=f'Mean = {df.WASPAS_score.mean():.3f}')
ax.set_title('WASPAS score distribution'); ax.set_xlabel('WASPAS score'); ax.set_ylabel('Count'); ax.legend(frameon=False); letter(ax,'(a)')
ax=fig.add_subplot(gs[0,1]); cnt=df.optimization_class.value_counts().sort_index()
ax.pie(cnt,labels=cls,colors=cc,autopct='%1.1f%%',startangle=90,wedgeprops=dict(width=0.42,edgecolor='white'),textprops={'fontsize':7},pctdistance=0.78)
ax.set_title('Optimization class distribution'); letter(ax,'(b)')
ax=fig.add_subplot(gs[0,2]); order=df.groupby('city_id').energy_consumption_kWh.median().sort_values().index
sns.violinplot(data=df,x='city_id',y='energy_consumption_kWh',order=order,ax=ax,palette='Set3',linewidth=0.5,inner='quartile',cut=0)
ax.set_title('Energy consumption by city'); ax.set_xlabel(''); ax.set_ylabel('kWh per day'); ax.tick_params(axis='x',rotation=60,labelsize=6); letter(ax,'(c)')
H,p=kruskal(*[g.energy_consumption_kWh.dropna() for _,g in df.groupby('city_id')]); stats['energy_KW']=(H,p,H/(len(df)-1))
ax=fig.add_subplot(gs[1,0]); v=['energy_consumption_kWh','air_quality_index','iot_sensor_density','renewable_energy_pct','digital_twin_score','cyber_resilience','structural_health','carbon_emission_tons','WASPAS_score']
lab=['Energy','AQI','IoT','Renew','DigiTwin','CyberRes','StructH','Carbon','WASPAS']
C=df[v].corr(method='spearman',min_periods=100).values; mask=np.triu(np.ones_like(C,bool),0)
sns.heatmap(pd.DataFrame(C,index=lab,columns=lab).iloc[1:,:-1],mask=mask[1:,:-1],cmap='RdBu_r',vmin=-1,vmax=1,annot=True,fmt='.2f',annot_kws={'size':4.4},ax=ax,cbar_kws={'shrink':0.8},linewidths=0.3)
ax.set_title('Spearman correlation matrix'); ax.tick_params(labelsize=5.5); letter(ax,'(d)')
cbar=ax.collections[0].colorbar; cbar.ax.tick_params(labelsize=6)
for a,b in [('iot_sensor_density','digital_twin_score'),('iot_sensor_density','cyber_resilience'),('carbon_emission_tons','renewable_energy_pct')]: _d=df[[a,b]].dropna(); stats[a+'|'+b]=spearmanr(_d[a],_d[b])[0]
ax=fig.add_subplot(gs[1,1]); r=df.groupby(['year','city_id']).renewable_energy_pct.mean().unstack()[cities]
pal=sns.color_palette('tab10',10)
for i,c in enumerate(cities): ax.plot(r.index,r[c],marker='o',ms=2.5,lw=0.9,color=pal[i],label=c)
ax.set_xticks(r.index); ax.set_ylabel('Renewable share (%)'); ax.set_title('Renewable share by year'); letter(ax,'(e)')
ax.legend(ncol=2,frameon=False,fontsize=5,loc='upper left',handlelength=1.2,columnspacing=0.6)
stats['renew']=r.round(1).to_dict()
ax=fig.add_subplot(gs[1,2]); _h=df[['iot_sensor_density','digital_twin_score']].dropna(); hb=ax.hexbin(_h.iot_sensor_density,_h.digital_twin_score,gridsize=22,cmap='YlOrRd',mincnt=1,linewidths=0.2)
ax.set_xlabel('IoT sensor density (sensors/km²)'); ax.set_ylabel('Digital twin score'); ax.set_title('IoT density vs digital twin'); cb=fig.colorbar(hb,ax=ax,shrink=0.8); cb.set_label('Count',fontsize=6); cb.ax.tick_params(labelsize=6); letter(ax,'(f)')
ax=fig.add_subplot(gs[2,0])
for k in range(3): s=df[df.optimization_class==k]; ax.scatter(s.carbon_emission_tons,s.air_quality_index,s=3,alpha=0.45,color=cc[k],label=cls[k],lw=0)
ax.set_xlabel('Carbon emission (t/day)'); ax.set_ylabel('Air quality index'); ax.set_title('Carbon emission vs AQI by class'); ax.legend(frameon=False,markerscale=3); letter(ax,'(g)')
ax=fig.add_subplot(gs[2,1]); o=np.argsort(W); ax.barh(np.array(short)[o],W[o],color=plt.cm.viridis(np.linspace(0.15,0.85,20)))
ax.set_xlabel('Composite weight $\\bar{w}_j$'); ax.set_title('MEREC criterion weights'); ax.tick_params(axis='y',labelsize=5.2); letter(ax,'(h)')
ax=fig.add_subplot(gs[2,2])
for k in range(3): ax.hist(df[df.optimization_class==k].structural_health.dropna(),bins=25,alpha=0.55,color=cc[k],label=cls[k],density=True)
ax.set_xlabel('Structural health score'); ax.set_ylabel('Density'); ax.set_title('Structural health by class'); ax.legend(frameon=False); letter(ax,'(i)')
save(fig,'Figure4')
# structural health by class - put as separate inset? we need 9 panels: make panel (i) by replacing layout
# ---------- Figure 5
fig=plt.figure(figsize=(7.2,5.6)); gs=fig.add_gridspec(2,3,hspace=0.55,wspace=0.5)
dims=['iot_sensor_density','digital_twin_score','cyber_resilience','coverage_5G','green_space_ratio','renewable_energy_pct']; dl=['IoT density','Digital twin','Cyber res.','5G coverage','Green space','Renewable']
cm=df.groupby('city_id')[dims].mean(); cmn=(cm-cm.min())/(cm.max()-cm.min())*0.8+0.2
ax=fig.add_subplot(gs[0,0],polar=True); ang=np.linspace(0,2*np.pi,6,endpoint=False).tolist(); ang+=ang[:1]
for c,col in zip(['Shanghai','Shenzhen','Beijing','Tianjin',"Xi'an"],['#1f77b4','#2ca02c','#ff7f0e','#7f7f7f','#17becf']):
    v=cmn.loc[c].tolist(); v+=v[:1]; ax.plot(ang,v,color=col,lw=1,label=c); ax.fill(ang,v,color=col,alpha=0.08)
ax.set_xticks(ang[:-1]); ax.set_xticklabels(dl,fontsize=5.4); ax.tick_params(axis='x',pad=1); ax.set_yticklabels([]); ax.set_ylim(0,1.05)
ax.set_title('(a) City performance profiles',pad=14); ax.legend(fontsize=5.2,frameon=False,loc='upper center',bbox_to_anchor=(0.5,-0.12),ncol=3)
ax=fig.add_subplot(gs[0,1]); cy=df.groupby(['city_id','year']).WASPAS_score.mean().unstack()
sns.heatmap(cy,annot=True,fmt='.3f',cmap='YlGnBu',annot_kws={'size':4.8},ax=ax,cbar_kws={'shrink':0.8},linewidths=0.3)
ax.set_title('(b) Mean WASPAS score: city × year'); ax.set_xlabel('Year'); ax.set_ylabel(''); ax.tick_params(labelsize=5.8); ax.collections[0].colorbar.ax.tick_params(labelsize=5.5)
stats['cy_change']=(cy[2025]-cy[2022]).round(3).to_dict()
ax=fig.add_subplot(gs[0,2]); so=['Spring','Summer','Autumn','Winter']
bp=ax.boxplot([df[df.season==s].WASPAS_score for s in so],labels=so,notch=True,patch_artist=True,widths=0.55,flierprops=dict(markersize=1.5))
for pch,col in zip(bp['boxes'],['#8FD18F','#F28B82','#FBC06A','#8AB6F9']): pch.set_facecolor(col); pch.set_linewidth(0.6)
ax.set_ylabel('WASPAS score'); ax.set_title('(c) WASPAS score by season')
H,p=kruskal(*[g.WASPAS_score for _,g in df.groupby('season')]); stats['season_KW']=(H,p,H/(len(df)-1)); stats['season_med']=df.groupby('season').WASPAS_score.median().round(4).to_dict()
ax=fig.add_subplot(gs[1,0]); tv=['iot_sensor_density','coverage_5G','digital_twin_score','renewable_energy_pct']; tl=['IoT density','5G coverage','Digital twin','Renewable share']
ym=df.groupby('year')[tv].mean(); idx=ym/ym.iloc[0]*100
for v_,l_,col in zip(tv,tl,['#1f77b4','#9467bd','#d62728','#2ca02c']): ax.plot(idx.index,idx[v_],marker='o',ms=3,lw=1.1,label=l_,color=col)
ax.set_xticks(idx.index); ax.set_ylabel('Index (2022 = 100)'); ax.set_title('(d) Technology adoption trends'); ax.legend(frameon=False)
stats['tech_idx']=idx.round(1).to_dict()
ax=fig.add_subplot(gs[1,1]); sh=pd.crosstab(df.year,df.optimization_class,normalize='index')*100
ax.stackplot(sh.index,sh[0],sh[1],sh[2],colors=['#C0392B','#E6A117','#2E8B57'],alpha=0.8,labels=cls); ax.set_xticks(sh.index); ax.set_ylim(0,100)
ax.set_ylabel('Share of records (%)'); ax.set_title('(e) Class composition by year'); ax.legend(frameon=False,loc='lower left',fontsize=5.5)
stats['shares']=sh.round(1).to_dict()
ax=fig.add_subplot(gs[1,2]); smp=df.dropna(subset=['coverage_5G','cyber_resilience','iot_sensor_density']).sample(600,random_state=1)
sc=ax.scatter(smp.coverage_5G,smp.cyber_resilience,s=smp.iot_sensor_density/3,c=smp.WASPAS_score,cmap='RdYlGn',alpha=0.7,lw=0.2,edgecolor='k')
cb=fig.colorbar(sc,ax=ax,shrink=0.85); cb.set_label('WASPAS score',fontsize=6); cb.ax.tick_params(labelsize=5.5)
ax.set_xlabel('5G coverage'); ax.set_ylabel('Cyber resilience'); ax.set_title('(f) 5G × cyber resilience\n(bubble area = IoT density)')
save(fig,'Figure5')
# ---------- Figure 6
fig=plt.figure(figsize=(7.2,2.8)); gs=fig.add_gridspec(1,3,wspace=0.45,width_ratios=[1,1.1,1.1])
ax=fig.add_subplot(gs[0]); o=np.argsort(W); y=np.arange(20)
ax.hlines(y,0,W[o],color='#7a9cc6',lw=1); ax.scatter(W[o],y,color=plt.cm.RdYlGn(np.linspace(0.1,0.9,20)),s=14,zorder=3,edgecolor='k',lw=0.3)
ax.axvline(0.05,color='#C0392B',ls='--',lw=0.8,label='Uniform = 0.050'); ax.set_yticks(y); ax.set_yticklabels(np.array(short)[o],fontsize=5.3)
ax.set_xlabel('Composite weight $\\bar{w}_j$'); ax.set_title('(a) Criterion weights (sorted)'); ax.legend(frameon=False,fontsize=5.5,loc='lower right')
X=df[crit].fillna(df[crit].median()).values.astype(float); B=np.array([1 if c in('green_space_ratio','smart_meter_coverage','iot_sensor_density','renewable_energy_pct','digital_twin_score','cyber_resilience','coverage_5G','waste_management_eff','structural_health') else 0 for c in crit])
N=np.where(B==1,X/X.max(0),X.min(0)/X); wsm=(N*W).sum(1); wpm=np.prod(N**W,1)
lam=np.round(np.arange(0,1.0001,0.05),2); sc_=np.array([l*wsm+(1-l)*wpm for l in lam]); m=sc_.mean(1); sd=sc_.std(1)
ax=fig.add_subplot(gs[1]); ax.plot(lam,m,'-o',ms=2.5,color='#1f5fa8',lw=1.1,label='Mean WASPAS score'); ax.fill_between(lam,m-sd,m+sd,color='#1f5fa8',alpha=0.15,label='± 1 SD')
ax.axvspan(0.20,0.40,color='#999999',alpha=0.15,label='Eq. 18 objective within 1% of minimum'); ax.axvline(0.30,color='#555555',ls=':',lw=1.0,label='λ* = 0.30 (minimum)'); ax.axvline(0.5,color='#C0392B',ls='--',lw=0.9,label='λ = 0.50 (used)')
ax.set_xlabel('λ'); ax.set_ylabel('WASPAS score'); ax.set_title('(b) λ sweep of the WASPAS score'); ax.legend(frameon=False,fontsize=4.8,loc='upper left')
stats['lam']=dict(wpm=float(m[0]),wsm=float(m[-1]),m03=float(m[6]),m07=float(m[14]),m05=float(m[10]))
ax=fig.add_subplot(gs[2]); o2=np.argsort(-W)
squarify.plot(sizes=W[o2],label=[(f'{short[i]}\n{W[i]:.3f}' if W[i]>0.03 else f'{short[i]}\n{W[i]:.3f}') for i in o2],color=plt.cm.tab20(np.arange(20)%20),ax=ax,text_kwargs={'fontsize':4.2},pad=False,ec='white',linewidth=0.6)
ax.axis('off'); ax.set_title('(c) Proportional weight treemap')
save(fig,'Figure6')
json.dump(stats,open('eda_stats.json','w'),indent=1,default=float)
