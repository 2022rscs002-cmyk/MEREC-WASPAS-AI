from common import *
import numpy as np, json
R=json.load(open('results.json')); P1=R['P1']
order=['proposed','B1','B6','B5','B7','B2','B4','B3']
nm={'proposed':'Proposed','B1':'B1 TOPSIS–SVM','B2':'B2 AHP–GradBoost','B3':'B3 VIKOR–RF','B4':'B4 ELECTRE–AdaBoost','B5':'B5 CRITIC–XGBoost','B6':'B6 PROMETHEE–MLP','B7':'B7 COPRAS–LightGBM'}
M=[nm[k] for k in order]
acc=[P1[k]['acc'] for k in order]; prec=[P1[k]['prec'] for k in order]; rec=[P1[k]['rec'] for k in order]; f1=[P1[k]['f1'] for k in order]; auc=[P1[k]['auc'] for k in order]
cols=['#1B7F3B']+['#4C72B0','#55A868','#8172B2','#CCB974','#64B5CD','#DD8452','#C44E52']
# Fig 7 horizontal bars
fig,ax=plt.subplots(figsize=(7.0,3.0)); y=np.arange(8)[::-1]
ax.barh(y,acc,color=cols,height=0.62,edgecolor='k',lw=0.3)
for yi,a in zip(y,acc): ax.text(a+0.003,yi,f'{a:.4f}',va='center',fontsize=6.5)
ax.axvline(acc[0],color='#C0392B',ls='--',lw=0.9); ib=int(np.argmax(acc)); ax.scatter([acc[ib]+0.022],[y[ib]],marker='*',s=90,color='gold',edgecolor='k',lw=0.4,zorder=4)
ax.set_yticks(y); ax.set_yticklabels(M); ax.set_xlim(0.70,0.99); ax.set_xlabel('Accuracy (P1 test partition, leakage-controlled, n = 400)')
save(fig,'Figure7')
# Fig 8 polar
fig=plt.figure(figsize=(4.4,4.4)); ax=fig.add_subplot(111,polar=True); th=np.linspace(0,2*np.pi,8,endpoint=False)
ax.bar(th,np.array(prec)-0.70,width=2*np.pi/8*0.8,bottom=0.70,color=cols,edgecolor='k',lw=0.3,alpha=0.9)
ax.set_ylim(0.60,0.99); ax.set_yticks([0.75,0.85,0.95]); ax.set_yticklabels(['0.75','0.85','0.95'],fontsize=5.5)
ax.set_xticks(th); ax.set_xticklabels([m.replace(' ','\n',1) for m in M],fontsize=5.4); ax.tick_params(axis='x',pad=6)
for t,p in zip(th,prec): ax.text(t,p+0.012,f'{p:.4f}',ha='center',fontsize=5.5)
ax.set_title('Macro-averaged precision',pad=12)
save(fig,'Figure8')
# Fig 9 diverging recall
fig,ax=plt.subplots(figsize=(7.0,3.0)); base=np.mean(rec[1:]); d=np.array(rec)-base; x=np.arange(8)
for xi,di in zip(x,d): ax.plot([xi,xi],[0,di],color='#2E8B57' if di>0 else '#C0392B',lw=2)
ax.scatter(x,d,color=['#2E8B57' if di>0 else '#C0392B' for di in d],s=36,zorder=3,edgecolor='k',lw=0.3)
for xi,di,r in zip(x,d,rec): ax.text(xi,di+(0.006 if di>0 else -0.012),f'{r:.4f}',ha='center',fontsize=6.3)
ax.axhline(0,color='k',lw=0.7); ax.set_xticks(x); ax.set_xticklabels(M,fontsize=6.2,rotation=28,ha='right')
ax.set_ylabel(f'Deviation from comparator mean ({base:.4f})'); ax.set_ylim(-0.10,0.125); ax.set_title('Macro-averaged recall relative to the mean of the seven comparators')
save(fig,'Figure9'); print('recall base',base,d[0])
# Fig 10 F1 gradient bars
fig,ax=plt.subplots(figsize=(7.0,3.2)); cmap=plt.cm.Greens; nrm=plt.Normalize(0.74,0.96)
ax.bar(x,f1,color=cmap(nrm(np.array(f1))),edgecolor='k',lw=0.3,width=0.65)
for xi,f in zip(x,f1): ax.text(xi,f+0.003,f'{f:.4f}',ha='center',fontsize=6.3)
ax.annotate(f'Proposed: {f1[0]:.4f}',xy=(0,f1[0]),xytext=(1.8,0.975),arrowprops=dict(arrowstyle='->',lw=0.8),fontsize=7)
ax.set_ylim(0.72,0.99); ax.set_xticks(x); ax.set_xticklabels(M,fontsize=6.2,rotation=28,ha='right'); ax.set_ylabel('Macro F1 score')
sm=plt.cm.ScalarMappable(norm=nrm,cmap=cmap); cb=fig.colorbar(sm,ax=ax,pad=0.01); cb.set_label('F1',fontsize=7); cb.ax.tick_params(labelsize=6)
save(fig,'Figure10')
# Fig 11 AUC step-fill
fig,ax=plt.subplots(figsize=(7.0,3.0))
ax.step(x,auc,where='mid',color='#1f5fa8',lw=1.2); ax.fill_between(x,0.88,auc,step='mid',color='#1f5fa8',alpha=0.18)
ax.scatter(x,auc,color=cols,s=30,zorder=3,edgecolor='k',lw=0.3)
for xi,a in zip(x,auc): ax.text(xi,a+0.003,f'{a:.4f}',ha='center',fontsize=6.3)
ax.axhline(auc[0],color='#C0392B',ls='--',lw=0.8); ax.set_ylim(0.88,1.0)
ax.set_xticks(x); ax.set_xticklabels(M,fontsize=6.2,rotation=28,ha='right'); ax.set_ylabel('Macro one-vs-rest AUC-ROC')
save(fig,'Figure11')
# Fig 12 ablation
A=[]
for k,lab in [('proposed','Full model'),('wo_xgb','w/o XGBoost'),('wo_lgbm','w/o LightGBM'),('wo_mlp','w/o MLP'),('wo_rf','w/o Random Forest'),('wo_et','w/o Extra Trees'),('softvote','Soft vote\n(no meta-learner)'),('xgb_lgbm','XGB + LightGBM\nonly'),('B5','Single XGBoost'),('proposed','Equal weights\n(labels fixed)'),('proposed','Classical MEREC\n(labels fixed)'),('multiplicative','Multiplicative\nweighting')]:
    A.append((lab,P1[k]['acc'],P1[k]['f1'],P1[k]['auc']))
fig,(ax,ax2)=plt.subplots(2,1,figsize=(7.2,4.6),sharex=True,gridspec_kw={'height_ratios':[1.25,1],'hspace':0.08}); x=np.arange(len(A)); bw=0.36
ax.bar(x-bw/2,[a[1] for a in A],bw,label='Accuracy',color='#4C72B0',edgecolor='k',lw=0.25); ax.bar(x+bw/2,[a[2] for a in A],bw,label='Macro F1',color='#55A868',edgecolor='k',lw=0.25)
for xi,a in zip(x,A): ax.text(xi-bw/2,a[1]+0.002,f'{a[1]:.4f}',ha='center',fontsize=4.8,rotation=90,va='bottom')
ax.axhline(A[0][1],color='#4C72B0',ls=':',lw=0.8); ax.set_ylim(0.80,0.96); ax.set_ylabel('Accuracy / macro F1'); ax.legend(frameon=False,ncol=2,loc='lower left')
ax2.bar(x,[a[3] for a in A],0.5,color='#C44E52',edgecolor='k',lw=0.25,label='Macro AUC'); ax2.set_ylim(0.94,1.0); ax2.set_ylabel('Macro AUC'); ax2.axhline(A[0][3],color='#C44E52',ls=':',lw=0.8)
for a_ in (ax,ax2): a_.axvspan(8.5,11.5,color='#999999',alpha=0.12)
ax2.text(10,0.9965,'Identical record-level predictions',ha='center',fontsize=6)
ax2.set_xticks(x); ax2.set_xticklabels([a[0] for a in A],fontsize=5.6,rotation=35,ha='right')
save(fig,'Figure12')
# Fig 13 CV violins
cv=R['misc']['cv']
V=[('Proposed',cv['proposed']),('B1 TOPSIS–SVM',cv['B1']),('B2 AHP–GradBoost',cv['B2']),('B3 VIKOR–RF',cv['B3'])]
fig,ax=plt.subplots(figsize=(6.4,3.0)); data=[np.array(v) for _,v in V]
parts=ax.violinplot(data,showmeans=False,showextrema=False,widths=0.7)
for pc,c in zip(parts['bodies'],['#1B7F3B','#4C72B0','#64B5CD','#DD8452']): pc.set_facecolor(c); pc.set_alpha(0.45); pc.set_edgecolor('k'); pc.set_linewidth(0.4)
for i,(d,(n_,v)) in enumerate(zip(data,V)):
    m=np.mean(d); s=np.std(d,ddof=1)
    ax.scatter(np.full(5,i+1)+np.linspace(-0.05,0.05,5),d,s=10,color='k',zorder=3)
    ax.text(i+1,max(d)+0.012,f'mean {m:.4f}\nSD {s:.4f}',ha='center',fontsize=6)
ax.set_xticks(range(1,5)); ax.set_xticklabels([v[0] for v in V]); ax.set_ylabel('Macro F1 per fold'); ax.set_ylim(0.68,0.99)
save(fig,'Figure13')

