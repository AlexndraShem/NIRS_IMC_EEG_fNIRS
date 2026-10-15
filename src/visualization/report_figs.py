"""Рисунки отчёта report/figures/: прогресс, модальности, сессии, участники, блоки, цена честности, гистограмма.

    python -m src.visualization.report_figs

Входы: results/metrics/runs (хронологический прогон сдачи), results/metrics_loseso_final/runs,
results/metrics_losubjo_final/runs, results/online_runs_baseline_stage0. Цифры этапов (fig_progress,
fig_honesty_cost) — из metrics.json соответствующих прогонов, см. REPORT.md §6.1, §7.5.
"""
import glob,os,numpy as np,pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R=str(Path(__file__).resolve().parents[2]); F=R+'/report/figures'
C_EEG,C_NIRS,C_HYB,C_GR='#2a78d6','#eb6834','#1baf7a','#898781'
INK,INK2,GRID='#0b0b0b','#52514e','#e1e0d9'
plt.rcParams.update({'font.size':10,'axes.edgecolor':'#c3c2b7','axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,
 'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#fcfcfb','axes.facecolor':'#fcfcfb','savefig.facecolor':'#fcfcfb'})
def met(t,p):
    f=[];r=[]
    for c in (1,2,3):
        tp=np.sum((t==c)&(p==c));fp=np.sum((t!=c)&(p==c));fn=np.sum((t==c)&(p!=c))
        f.append(2*tp/(2*tp+fp+fn) if tp+fp+fn else 0); r.append(tp/(tp+fn) if tp+fn else 0)
    return min(f),np.mean(r)
def load(d):
    out={}
    for s in sorted(glob.glob(d+'/*/predictions.csv')):
        df=pd.read_csv(s); name=os.path.basename(os.path.dirname(s))
        if 'block' not in df:
            tm=pd.read_csv(os.path.join(os.path.dirname(s),'timings.csv'))
            nb=0;m={}
            for ev,es in zip(tm.event,tm.end_sample):
                if ev=='fit_block': nb+=1
                elif ev=='predict': m[int(es)]=nb+1
            df['block']=df.end_sample.map(m)
        out[name]=df
    return out
def sc(df,col):
    if col not in df: return None
    x=df[col]; m=x.notna()
    if m.sum()==0: return None
    f,r=met(df.true_y[m].values.astype(int),x[m].astype(float).astype(int).values); return f,r,60*f+30*r
fin=load(R+'/results/metrics/runs'); base=load(R+'/results/online_runs_baseline_stage0')
# 1 progress
steps=[('Бейзлайн stage0\n(свой, ЭЭГ logvar+NIRS sLDA)',55.49,None,None),
 ('TS-пул ЭЭГ + NIRS-каскад\n+ пулы участника',68.19,72.04,67.91),
 ('ЭЭГ-эксперты L/R\n(nb, sm4, зеркало)',71.19,73.98,70.08),
 ('Стекинг-слияние\n(τ ≤ 4 с)',72.21,75.67,71.79),
 ('+ 4 нейросети\nFBShallow',72.45,76.69,73.05),
 ('+ охрана артефактов\nФЭС/ЭМГ',72.06,76.07,72.63),
 ('+ асинхронный режим,\nτ ≤ 1 с (СДАЧА)',70.81,74.68,70.12)]
fig,ax=plt.subplots(figsize=(12.5,4.8)); x=np.arange(len(steps))
ch=[s[1] for s in steps]; ax.plot(x,ch,'-o',color=C_HYB,lw=2,ms=8,label='хронологический (официальная цифра)',zorder=3)
for i,v in enumerate(ch): ax.annotate(f'{v:.2f}',(i,v),textcoords='offset points',xytext=(0,10),ha='center',color=INK,fontsize=9)
xs=[i for i,s in enumerate(steps) if s[2]]; ax.plot(xs,[steps[i][2] for i in xs],'--s',color=C_EEG,lw=1.5,ms=6,label='LOSessO (прокси закрытого теста)')
ax.plot(xs,[steps[i][3] for i in xs],':^',color=C_NIRS,lw=1.5,ms=7,label='LOSubjO (участник незнаком)')
ax.axhline(55.49,color=C_GR,lw=1,ls='-',alpha=.5)
ax.set_xticks(x); ax.set_xticklabels([s[0] for s in steps],fontsize=8); ax.set_ylabel('итоговый балл (без 20 за отчёт)')
ax.set_ylim(52,79); ax.grid(axis='y',color=GRID,lw=.6); ax.legend(frameon=False,fontsize=9,loc='lower right')
ax.set_title('Путь решения: итог по официальному каркасу, 56 сессий, штрафов 0',color=INK,loc='left')
plt.tight_layout(); plt.savefig(F+'/fig_progress.png',dpi=150); plt.close()
# 2 modalities
prot=[('Бейзлайн stage0',base),('Хронологический',fin),('LOSessO',load(R+'/results/metrics_loseso_final/runs')),('LOSubjO',load(R+'/results/metrics_losubjo_final/runs'))]
res={}
for n,d in prot:
    res[n]={c:np.mean([sc(df,c)[2] for df in d.values() if sc(df,c)]) for c in ('y_eeg','y_nirs','y')}
    res[n]['n_nirs']=sum(1 for df in d.values() if sc(df,'y_nirs'))
print(res)
fig,ax=plt.subplots(figsize=(9,4.2)); w=.26; x=np.arange(len(prot))
for k,(c,lab,col) in enumerate([('y_eeg','только ЭЭГ',C_EEG),('y_nirs','только NIRS',C_NIRS),('y','гибрид ЭЭГ+NIRS',C_HYB)]):
    v=[res[n][c] for n,_ in prot]; b=ax.bar(x+(k-1)*w,v,w-0.03,color=col,label=lab)
    for xi,vi in zip(x+(k-1)*w,v): ax.text(xi,vi+0.6,f'{vi:.1f}',ha='center',fontsize=8,color=INK)
ax.set_xticks(x); ax.set_xticklabels([n for n,_ in prot]); ax.set_ylabel('60·minF1 + 30·recall'); ax.set_ylim(0,72)
ax.grid(axis='y',color=GRID,lw=.6); ax.set_axisbelow(True); ax.legend(frameon=False,ncol=3,loc='upper left',fontsize=9)
ax.set_title('Выигрыш объединения модальностей (среднее по сессиям; NIRS — 54 сессии)',color=INK,loc='left')
plt.tight_layout(); plt.savefig(F+'/fig_modalities.png',dpi=150); plt.close()
# 3 per-session scatter
rows=[]
for s,df in fin.items():
    e=sc(df,'y_eeg'); h=sc(df,'y'); n=sc(df,'y_nirs')
    rows.append(dict(session=s,subj=s.split('_')[-1],eeg=e[2],hyb=h[2],nirs=n[2] if n else np.nan,h_f=h[0],h_r=h[1],e_f=e[0],n_f=n[0] if n else np.nan))
T=pd.DataFrame(rows)
fig,ax=plt.subplots(figsize=(5.4,5.2))
m=T.nirs.notna(); ax.scatter(T.eeg[m],T.hyb[m],s=30,color=C_HYB,edgecolor='#fcfcfb',lw=1,label='сессии с NIRS (54)',zorder=3)
ax.scatter(T.eeg[~m],T.hyb[~m],s=40,color=C_GR,marker='s',label='без NIRS (2): ЭЭГ-only',zorder=3)
lo,hi=25,95; ax.plot([lo,hi],[lo,hi],color=C_GR,lw=1,ls='--'); ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
d=(T.hyb-T.eeg)[m]; ax.text(28,90,f'гибрид лучше ЭЭГ в {int((d>0).sum())} из {m.sum()} сессий\nмедиана Δ = {d.median():+.2f}, среднее Δ = {d.mean():+.2f}',fontsize=9,color=INK,va='top')
ax.set_xlabel('только ЭЭГ: 60F+30P'); ax.set_ylabel('гибрид: 60F+30P'); ax.grid(color=GRID,lw=.6); ax.legend(frameon=False,fontsize=8,loc='lower right')
ax.set_title('По сессиям, хронологический протокол',color=INK,loc='left',fontsize=10)
plt.tight_layout(); plt.savefig(F+'/fig_sessions_eeg_vs_hybrid.png',dpi=150); plt.close()
print('hyb>eeg',int((d>0).sum()),m.sum(),d.median(),d.mean(), 'min',d.min(),'max',d.max())
# 4 per-subject: все модальности — на одних и тех же сессиях (с NIRS), иначе у N02/N03 NIRS усреднён по 3 сессиям, а гибрид по 4
TN=T[T.nirs.notna()]
S=TN.groupby('subj').agg(n=('session','size'),h_f=('h_f','mean'),h_r=('h_r','mean'),e_f=('e_f','mean'),n_f=('n_f','mean')).sort_values('h_f')
print(S.round(3))
fig,ax=plt.subplots(figsize=(9,4.2)); y=np.arange(len(S))
ax.hlines(y,S[['e_f','n_f','h_f']].min(1),S[['e_f','n_f','h_f']].max(1),color=GRID,lw=2)
ax.scatter(S.n_f,y,color=C_NIRS,s=36,label='только NIRS',zorder=3)
ax.scatter(S.e_f,y,facecolor='none',edgecolor=C_EEG,lw=1.8,marker='s',s=70,label='только ЭЭГ',zorder=4)
ax.scatter(S.h_f,y,color=C_HYB,marker='D',s=34,label='гибрид',zorder=5)
ax.set_yticks(y); ax.set_yticklabels([f'{i} ({int(r.n)})' for i,r in S.iterrows()]); ax.set_xlabel('min-F1 (среднее по сессиям участника с NIRS)')
ax.grid(axis='x',color=GRID,lw=.6); ax.legend(frameon=False,fontsize=9,loc='lower right')
ax.set_title('Участники: min-F1 по модальностям, хронологический протокол, 54 сессии с NIRS (в скобках — число сессий)',color=INK,loc='left',fontsize=10)
plt.tight_layout(); plt.savefig(F+'/fig_subjects.png',dpi=150); plt.close()
# 5 per block recall
def blk(d,col):
    out=[]
    for b in range(2,8):
        v=[]
        for df in d.values():
            q=df[df.block==b]
            if col not in q: continue
            x=q[col]; mm=x.notna()
            if mm.sum()==0: continue
            v.append(met(q.true_y[mm].values.astype(int),x[mm].astype(float).astype(int).values)[1])
        out.append(np.mean(v))
    return out
fig,ax=plt.subplots(figsize=(7,4)); bb=range(2,8)
for c,lab,col,ls,d in [('y','гибрид (сдача)',C_HYB,'-',fin),('y_eeg','только ЭЭГ (сдача)',C_EEG,'-',fin),('y_nirs','только NIRS (сдача)',C_NIRS,'-',fin),('y','гибрид бейзлайна stage0',C_GR,'--',base)]:
    v=blk(d,c); print(lab,np.round(v,3)); ax.plot(bb,v,ls,marker='o',color=col,lw=2,label=lab)
ax.axhline(1/3,color=C_GR,lw=.8,ls=':'); ax.text(1.85,0.342,'случайный уровень',ha='left',fontsize=8,color=INK2)
ax.set_xlabel('номер оцениваемого блока b (модели дообучены на блоках 1…b−1)'); ax.set_ylabel('macro-recall'); ax.set_ylim(0.3,0.85); ax.set_xlim(1.8,7.2)
ax.grid(color=GRID,lw=.6); ax.legend(frameon=False,fontsize=8,loc='lower right',bbox_to_anchor=(1,0.08))
ax.set_title('Дообучение по блокам, хронологический протокол',color=INK,loc='left',fontsize=10)
plt.tight_layout(); plt.savefig(F+'/fig_blocks.png',dpi=150); plt.close()
# 6 honesty cost
labs=['синхронно,\nбез охраны','+ охрана\nартефактов','+ асинхронно,\nτ ≤ 1 с (сдача)']
vals={'хронологический':[72.45,72.06,70.81],'LOSessO':[76.69,76.07,74.68],'LOSubjO':[73.05,72.63,70.12]}
fig,ax=plt.subplots(figsize=(7.5,4)); w=.26; x=np.arange(3)
for k,(n,col) in enumerate(zip(vals,[C_HYB,C_EEG,C_NIRS])):
    ax.bar(x+(k-1)*w,vals[n],w-0.03,color=col,label=n)
    for xi,vi in zip(x+(k-1)*w,vals[n]): ax.text(xi,vi+0.15,f'{vi:.2f}',ha='center',fontsize=8,color=INK)
ax.set_ylim(66,78); ax.set_xticks(x); ax.set_xticklabels(labs); ax.set_ylabel('итоговый балл'); ax.grid(axis='y',color=GRID,lw=.6); ax.set_axisbelow(True)
ax.legend(frameon=False,fontsize=9,ncol=3,loc='upper right'); ax.set_title('Цена «честности»: сколько баллов стоили решения о соответствии правилам',color=INK,loc='left',fontsize=10)
plt.tight_layout(); plt.savefig(F+'/fig_honesty_cost.png',dpi=150); plt.close()
# 7 hist per-session hybrid minF1
fig,ax=plt.subplots(figsize=(6.5,3.6))
bins=np.linspace(0.1,1,19)
bf=[sc(df,'y')[0] for df in base.values()]
ax.hist(bf,bins=bins,color=C_GR,alpha=.55,label=f'бейзлайн stage0 (медиана {np.median(bf):.2f})',edgecolor='#fcfcfb')
ax.hist(T.h_f,bins=bins,color=C_HYB,alpha=.8,label=f'сдача (медиана {T.h_f.median():.2f})',edgecolor='#fcfcfb')
ax.set_xlabel('min-F1 гибрида в сессии'); ax.set_ylabel('число сессий'); ax.legend(frameon=False,fontsize=8); ax.grid(axis='y',color=GRID,lw=.6); ax.set_axisbelow(True)
ax.set_title('Распределение по 56 сессиям, хронологический протокол',color=INK,loc='left',fontsize=10)
plt.tight_layout(); plt.savefig(F+'/fig_session_hist.png',dpi=150); plt.close()
print('baseline minF1 median',np.median(bf),'final median',T.h_f.median(),'q10/q90',T.h_f.quantile([.1,.9]).values)
# class F1 pooled? per-class mean F1
for n,d in [('final',fin),('base',base)]:
    fs=[]
    for df in d.values():
        t=df.true_y.values; p=df.y.values
        fs.append([2*np.sum((t==c)&(p==c))/(np.sum(t==c)+np.sum(p==c)) for c in (1,2,3)])
    print(n,'mean per-class F1',np.round(np.mean(fs,0),3))
