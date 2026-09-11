import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.linewidth':0.6,'axes.edgecolor':'#555',
 'xtick.color':'#444','ytick.color':'#444','axes.labelcolor':'#222','lines.linewidth':1.6,'legend.frameon':False,
 'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
B,O,Aq,G='#2a78d6','#eb6834','#1baf7a','#8a8a85'
ramp=['#86b6ef','#2a78d6','#104281']
f1=json.load(open('fig1.json')); f2=json.load(open('fig2.json'))
fig,ax=plt.subplots(2,2,figsize=(7.2,5.6))
a=ax[0,0]
a.plot(f1['a_t'],f1['a_bare'],color=G,lw=1.2,label='bare JC (lossless)')
a.plot(*f1['a_lossless'],color=B,label=r'$\sigma_z$ echo, lossless')
a.plot(*f1['a_C1e3'],color=O,ls='--',label=r'$\sigma_z$ echo, $C\simeq10^3$')
a.axvline(5,color='#bbb',lw=0.6,ls=':'); a.text(5.15,-0.93,'kick',fontsize=7,color='#666')
a.set_xlabel(r'$g t$'); a.set_ylabel(r'$\langle\sigma_z(t)\rangle$'); a.set_ylim(-1.05,1.05)
a.legend(fontsize=7,loc='upper left',bbox_to_anchor=(0.18,1.0)); a.set_title('(a) chiral echo, $\\bar n=5$, $gT=5$',fontsize=9,loc='left')
a=ax[0,1]
for c,(lab,tex) in zip(ramp,[('C1e3',r'$10^3$'),('C1e4',r'$10^4$'),('C1e5',r'$10^5$')]):
    d=f1['b_'+lab]
    a.plot(d['T2'],d['F'],color=c,marker='o',ms=3.2,lw=1.3,label=r'$F(2T)$, $C=$'+tex)
    a.plot(d['T2'],d['Fbound'],color=c,ls=':',lw=1.3)
a.set_xlabel(r'echo time $2gT$'); a.set_ylabel('echo fidelity'); a.set_ylim(0,1.02)
a.text(0.97,0.97,'dotted: Theorem 4 bound',transform=a.transAxes,ha='right',va='top',fontsize=7,color='#555')
a.legend(fontsize=7,loc='lower left'); a.set_title('(b) dissipative horizon',fontsize=9,loc='left')
a=ax[1,0]
a.plot(f2['bare_t'],f2['bare'],color=G,lw=1.1,label='bare JC, $C\\simeq10^3$')
for c,T,m in zip([O,Aq,B],[2.0,0.5,0.125],['s','^','o']):
    st,ss=f2['kick_%g'%T]; a.plot(st,ss,color=c,marker=m,ms=3,lw=0.9,ls='-' if T<1 else '--',label=r'kicks, $gT=%g$'%T,markevery=max(1,len(st)//14))
tt=np.linspace(0,40,200); a.plot(tt,2*np.exp(-0.06*tt)-1,color='k',lw=0.9,ls=':',label=r'$2e^{-\gamma t}-1$')
a.set_xlabel(r'$g t$'); a.set_ylabel(r'$\langle\sigma_z\rangle$ (stroboscopic)'); a.set_ylim(-1.05,1.05)
a.legend(fontsize=6.5,loc='lower left',ncol=1); a.set_title('(c) fast kicks decouple, dissipator survives',fontsize=9,loc='left')
a=ax[1,1]
t=f2['frame_t']
a.fill_between(t,-np.array(f2['bloch']),f2['bloch'],color='#cde2fb',lw=0,label=r'$\pm|\mathbf{s}_0(t)|$ (bare Bloch length)')
a.plot(t,f2['sz_bare'],color=G,lw=1.1,label='bare inversion')
a.plot(t,f2['sz_frame'],color=O,lw=1.3,label='classical-frame Darboux')
a.set_xlabel(r'$g t$'); a.set_ylabel(r'$\langle\sigma_z\rangle$'); a.set_ylim(-1.05,1.05)
a.legend(fontsize=6.5,loc='lower right'); a.set_title('(d) Theorem 3 envelope',fontsize=9,loc='left')
fig.tight_layout(); fig.savefig('fig_physical.pdf'); fig.savefig('fig_physical.eps'); fig.savefig('fig_physical.png',dpi=160)
# logical figure
rows=json.load(open('logical.json'))
n=np.array([r['n'] for r in rows])
fig,ax=plt.subplots(1,2,figsize=(7.2,2.9))
a=ax[0]
a.loglog(n,[r['Vr'] for r in rows],color=O,marker='s',ms=3.5,ls='--',label='ripple-carry [CDKM]')
a.loglog(n,[r['Vc'] for r in rows],color=B,marker='o',ms=3.5,label='carry-lookahead [DKRS]')
a.set_xlabel('register size $n$'); a.set_ylabel(r'logical volume $W\times D$')
a.axvline(21,color='#bbb',lw=0.6,ls=':'); a.text(23,3e2,'crossover\n$n\\approx21$',fontsize=7,color='#666')
a.legend(fontsize=7,loc='upper left'); a.set_title('(a) storage-inclusive volume',fontsize=9,loc='left')
a=ax[1]
nn=np.array([8,16,32,64,128,256,512,1024,4096])
a.semilogx(nn,2*nn,color=O,marker='s',ms=3.5,ls='--',label=r'ripple-carry: $R=2n$')
a.semilogx(n,[r['Rc'] for r in rows],color=B,marker='o',ms=3.5,label=r'carry-lookahead: $R\leq D+1$')
a.semilogx(n,np.ceil(np.log2(2*n)),color=Aq,ls=':',marker='^',ms=3,label=r'circuit light cone: $\lceil\log_2 2n\rceil$')
a.set_yscale('log'); a.set_xlabel('register size $n$'); a.set_ylabel('adaptive rounds $R$')
a.legend(fontsize=7,loc='upper left'); a.set_title('(b) measurement depth',fontsize=9,loc='left')
fig.tight_layout(); fig.savefig('fig_logical.pdf'); fig.savefig('fig_logical.eps'); fig.savefig('fig_logical.png',dpi=160)
