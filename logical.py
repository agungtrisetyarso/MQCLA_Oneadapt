import math, random, json
fl=lambda x: math.floor(math.log2(x))
def w(n): return bin(n).count('1')
# KPG monoid: element (g,p) represents map c -> g | (p & c). compose (left applied after right): (g2,p2)o(g1,p1) = (g2 | p2&g1, p2&p1)
def comp(x2,x1): return (x2[0] | (x2[1]&x1[0]), x2[1]&x1[1])
def bk_prefix(xs):
    # Brent-Kung inclusive prefix: out[i] = xs[i] o ... o xs[0]; count ops and depth
    n=len(xs); v=list(xs); dep=[0]*n; ops=0; maxd=0
    d=1
    while d<n:
        for i in range(2*d-1,n,2*d):
            v[i]=comp(v[i],v[i-d]); dep[i]=max(dep[i],dep[i-d])+1; ops+=1
        d*=2
    d//=2
    while d>=1:
        for i in range(3*d-1,n,2*d):
            v[i]=comp(v[i],v[i-d]); dep[i]=max(dep[i],dep[i-d])+1; ops+=1
        d//=2
    return v,ops,max(dep)
ok=True
for k in range(1,11):
    n=2**k
    for trial in range(30):
        A=random.getrandbits(n); B=random.getrandbits(n)
        xs=[(((A>>i)&1)&((B>>i)&1), ((A>>i)&1)^((B>>i)&1)) for i in range(n)]
        v,ops,dep=bk_prefix(xs)
        S=A+B; carries=[(S^A^B)>>(i+1)&1 for i in range(n)]
        ok&= all(v[i][0]==carries[i] for i in range(n))
    print(n,'ops',ops,'2n-2-log n',2*n-2-k,'depth',dep,'2log n-1',2*k-1)
print('prefix carries correct',ok)
# associativity & noncommutativity of KPG
Kp,Pp,Gp=(0,0),(0,1),(1,0)
els=[Kp,Pp,Gp]
print('assoc',all(comp(comp(x,y),z)==comp(x,comp(y,z)) for x in els for y in els for z in els),
      'G o K',comp(Gp,Kp),'K o G',comp(Kp,Gp))
# resources
def cla(n):
    W=2*n+1+(2*n-w(n)-fl(n)-1)
    D=fl(n)+fl(n-1)+fl(n/3)+fl((n-1)/3)+14
    tof=10*n-3*w(n)-3*w(n-1)-3*fl(n)-3*fl(n-1)-7
    return W,D,tof
def rca(n): return 2*n+2,2*n+4,2*n-1
def S_opt(n): return 542*n+162*(w(n)+fl(n-1)+fl(n)-w(n-1))-395
def dreq(V,eps=1e-3,p=1e-3,pth=7.5e-3,Acoef=0.1):
    d=3
    while V*Acoef*(p/pth)**((d+1)/2)>eps: d+=2
    return d
rows=[]
for n in [8,16,32,64,128,256,512,1024,4096]:
    Wc,Dc,Tc=cla(n); Wr,Dr,Tr=rca(n)
    Vc=Wc*Dc; Vr=Wr*Dr
    dc=dreq(Vc); dr=dreq(Vr)
    rows.append(dict(n=n,Wc=Wc,Dc=Dc,Tc=Tc,Wr=Wr,Dr=Dr,Tr=Tr,Vc=Vc,Vr=Vr,ratio=Vr/Vc,dc=dc,dr=dr,VFTc=Vc*dc**3,VFTr=Vr*dr**3,ratioFT=Vr*dr**3/(Vc*dc**3),S=S_opt(n),Rc=Dc+1,Rr=Dr+1))
for r in rows: print({k:(round(v,2) if isinstance(v,float) else v) for k,v in r.items()})
# crossover
for n in range(4,200):
    Wc,Dc,_=cla(n);Wr,Dr,_=rca(n)
    if Wr*Dr>Wc*Dc: print('crossover n=',n);break
json.dump(rows,open('logical.json','w'))
