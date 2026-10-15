import numpy as np
exec(open('src/motiv_base.py').read().split("def tri_z")[0])
def bt(C,it=400):
    T=C+C.T; w=np.ones(400); W=C.sum(0)
    for _ in range(it):
        den=(T/(w[:,None]+w[None,:])).sum(1); w=np.where(den>0,(W+1e-9)/den,w); w/=np.exp(np.mean(np.log(w)))
    return w
def triangles(T,thr):
    A=T>=thr; out=[]
    for a in range(400):
        nb=np.where(A[a])[0]; nb=nb[nb>a]
        for i,b in enumerate(nb):
            cs=nb[i+1:]; cs=cs[A[b,cs]]
            out+= [(a,b,c) for c in cs]
    return np.array(out).reshape(-1,3)
def z(C,tr):
    a,b,c=tr.T; C=C+0.5
    f=np.log(C[a,b])+np.log(C[b,c])+np.log(C[c,a])-np.log(C[b,a])-np.log(C[c,b])-np.log(C[a,c])
    v=1/C[a,b]+1/C[b,c]+1/C[c,a]+1/C[b,a]+1/C[c,b]+1/C[a,c]
    return f/np.sqrt(v)
rng=np.random.default_rng(0); Zr=[];Zn=[]
for h in range(7,21):
    kk=hh==h; C=np.zeros((400,400)); np.add.at(C,(oo[kk],ee[kk]),1); np.fill_diagonal(C,0)
    T=C+C.T; tr=triangles(T,8)
    if not len(tr): continue
    w=bt(C); P=w[None,:]/(w[:,None]+w[None,:]); Zr.append(z(C,tr))
    for r in range(5):
        iu=np.triu_indices(400,1); X=rng.binomial(T[iu].astype(int),P[iu]); S=np.zeros_like(C); S[iu]=X; S[(iu[1],iu[0])]=T[iu]-X
        Zn.append(z(S,tr))
Zr=np.concatenate(Zr);Zn=np.concatenate(Zn)
print('triangles',len(Zr),'| real var',Zr.var().round(2),'|z|>3',np.mean(np.abs(Zr)>3).round(4),'| tide-only null var',Zn.var().round(2),'|z|>3',np.mean(np.abs(Zn)>3).round(4))
np.savez('results/figdata_motiv_triangles.npz',zr=Zr,zn=Zn,am=am,pm=pm,net=net,top=top)
