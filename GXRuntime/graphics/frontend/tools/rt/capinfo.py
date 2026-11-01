#!/usr/bin/env python3
# capinfo.py CAPTURE: the additive and reflective draws of a capture: where, how big, what colour
import struct,sys,math
b=open(sys.argv[1],'rb').read()
mag,nd,nv,nt=struct.unpack_from('4I',b,0); off=32
SR=4*(3+9+9+4)+16; DR=64+24+16+28+20+64+64+16*SR; VS=(3+3+4+8)*4+4
print('draws',nd,'verts',nv,'tex',nt,'SR',SR,'DR',DR)
ds=[]
vo=off+nd*DR
for i in range(nd):
    o=off+i*DR
    proj=struct.unpack_from('16f',b,o); o2=o+64+24
    pt,st,fv,vc=struct.unpack_from('4I',b,o2); o2+=16
    bm,bs,bd,du,dc,cull,cu=struct.unpack_from('7I',b,o2); o2+=28
    prim,vcnt,hn,hc,lit=struct.unpack_from('5I',b,o2); o2+=20
    regs=struct.unpack_from('16f',b,o2); o2+=64+64
    stages=[struct.unpack_from('25I4f',b,o2+s*SR) for s in range(min(st,16))]
    if pt!=0 or vc==0: continue
    mn=[1e9]*3;mx=[-1e9]*3;ras=[0]*4
    for v in range(vc):
        p=struct.unpack_from('3f',b,vo+(fv+v)*VS); r=struct.unpack_from('4f',b,vo+(fv+v)*VS+24)
        for c in range(3): mn[c]=min(mn[c],p[c]);mx[c]=max(mx[c],p[c])
        for c in range(4): ras[c]+=r[c]/vc
    ctr=[(mn[c]+mx[c])/2 for c in range(3)]; rad=max(mx[c]-mn[c] for c in range(3))/2
    refl=[s[25-1] for s in stages]
    add=bm==1 and bd==1
    ds.append(dict(i=i,ctr=ctr,rad=rad,tri=vc//3,add=add,bm=bm,bs=bs,bd=bd,du=du,refl=refl,ras=ras,st=st,tex=[s[2] for s in stages],k=[s[25:] for s in stages],dist=math.dist(ctr,[0,0,0])))
print('perspective',len(ds))
print('--additive')
for d in ds:
    if d['add']: print(d['i'],'ctr %.1f %.1f %.1f rad %.1f dist %.0f tri %d st %d du %d ras %.2f %.2f %.2f %.2f tex %s k %s'%(*d['ctr'],d['rad'],d['dist'],d['tri'],d['st'],d['du'],*d['ras'],[t if t<1e6 else -1 for t in d['tex']],[tuple(round(x,2) for x in k) for k in d['k'][:2]]))
print('--reflective, small')
for d in ds:
    if any(r==1 for r in d['refl']) and d['rad']<40: print(d['i'],'ctr %.1f %.1f %.1f rad %.1f dist %.0f tri %d bm %d'%(*d['ctr'],d['rad'],d['dist'],d['tri'],d['bm']))
