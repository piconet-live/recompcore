import struct,sys
b=open(sys.argv[1],'rb').read()
mag,nd,nv,nt=struct.unpack_from('4I',b,0); off=32; DR=2136; VS=76; vo=off+nd*DR
g={}
prev=None
for i in range(nd):
    o=off+i*DR
    p=struct.unpack_from('16f',b,o); pt,st,fv,vc=struct.unpack_from('4I',b,o+88)
    bm,bs,bd,du,dc,cu,cc=struct.unpack_from('7I',b,o+104)
    k=(pt,round(p[0],3),round(p[5],3),round(p[2],3),round(p[6],3),round(p[10],4),round(p[11],2))
    r=g.setdefault(k,dict(first=i,last=i,n=0,tri=0,zmin=1e9,zmax=-1e9,rmax=0,du=0,tex=set()))
    r['last']=i;r['n']+=1;r['tri']+=vc//3;r['du']+=du
    for v in range(0,vc,max(1,vc//8)):
        x=struct.unpack_from('3f',b,vo+(fv+v)*VS); r['zmin']=min(r['zmin'],x[2]);r['zmax']=max(r['zmax'],x[2]);r['rmax']=max(r['rmax'],(x[0]**2+x[1]**2+x[2]**2)**.5)
    r['tex'].add(struct.unpack_from('I',b,o+280+8)[0] if st else -1)
for k,r in g.items(): print(k,{a:(round(v,1) if isinstance(v,float) else v) for a,v in r.items() if a!='tex'},'tex',sorted(r['tex'])[:8])
