import math,collections,sys
F=[];cur=None
for l in open(sys.argv[1]):
  w=l.strip().split(',')
  try:
    if w[0]=='F': cur={'c':int(w[1]),'F':[float(x) for x in w[2:]],'E':[],'M':[],'C':[]}; F.append(cur)
    elif cur is not None: cur[w[0]].append([float(x) for x in w[1:]])
  except ValueError: pass
print('frames',len(F))
hit=tot=0;off=[];miss=collections.Counter();totd=collections.Counter();mx=[];why=[]
for f in F:
  mx.append(len(f['M']))
  for e in f['E']:
    tot+=1;dist=math.dist(e[:3],[0,0,0]);totd[int(dist//10*10)]+=1
    d=[math.dist(e[:3],m[:3]) for m in f['M']]
    if d and min(d)<6: hit+=1;off.append(min(d))
    else:
      miss[int(dist//10*10)]+=1
      n=[c for c in f['C'] if math.dist(c[2:5],e[:3])<6]
      why.append((f['c'],round(dist,1),len(n),sorted(round(c[9],3) for c in n)[-3:],sorted(round(c[8],1) for c in n)[-3:],[(c[10],c[11],c[12],c[13]) for c in n][:2]))
off.sort();print('old engines',tot,'found',hit,'p50 off',off[len(off)//2] if off else None);print(' miss/total by distance',[(k,miss[k],totd[k]) for k in sorted(totd)][:10]);print(why[::max(1,len(why)//10)][:10])
mx.sort();print('merged p50/95/max',mx[len(mx)//2],mx[int(len(mx)*.95)],mx[-1])
ws=sorted(m[6] for f in F for m in f['M']);print('weights p10/50/90',ws[len(ws)//10],ws[len(ws)//2],ws[len(ws)*9//10])
we=[]
for f in F:
  for e in f['E']:
    d=[(math.dist(e[:3],m[:3]),m[6]) for m in f['M']]
    if d:
      x=min(d)
      if x[0]<6: we.append(x[1])
we.sort();print('weight at an engine p10/50/90',we[len(we)//10],we[len(we)//2],we[len(we)*9//10] if we else 0)
# what the non-engine lights near the camera are
near=collections.Counter()
for f in F:
  for m in f['M']:
    if math.dist(m[:3],[0,0,0])<60 and not any(math.dist(m[:3],e[:3])<6 for e in f['E']):
      near[(round(m[3],1),round(m[4],1),round(m[5],1))]+=1
print('near non-engine light colours',near.most_common(8))
# frames with strong line picks: spark candidates
sp=[]
for f in F:
  L=[c for c in f['C'] if c[0] in(168,176) and c[9]>.15 and math.dist(c[2:5],[0,0,0])<45]
  if L: sp.append((len(L),f['c'],round(min(math.dist(c[2:5],[0,0,0]) for c in L),1)))
sp.sort(reverse=True);print('frames with near line picks',len(sp),sp[:10])
