#!/usr/bin/env python3
# ov.py RUN COPY OUT: marks the frame's lights on its picture: E old engine rule (cyan), M merged lights (yellow, size by weight)
import sys
from PIL import Image,ImageDraw
r,c,o=sys.argv[1],int(sys.argv[2]),sys.argv[3]
R=f'/mnt/d/FZero/aurora-rt/runs/{r}/'
im=Image.open(R+f'aurora-aurora-{c:06d}.png').convert('RGB');W,H=im.size;d=ImageDraw.Draw(im)
cur=None
for l in open(R+'lights.csv'):
  w=l.strip().split(',')
  if w[0]=='F': cur=int(w[1]);P=[float(x) for x in w[2:]] if cur==c else None;continue
  if cur!=c or w[0] not in 'EM': continue
  try: x,y,z=map(float,w[1:4])
  except: continue
  if z>=0: continue
  # view space to screen: the projection's NDC, the viewport's EFB (the picture shows all of 640 wide)
  nx=(P[0]*x+P[2]*z)/-z; ny=(P[1]*y+P[3]*z)/-z
  ex=P[4]+(nx+1)/2*P[6]; ey=P[5]+(1-ny)/2*P[7]
  sx=ex/P[6]*W; sy=(ey-P[5])/P[7]*H
  rad=6 if w[0]=='E' else 4+min(40,float(w[7])*3)
  d.ellipse((sx-rad,sy-rad,sx+rad,sy+rad),outline=(0,255,255) if w[0]=='E' else (255,255,0),width=2)
im.save(o,quality=90)
