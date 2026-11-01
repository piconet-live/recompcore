#!/usr/bin/env python3
# rtrun.py NAME COURSE MACHINE SECONDS SHOT [KEY=VAL ...]: writes a lone-race launch file (AI drives seat 0, 29 CPUs, modern HUD, true widescreen) from the owner's template
import json,sys
n,c,m,shot=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),sys.argv[4]
H='D:\\FZero\\aurora-rt\\'
t=open('/mnt/d/FZero/aurora-mi19/port/owner-mi19.template.json').read().replace('COURSE',str(c)).replace('HUDMETER','blade').replace('COLOURBLIND','off')
d=json.loads(t)
d['about']='RT still: lone race, the AI drives'
d['match']['match']['setup']['seats'][0]['machine']=m
d['match']['match']['setup']['seats'][0].pop('color',None)
d['bundle']='D:\\FZero\\aurora-mi19\\port\\bundle-mi18b'
d['debug']={'auroraLog':H+'runs\\'+n+'\\aurora','autopilot':True,'windowNoActivate':True,'automation':True,'pad':'0:0','auroraShots':[int(s) for s in shot.split(',')],'paceLog':H+'runs\\'+n+'\\pace.csv'}
import os
if os.environ.get('GRID')=='back':
  g=d['match']['match']['setup']['grid']; g[:]=list(reversed(g))
json.dump(d,open('/mnt/d/FZero/aurora-rt/runs/'+n+'/launch.json','w'),indent=1)
