#!/usr/bin/env python3
# pace.py RUN [FROM TO]: presenter time per frame and real pictures a second over a copy range (runs/RUN/aurora)
import sys
r=sys.argv[1];a=int(sys.argv[2]) if len(sys.argv)>2 else 1210;b=int(sys.argv[3]) if len(sys.argv)>3 else 1450
rows=[l.strip().split(',') for l in open(f'/mnt/d/FZero/aurora-rt/runs/{r}/aurora',errors='replace') if l[:1].isdigit() and l.count(',')==7]
rows=[[float(x) for x in w] for w in rows if a<=int(w[0])<b]
p=sorted((w[4]-w[2])/1000 for w in rows)
sp=(rows[-1][4]-rows[0][4])/1e6
print(f'{r}: copies {a}-{b}: {len(rows)} pictures in {sp:.2f} s = {(len(rows)-1)/sp:.1f}/s; of {b-a} copies; present mean {sum(p)/len(p):.2f} p95 {p[int(len(p)*.95)]:.2f} worst {p[-1]:.2f} ms')
