#!/usr/bin/env python3
# img.py png2raw IN.png OUT.raw | raw2png IN.raw OUT.png  (raw: u32 w, h, then RGBA8)
import sys, struct
from PIL import Image
if sys.argv[1]=='png2raw':
    im=Image.open(sys.argv[2]).convert('RGBA')
    open(sys.argv[3],'wb').write(struct.pack('II',im.width,im.height)+im.tobytes())
else:
    b=open(sys.argv[2],'rb').read(); w,h=struct.unpack('II',b[:8])
    im=Image.frombytes('RGBA',(w,h),b[8:]); im.convert('RGB').save(sys.argv[3])
