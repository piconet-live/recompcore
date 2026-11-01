import sys
from PIL import Image
i=Image.open(sys.argv[1]); x,y,w,h=map(int,sys.argv[3:7]); s=int(sys.argv[7]) if len(sys.argv)>7 else 1
i.crop((x,y,x+w,y+h)).resize((w*s,h*s),Image.LANCZOS).save(sys.argv[2])
