# comp.py TRACED RASTER_WITH_HUD RASTER_NO_HUD OUT: the game's HUD over the traced picture (the live pass runs before the HUD; here a mask stands in)
import sys
from PIL import Image, ImageChops, ImageFilter
t,r,n=[Image.open(p).convert('RGB') for p in sys.argv[1:4]]
d=ImageChops.difference(r,n).convert('L').point(lambda v:255 if v>3 else 0).filter(ImageFilter.MaxFilter(3))
Image.composite(r,t,d).save(sys.argv[4])
