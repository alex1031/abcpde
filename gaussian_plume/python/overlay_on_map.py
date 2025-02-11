import matplotlib
matplotlib.use('agg')
from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np
import os

import getpass

username=getpass.getuser()

def overlay_on_map(x,y,C1):
    import scipy.io as sio
    if not os.path.exists('/tmp/' + username):
       os.mkdir('/tmp/' + username)

    # Overlay concentrations on map
    plt.ion()
    mat_contents = sio.loadmat('map_green_lane')
    plt.figure()
    plt.imshow(mat_contents['A'], \
       extent=[np.min(mat_contents['ximage']), \
               np.max(mat_contents['ximage']), \
               np.min(mat_contents['yimage']), \
               np.max(mat_contents['yimage'])])
               
    plt.xlabel('x (m)');
    plt.ylabel('y (m)');
    cs=plt.contour(x,y,np.mean(C1,axis=2)*1e6, cmap='hot')
    plt.clabel(cs,cs.levels,inline=True, fmt='%.1f',fontsize=5)
    plt.show()

    plt.savefig("/tmp/" + username +  "/overlay.png")
    return
    
if __name__ == "__main__":
    overlay_on_map(x,y,C1)
