import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
plt.rcParams.update({'font.family':'Liberation Sans','font.size':8,'axes.titlesize':8.5,'axes.titleweight':'bold','axes.labelsize':8,
 'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':6.5,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight','savefig.pad_inches':0.04})
def save(fig,name):
    fig.savefig(f'../figures/{name}.png',dpi=300)
    fig.savefig(f'../figures/{name}_600.png',dpi=600)
    im=Image.open(f'../figures/{name}_600.png').convert('RGB')
    im.save(f'../figures/{name}.tif',compression='tiff_lzw',dpi=(600,600))
    import os; os.remove(f'../figures/{name}_600.png')
    plt.close(fig)
def letter(ax,s): ax.set_title(s+' '+ax.get_title(),loc='left' if False else 'center')
