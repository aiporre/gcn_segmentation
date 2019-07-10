import os

def savefigs(fig_name, fig_dir, fig):
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    fig.savefig(os.path.join(fig_dir,str(fig_name) + '.png'), format='png')
