import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as shapes

sensors = [(8, 15),(15, 15),(22, 15),(15, 22)]

def plot_likelihood(cur_pos, bel):
    _, ax = plt.subplots()
    position_marker = shapes.Circle((cur_pos), radius=0.5, color='red')

    #plotting beleifs

    for i in range(30):
        for j in range(30):
            c = bel[i, j]
            like_marker = shapes.Rectangle((i-0.5, j-0.5), width=1, height=1, color=f'{1-c}')
            ax.add_artist(like_marker)
    for sens in sensors:
        x, y = sens
        marker = shapes.Rectangle((x, y - 0.5), width=0.5*np.sqrt(2), height=0.5*np.sqrt(2), angle=45, color='blue')
        ax.add_artist(marker)
    
    ax.set_box_aspect(1)
    ax.add_artist(position_marker)
    ax.set_xlim(-0.5,29.5)
    ax.set_ylim(-0.5, 29.5)
    ax.set_xticks(np.arange(-0.5,29.5,1))
    ax.set_yticks(np.arange(-0.5,29.5,1))
    ax.grid(b=True, which='major')
    plt.show()

if __name__ == '__main__':
    bel = np.ones((30,30), dtype=np.float64)
    plot_likelihood((2,2), np.random.rand(30, 30))
