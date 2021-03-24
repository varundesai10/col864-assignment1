import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.patches as shapes

sensors = [(8, 15),(15, 15),(22, 15),(15, 22)]
def d(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def plot_likelihood(cur_pos, bel, estimated_position=None, log=True, filename=None, title=None, error=True):
    _, ax = plt.subplots()
    bel = bel.copy()

    position_marker = shapes.Circle((cur_pos), radius=0.5, color='red')
    
    print(np.max(bel), np.min(bel))
    
    bel = np.abs(np.log(bel + 1e-12))
    bel /= np.max(bel)

    print(np.max(bel), np.min(bel))
    #plotting beleifs

    for i in range(30):
        for j in range(30):
            c = bel[i, j]
            like_marker = shapes.Rectangle((i-0.5, j-0.5), width=1, height=1, color=f'{c}')
            ax.add_artist(like_marker)
    for sens in sensors:
        x, y = sens
        marker = shapes.Rectangle((x, y - 0.5), width=0.5*np.sqrt(2), height=0.5*np.sqrt(2), angle=45, color='blue')
        ax.add_artist(marker)
    if estimated_position:
        x_est, y_est = estimated_position
        marker = shapes.Rectangle((x_est, y_est - 0.5), width=0.5*np.sqrt(2), height=0.5*np.sqrt(2), angle=45, color='green')
        ax.add_artist(marker)
    ax.set_box_aspect(1)
    ax.add_artist(position_marker)
    ax.set_xlim(-0.5,29.5)
    ax.set_ylim(-0.5, 29.5)
    ax.set_xticks(np.arange(-0.5,29.5,1))
    ax.set_yticks(np.arange(-0.5,29.5,1))
    ax.grid(b=True, which='major')
    ax.set_title(title)
    if(error and estimated_position is not None):
        ax.set_xlabel(f'Manhattan Distance = {d(cur_pos, estimated_position)}')
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


def plot_paths(true_path, estimated_path, filename=None, title=None):
    assert len(true_path) == len(estimated_path)
    _, ax = plt.subplots(1, 2)

    
    ax[0].set_box_aspect(1)
    ax[1].set_box_aspect(1)
    i = 1
    for pos1, pos2 in zip(true_path, estimated_path):
        marker1 = shapes.Circle(pos1, radius=0.5, color='red')
        marker2 = shapes.Circle(pos2, radius=0.5, color='blue')
        ax[0].add_artist(marker1)
        ax[0].text(pos1[0], pos1[1], f'{i}')
        ax[1].add_artist(marker2)
        ax[1].text(pos2[0], pos2[1], f'{i}')
        i+=1

    for sens in sensors:
        x, y = sens
        marker = shapes.Rectangle((x, y - 0.5), width=0.5*np.sqrt(2), height=0.5*np.sqrt(2), angle=45, color='blue')
        ax[1].add_artist(marker); 

        marker = shapes.Rectangle((x, y - 0.5), width=0.5*np.sqrt(2), height=0.5*np.sqrt(2), angle=45, color='blue')
        ax[0].add_artist(marker)

    
    ax[0].set_xlim(-0.5,29.5)
    ax[0].set_ylim(-0.5, 29.5)
    ax[0].set_xticks(np.arange(-0.5,29.5,1))
    ax[0].set_yticks(np.arange(-0.5,29.5,1))
    ax[0].grid(b=True, which='major')
    ax[0].set_title('True Positions')
    
    ax[1].set_xlim(-0.5,29.5)
    ax[1].set_ylim(-0.5, 29.5)
    ax[1].set_xticks(np.arange(-0.5,29.5,1))
    ax[1].set_yticks(np.arange(-0.5,29.5,1))
    ax[1].grid(b=True, which='major')
    ax[1].set_title('Estimated Positions')
    
    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()


if __name__ == '__main__':
    bel = np.ones((30,30), dtype=np.float64)
    plot_likelihood((2,2), np.random.rand(30, 30))
