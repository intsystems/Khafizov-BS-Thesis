import numpy as np
from matplotlib import pylab as plt

def generate_matrix(d, mu, L):
    diag = (L - mu) * np.random.random_sample(d) + mu
    sigma = np.diag(diag)
    sigma[0][0] = L
    sigma[d - 1][d - 1] = mu
    rand_matrix = np.random.rand(d, d)
    rand_ort, _, _ = np.linalg.svd(rand_matrix)
    matrix = rand_ort.T @ sigma @ rand_ort
    return matrix

def quadratic_func(x, A, b, c):
    return 1./2 * x.T @ A @ x - b.T @ x + c

def quadratic_grad(x, A, b):
    return A @ x - b

def logreg(x, A, b, alpha):
    p = 1 / (1 + np.exp(-A @ x))
    return np.mean(-b * np.log(p) - (1 - b) * np.log(1 - p)) + 0.5 * alpha * np.linalg.norm(x)**2

def logreg_grad(x, A, b, alpha):
    p = 1 / (1 + np.exp(-A @ x))
    return alpha * x + np.mean((p - b)[:, np.newaxis] * A, axis=0)

def make_err_plot(optimizers_list, labels=None, title=None, markers=None, colors=None, save_name=None):
    if markers is None:
        markers = [None] * 100
    if colors is None:
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'black', 'olive', 'pink', 'brown']

    x_label = "The number of bits used"
    if optimizers_list[0].x_sol is not None:
        y_label = r'$\frac{||x^k - x^*||}{||x^0 - x^*||}$'
    else:
        y_label = r'$\frac{||\nabla f(x^k)||}{||\nabla f(x^0)||}$'

    fig, ax = plt.subplots(figsize=(10, 7))
    if title is not None and title != '':
        ax.set_title(title + "\n logarithmic scale on the axis y")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    for optimizer, label, color, marker in zip(optimizers_list, labels, colors, markers):
        bits_used = optimizer.bits_list
        errors = optimizer.errors_list
        ax.semilogy(bits_used, errors, color=color, label=label, markevery=100, marker=marker)

    ax.legend()
    ax.grid(True)
    
    if save_name is not None:
        plt.savefig(f"figures/{save_name}.pdf", format='pdf')
        plt.savefig(f"figures/{save_name}.png", format='png')
        
    plt.show()
    
def plot_graphs(data, columns_to_plot, save_name=None, plot_params=None, ticks_params=None):

    for column in columns_to_plot:
        params = plot_params.get(column, {})
        plt.plot(data['Step'], data[column], label=params.get('label', column), markevery=100,
                 marker=params.get('marker', 'o'),
                 color=params.get('color', None))

    plt.xlabel('Information sent')
    plt.ylabel(r'$\frac{f(x^k)-f(x^*)}{f(x^0) - f(x^*)}$')
    plt.title('Convergence of GD of MNIST')
    plt.legend()
    plt.grid(True)

    if save_name is not None:
        plt.savefig(output_filename + '.pdf', format='pdf')
        plt.savefig(output_filename + '.png', format='png')

    plt.show()

