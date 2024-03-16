"""
Script for create an animation to demonstrate steepest descent.

Author: Sivakumar Balasubramanian
Date: 16 March 2024
"""

import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import aladaopt

mpl.rc('font',**{'family':'Helvetica', 'sans-serif': 'Helvetica'})
mpl.rcParams['toolbar'] = 'None' 


def reset_params():
    global Xk, ak, k
    # Reset the solution
    Xk = np.random.rand(2, 1) * 8 - 4
    ak = 2
    k = 0


def update():
    global Xk, Xbt, k, ak
    _xk = Xk[:, -1].reshape(-1, 1)
    _grad = funclass.grad(_xk[0, 0], _xk[1, 0])
    ak = exact_search(ak=ak, dk=_grad)
    return aladaopt.GradientDescent.update(_xk, ak, _grad)


def get_function_along_dir(xk, dirvec):
    _t = np.linspace(-10, 10, 501)
    _x = xk[0] + _t * dirvec[0]
    _y = xk[1] + _t * dirvec[1]
    return _t, funclass.func(_x, _y)


def get_search_lines(xk, dirvec):
    _x = xk[0] + 10 * np.array([-dirvec[0], dirvec[0]])
    _y = xk[1] + 10 * np.array([-dirvec[1], dirvec[1]])
    return _x, _y


def exact_search(ak, dk):
    global Xk, tau, k
    # Check there is previous backtracked solution.
    _xk = Xk[:, -1].reshape(-1, 1)
    dk = funclass.grad(_xk[0, 0], _xk[1, 0])
    _fk = funclass.func(_xk[0, 0], _xk[1, 0])
    while True:
        # Get the new point
        _xb = _xk - ak * dk
        # Find the function value and gradient at the new point.
        _fb = funclass.func(_xb[0, 0], _xb[1, 0])
        _gb = funclass.grad(_xb[0, 0], _xb[1, 0])
        _gbd = (dk.T @ _gb)[0, 0]
        # Check current function value is less than the previous one, and if 
        # the gradient .
        if (_fb <= _fk and np.abs(_gbd) <= 1e-2): return ak
        # Change step-size
        k += 1
        dak = 0.001 * np.min([1, np.max([-1, _gbd])])
        ak = np.max([0.01, ak + dak])


def update_text():
    global k, ak
    # Remove the previous text
    ax2.cla()
    ax2.axis('off')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-1.1, 1.2)

    # Text positions
    xpos, ypos, delypos = 0.1, 0.8, 0.1
    
    # Instruction.
    ax2.text(0.1, 1.2,
             'Use the right arrow key to iterate.',
             fontsize=12, verticalalignment='center',
             horizontalalignment='left', color='gray', style='italic')
    ax2.text(0.1, 1.12,
             r'Use 0-2 to select function $f(\mathbf{x})$.',
             fontsize=12, verticalalignment='center',
             horizontalalignment='left', color='gray', style='italic')
    ax2.text(0.1, 1.04,
             r'Use the left mouse click to select a location.',
             fontsize=12, verticalalignment='center',
             horizontalalignment='left', color='gray', style='italic')
    ax2.text(0.1, 0.96,
             "Use 'r' to reset search.",
             fontsize=12, verticalalignment='center',
             horizontalalignment='left', color='gray', style='italic')
    
    j = 0
    ax2.text(xpos, ypos - j * delypos, f"Steepest Descent",
             fontsize=14, backgroundcolor='tab:red', color='white')

    # Method details.
    # xpos, ypos, delypos = 0.1, 1.1, 0.1 
    j = 0
    ax2.text(xpos, ypos - j * delypos, f"Gradient Descent: Backtracking",
             fontsize=14, backgroundcolor='tab:red', color='white')

    # Iteration
    j += 1
    ax2.text(xpos, ypos - j * delypos, f"Iteration $k = $" + f"{k}",
            fontsize=14)
    j += 1
    ax2.text(xpos, ypos - j * delypos,
             f"Step size " + f"$\\alpha_{{{k}}} = $" + f"{ak:.3f}",
             fontsize=14)
    
    # Minimum point
    j += 1
    _xmin = f"{np.array2string(funclass.xmin.T[0], precision=3, floatmode='fixed')}"
    ax2.text(xpos, ypos - j * delypos, f"$\\mathbf{{x}}^{{\star}}$ = " + _xmin + r"$^\top$", fontsize=14)

    # Current point
    j += 1
    _xk = f"{np.array2string(Xk[:, -1].T, precision=3, floatmode='fixed')}"
    ax2.text(xpos, ypos - j * delypos, f"$\\mathbf{{x}}_{{{k}}}$ = " + _xk + r"$^\top$", fontsize=14)
    
    # Current function value
    j += 1
    _fk = f"{funclass.func(Xk[0, -1], Xk[1, -1]):.3f}"
    ax2.text(xpos, ypos - j * delypos, f"$f\\left(\\mathbf{{x}}_{{{k}}}\\right) = {_fk}$", fontsize=14)


def plot_contour():
    # Generate data for plotting
    x1 = np.linspace(-4, 4, 500)
    x2 = np.linspace(-4, 4, 500)
    X1, X2 = np.meshgrid(x1, x2)
    Z = funclass.func(X1, X2)

    # Plotting the contour
    if funclass.name == "Flipped Gaussian":
        ax.contour(X1, X2, Z, levels=np.logspace(-5, 5, 40),
                   cmap='Blues_r', linewidths=0.5)
    else:
        contours = ax.contour(X1, X2, Z, levels=np.logspace(-1, 5.5, 20),
                            cmap='Blues_r', linewidths=0.5)
    
    # Plot the three search directions
    _xk = Xk[:, -1].reshape(-1, 1)
    _grad = funclass.grad(_xk[0, 0], _xk[1, 0])
    _gradn = _grad / np.linalg.norm(_grad)
    _x, _y = get_search_lines(_xk[:, 0], _gradn[:, 0])
    
    # Current point
    ax.plot(_xk[0, 0], _xk[1, 0], 'tab:red', marker='o', markersize=6)
    ax.plot(Xk[0, :], Xk[1, :], 'tab:red', lw=1, alpha=0.8)
    ax.plot(_x, _y, 'black', lw=0.5, linestyle='--')
    ax.arrow(_xk[0, 0], _xk[1, 0], 0.5 * -_gradn[0, 0], 0.5 * -_gradn[1, 0],
             head_width=0.05, head_length=0.1, fc='black', ec='black')
    
    # Plot backtracking points if any.
    if Xbt is not None:
        ax.plot(Xbt[0, :], Xbt[1, :], 'tab:gray', marker='o', markersize=4,
                linestyle="None")

    # Remove top and right spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['bottom'].set_color('#dddddd')
    ax.spines['left'].set_color('#dddddd')
    
    # Plot the minimum point
    ax.plot(funclass.xmin[0], funclass.xmin[1], 'r*', markersize=10)

    # Set xlims and ylims
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)

    # Add labels and title
    ax.set_title(r"$f(\mathbf{x}) = $" + funclass.title, fontsize=18)


def plot_dist_to_min():
    axins.cla()
    # Update the inset plot.
    _k = Xk.shape[1]
    _dist = [np.linalg.norm(_x - funclass.xmin.T[0]) for _x in Xk.T]
    axins.plot(np.arange(_k), _dist, lw=1, color="tab:green")
    axins.set_xlim(0, (_k // 40 + 1) * 40)
    axins.set_ylim(np.min([-5, 1.1 * np.min(_dist)]),
                   np.max([5, 1.1 * np.max(_dist)]),)
    axins.spines["right"].set_visible(False)
    axins.spines["top"].set_visible(False)
    axins.spines["bottom"].set_position("zero")
    axins.spines["left"].set_position(("axes", -0.05))
    axins.spines['bottom'].set_color('#bbbbbb')
    axins.spines['left'].set_color('#bbbbbb')
    axins.tick_params(axis='both', colors='#bbbbbb')
    axins.set_title(r"$\Vert \mathbf{x}^{{\star}} - \mathbf{x}_k \Vert_2$",
                    color='#000', fontsize=16)


# Handling mouse click events
def on_mouse_click(event):
    # Set xk to the mouse click location
    global Xk, ax, k, fig
    # Reset the solution
    Xk = np.array([event.xdata, event.ydata]).reshape(-1, 1)
    ak = 2
    k = 0
    ax.cla()
    
    # Draw the plot and text
    update_text()
    plot_contour()
    plot_dist_to_min()
    fig.canvas.draw()


# Handling key press events
def on_press(event):
    global fig, funclass, methodclass, Xk, Xbt, ak, tau

    # Close figure if escaped.
    if event.key == 'escape':
        plt.close(fig)
        return

    # Choose which function to minimize.
    if event.key in function_dict.keys():
        funclass = function_dict[event.key]
        # Reset variables
        reset_params()
        ax.cla()
    
    # Return if no function has been selected.
    if funclass is None:
        return
    
    # Chekc if the solution needs to be updated.
    if event.key == 'right':
        # Compute the next step        
        # Update the solution
        _xk1 = update()
        if _xk1 is not None:
            Xk = np.hstack((Xk, _xk1))
        # Reset for backtracking
        Xbt = None
        # Clear axis
        ax.cla()
    elif event.key in ['r', 'R']:
        # Reset the plot
        reset_params()
        ax.cla()
    
    # Draw the plot and text
    update_text()
    plot_contour()
    plot_dist_to_min()
    fig.canvas.draw()
    
    # Save plot
    if event.key == 'ctrl+s':
        fig.savefig("multivar_steepdesc.png", dpi=300, bbox_inches='tight')
        fig.savefig("multivar_steepdesc.pdf", bbox_inches='tight')


if __name__ == "__main__":
    # Function dictionary
    function_dict = {
        '0': aladaopt.Circle(xmin=np.array([2, 1])),
        '1': aladaopt.Ellipse(xmin=np.array([1, 0.5]),
                            Q=np.array([[3, 1], [1, 2]])),
    }

    # Function and Method ID
    funclass = function_dict["0"]

    # Create the figure and the axis.
    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(1, 3, width_ratios=(1.2, 0.1, 1))
    ax = fig.add_subplot(gs[0, 0])
    ax.equal_aspect = True
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    axins = inset_axes(ax2, width="80%", height="40%", loc=4, borderpad=1)
    axins.axis('off')

    # Initialize the solution
    Xk = None
    Xbt = None
    ak = 2
    k = 0
    tau = 0.99
    reset_params()
    
    # Draw the plot and text
    update_text()
    plot_contour()
    plot_dist_to_min()
    fig.canvas.draw()

    # Create the figure and the axis.
    fig.canvas.manager.set_window_title('ALADA Optimization Animations: Steepest Descent')
    fig.canvas.mpl_connect('key_press_event', on_press)
    fig.canvas.mpl_connect('button_press_event', on_mouse_click)
    plt.tight_layout(pad=3)
    plt.show()