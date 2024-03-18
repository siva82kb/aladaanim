"""
Script for create an animation to demonstrate how the line search direction 
impacts the optimization process.

Author: Sivakumar Balasubramanian
Date: 14 March 2024
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import platform
if platform.system() == "Windows":
    mpl.rc('font',**{'family':'Times New Roman', 'sans-serif': 'Arial'})
else:
    mpl.rc('font',**{'family':'Helvetica', 'sans-serif': 'Helvetica'})
mpl.rcParams['toolbar'] = 'None' 

import aladaopt


def reset_params():
    global funclass, xk, theta_grad, theta_dir
    # Reset the solution
    xk = np.random.rand(2) * 8 - 4
    theta_grad = get_theta_grad()
    theta_dir = 0


def get_theta_grad():
    global funclass, xk
    _grad = funclass.grad(xk[0], xk[1])
    _gradn = _grad / np.linalg.norm(_grad)
    return np.rad2deg(np.arctan2(_gradn[1, 0], _gradn[0, 0]))


def get_function_along_dir(xk, dirvec):
    _t = np.linspace(-10, 10, 501)
    _x = xk[0] + _t * dirvec[0]
    _y = xk[1] + _t * dirvec[1]
    return _t, funclass.func(_x, _y)


def get_function_along_theta(xk, dirang):
    _t = np.linspace(-10, 10, 501)
    _x = xk[0] + _t * np.cos(np.deg2rad(dirang))
    _y = xk[1] + _t * np.sin(np.deg2rad(dirang))
    return _t, funclass.func(_x, _y)


def get_search_line_along_dir(xk, dirvec):
    _x = xk[0] + 10 * np.array([-dirvec[0], dirvec[0]])
    _y = xk[1] + 10 * np.array([-dirvec[1], dirvec[1]])
    return _x, _y


def get_search_line_along_theta(xk, theta):
    _t = np.deg2rad(theta)
    _x = xk[0] + 10 * np.array([-np.cos(_t), np.cos(_t)])
    _y = xk[1] + 10 * np.array([-np.sin(_t), np.sin(_t)])
    return _x, _y


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
    _x, _y = get_search_line_along_theta(xk, theta_dir)
    ax.plot(xk[0], xk[1], 'tab:red', marker='o', markersize=10)
    ax.plot(_x, _y, 'tab:red', lw=0.5, linestyle='--', label=r"Search Direction $\mathbf{d}_{\theta}$")
    ax.arrow(xk[0], xk[1], np.cos(np.deg2rad(theta_dir)),
             np.sin(np.deg2rad(theta_dir)),
             head_width=0.05, head_length=0.1, fc='tab:red', ec='tab:red')
    
    # Plot the three search directions
    _grad = funclass.grad(xk[0], xk[1])
    _gradn = _grad / np.linalg.norm(_grad)
    _x, _y = get_search_line_along_dir(xk, _gradn[:, 0])
    
    # Current point
    ax.plot(_x, _y, 'black', lw=0.5, linestyle='--', label=r"Gradient Direction $\nabla f(\mathbf{x})$")
    ax.arrow(xk[0], xk[1], _gradn[0, 0], _gradn[1, 0],
             head_width=0.05, head_length=0.1, fc='black', ec='black')
    
    # Plot the minimum point
    ax.plot(funclass.xmin[0], funclass.xmin[1], 'r*', markersize=10)

    # Remove top and  right spines
    ax.axis('off')

    # Set xlims and ylims
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)

    # Legend
    ax.legend(loc='upper left', fontsize=14, frameon=False)

    # Add labels and title
    ax.set_title(r"$f(\mathbf{x}) = $" + funclass.title, fontsize=18)

    # Update function plot along the search direction
    axins.cla()
    _dir ,_fdir = get_function_along_dir(xk, _gradn[:, 0])
    axins.plot(_dir[250], _fdir[250], 'tab:red', marker='o', markersize=3)
    axins.plot(_dir, _fdir, 'black', lw=2.0, alpha=0.7, label=r"$\phi_{\nabla f}(t)$")
    _dir ,_fdir = get_function_along_theta(xk, theta_dir)
    axins.plot(_dir, _fdir, 'tab:red', lw=2.0, alpha=0.7, label=r"$\phi_{\mathbf{d}_\theta}(t)$")
    # Plot the derivative of the function along the direction theta_dir
    _dx = _dir[1] - _dir[0]
    _dy = _fdir[251] - _fdir[250]
    _m = _dy / _dx
    _x = np.array([-10, 10])
    _y = _m * _x + _fdir[250]
    axins.plot(_x, _y, 'tab:blue', lw=1.0, linestyle='-', label=r"$\frac{d\phi_{\mathbf{d}_{\theta}}(0)}{dt}$")

    # Remove top and right spines
    axins.spines['right'].set_visible(False)
    axins.spines['top'].set_visible(False)
    axins.spines.bottom.set_position(('axes', -0.02))    
    axins.spines['bottom'].set_color('#bbbbbb')
    axins.spines['left'].set_color('#bbbbbb')

    # Legend
    axins.legend(loc='upper right', fontsize=12, frameon=False, bbox_to_anchor=(1.3, 1))

    # Label
    axins.set_xlabel(r"$t$", fontsize=12)

    # Set xlims and ylims
    axins.set_xlim(-10, 10)
    _delta = 0.05 * np.abs(np.max(Z) - np.min(Z))
    axins.set_ylim(np.min(Z) - _delta,
                   np.max(Z) + _delta)
    
    # Set title
    axins.set_title(r"$\phi_{\mathbf{d}}(t) = f(\mathbf{x} + t\mathbf{d})$", fontsize=18)


def update_text():
    # Remove the previous text
    ax2.cla()
    ax2.axis('off')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-1.1, 1.2)

    # Instruction.
    ax2.text(0.1, 1.2,
             'Use the up/down arrow change direction.',
             fontsize=12, verticalalignment='center',
             horizontalalignment='left', color='gray', style='italic')
    ax2.text(0.1, 1.12,
             "Use 'r' to set a random location.",
             fontsize=12, verticalalignment='center',
             horizontalalignment='left', color='gray', style='italic')
    ax2.text(0.1, 1.04,
             r'Use the left mouse click to select a location.',
             fontsize=12, verticalalignment='center',
             horizontalalignment='left', color='gray', style='italic')
    # Current location
    _xk = f"{np.array2string(xk, precision=3, floatmode='fixed')}"
    ax2.text(0.1, 0.85, "$\\mathbf{{x}}$ = " + _xk + r"$^\top$", fontsize=14)


# Handling mouse click events
def on_mouse_click(event):
    # Set xk to the mouse click location
    global xk, ax, fig
    xk = np.array([event.xdata, event.ydata])
    ax.cla()
    
    # Draw the plot and text
    plot_contour()
    update_text()
    fig.canvas.draw()


# Handling key press events
def on_press(event):
    global funclass, xk, theta_dir

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
    

    # Check if the step size needs to be updated.
    if event.key == 'up':
        theta_dir += 5
    elif event.key == 'down':
        theta_dir -= 5
    if event.key == 'shift+up':
        theta_dir += 0.1
    elif event.key == 'shift+down':
        theta_dir -= 0.1
    # Clear axis
    ax.cla()
    
    # Reset the plot
    if event.key in ['r', 'R']:
        # Reset the plot
        reset_params()
        ax.cla()
    
    # Draw the plot and text
    plot_contour()
    update_text()
    fig.canvas.draw()
    
    # Save plot
    if event.key == 'ctrl+s':
        fig.savefig("multivar_linesearch_demo.png", dpi=300, bbox_inches='tight')
        fig.savefig("multivar_linesearch_demo.pdf", bbox_inches='tight')


if __name__ == "__main__":
    # Function dictionary
    function_dict = {
        '0': aladaopt.Circle(xmin=np.array([2, 1])),
        '1': aladaopt.Ellipse(xmin=np.array([1, 0.5]),
                            Q=np.array([[3, 1], [1, 2]])),
        '2': aladaopt.Rosenbrock(a=1, b=5),
        '3': aladaopt.Quartic(xmin=np.array([2, 1]), a=2, b=5, c=3),
        '4': aladaopt.FlippedGaussian(xmin=np.array([0, 0]),
                                      Q=np.linalg.inv(np.array([[5, 2], [2, 3]])))
    }

    # Method dictionary
    method_dict = {
        'ctrl+0': aladaopt.GradientDescent,
        'ctrl+1': aladaopt.NewtonRaphson,
        'ctrl+2': aladaopt.LevenbergMarquardt
    }

    # Function and Method ID
    funclass = function_dict['0']
    # methodclass = None

    # Create the figure and the axis.
    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(1, 3, width_ratios=(1.2, 0.1, 1))
    ax = fig.add_subplot(gs[0, 0])
    ax.equal_aspect = True
    # ax.axis('off')
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    axins = inset_axes(ax2, width="75%", height="30%", loc=3, borderpad=1)
    axins.axis('off')

    # Initialize the solution
    theta_grad = 0
    theta_dir = 0
    xk = None
    reset_params()

    # Plot stuff
    update_text()
    plot_contour()
    fig.canvas.draw()

    # Create the figure and the axis.
    fig.canvas.manager.set_window_title('ALADA Optimization Animations: Line Search')
    fig.canvas.mpl_connect('key_press_event', on_press)
    fig.canvas.mpl_connect('button_press_event', on_mouse_click)
    plt.tight_layout(pad=3)
    plt.show()