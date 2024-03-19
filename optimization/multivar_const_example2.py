"""
Script to demonstrate Lagrange multipliers method through example 2.

Author: Sivakumar Balasubramanian
Date: 19 March 2024
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
    global t
    # Reset the solution
    t = np.random.rand(1)[0] * 7 - 3.5
    update()


def update():
    global funclass, ecfunclass, t, xc, norms, tangs
    xc = ecfunclass.func(t)
    norms = ecfunclass.normal(xc[0, 0], xc[1, 0])
    tangs = ecfunclass.tangent(xc[0, 0], xc[1, 0])


def plot_contour():
    global funclass, ecfunclass, t, xc, norms, tangs
    # Generate data for plotting
    x1 = np.linspace(-4, 4, 500)
    x2 = np.linspace(-4, 4, 500)
    X1, X2 = np.meshgrid(x1, x2)
    Z = funclass.func(X1, X2)

    # Plotting the contour
    _levels = np.logspace(-1, np.log10(32), 10)
    contours = ax.contour(X1, X2, Z, levels=np.concatenate((-_levels[::-1], _levels)),
                          cmap='Greys_r', linewidths=0.25, linestyles='dashed')
    for c in contours.collections:
        c.set_dashes([(0, (5.0, 5.0))])
    
    # Find the value of the fucntion at the current equality constraint point
    _fxc = funclass.func(xc[0, 0], xc[1, 0])
    ax.contour(X1, X2, Z, levels=[_fxc], linewidths=1, colors='tab:green',
               linestyles='solid')
    
    # Plot gradient at the current point
    _grad = funclass.grad(xc[0, 0], xc[1, 0])
    _gradn = _grad / np.linalg.norm(_grad)
    ax.arrow(xc[0, 0], xc[1, 0], 0.5 * _gradn[0, 0], 0.5 * _gradn[1, 0], head_width=0.1,
             head_length=0.2, fc='tab:green', ec='tab:green')

    # h function.
    _t = np.linspace(-5, 5, 500)
    _h = np.hstack([ecfunclass.func(__t) for __t in _t])

    # Plot the search trajectory
    ax.plot(_h[0], _h[1], color='black', lw=2.0, alpha=0.6)

    # Current location.
    ax.plot(xc[0, 0], xc[1, 0], 'ko', markersize=5)
    
    # Plot tangent and normal spaces
    _p1 = - 10. * tangs
    _p2 = 10. * tangs
    ax.plot([_p1[0, 0], _p2[0, 0]], [_p1[1, 0], _p2[1, 0]], 'tab:red',
            lw=0.5, ls="dotted", alpha=0.5)
    print(_p1, xc)
    _p1 += xc
    _p2 += xc
    ax.plot([_p1[0, 0], _p2[0, 0]], [_p1[1, 0], _p2[1, 0]], 'tab:red',
            lw=1, alpha=0.6)
    _p1 = - 10. * norms
    _p2 = 10. * norms
    ax.plot([_p1[0, 0], _p2[0, 0]], [_p1[1, 0], _p2[1, 0]], 'tab:blue',
            lw=0.5, ls="dotted", alpha=0.5)
    _nsn = norms / np.linalg.norm(norms)
    ax.arrow(xc[0, 0], xc[1, 0], _nsn[0, 0], _nsn[1, 0], head_width=0.1,
             head_length=0.2, fc='tab:blue', ec='tab:blue')
    _p1 += xc
    _p2 += xc
    ax.plot([_p1[0, 0], _p2[0, 0]], [_p1[1, 0], _p2[1, 0]], 'tab:blue',
            lw=1, alpha=0.6)

    # Remove top and right spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['bottom'].set_color('#bbbbbb')
    ax.spines['left'].set_color('#bbbbbb')

    # Set xlims and ylims
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Add labels and title
    ax.set_title(ecfunclass.equation() + "\n", fontsize=18)


def plot_const_func():
    axins.cla()
    # Update the inset plot.
    _t = np.linspace(-4, 4, 500)
    _h = np.hstack([ecfunclass.func(__t) for __t in _t])
    _f = funclass.func(_h[0], _h[1])
    axins.plot(_t, _f, 'tab:red', lw=2.0, alpha=0.7, label=r"$f(\mathbf{x})$")
    axins.plot(t, funclass.func(xc[0, 0], xc[1, 0]), 'black', marker='o',
               markersize=4, alpha=0.5)
    # Axis limits
    axins.set_xlim(-4, 4)
    _fmin, _fmax = np.min(_f), np.max(_f)
    _delf = 0.1 * (_fmax - _fmin)
    # x and y lines
    axins.set_ylim(_fmin - _delf, _fmax + _delf)
    axins.axhline(y=funclass.func(xc[0, 0], xc[1, 0]), color='black',
                  lw=0.5, linestyle='--')
    axins.plot([t, t], [_fmin - _delf, funclass.func(xc[0, 0], xc[1, 0])], 'black',
                lw=0.5, linestyle='--')
    axins.spines["right"].set_visible(False)
    axins.spines["top"].set_visible(False)
    axins.spines["left"].set_position(("axes", -0.05))
    axins.spines['bottom'].set_color('#bbbbbb')
    axins.spines['left'].set_color('#bbbbbb')
    axins.tick_params(axis='both', colors='#bbbbbb')
    axins.set_xlabel(r"$t$", fontsize = 14)
    axins.set_title(r"$f\left( h\left( \mathbf{x}\left( t \right)\right)\right)$",
                    color='#000', fontsize=16)


def update_text():
    # Remove the previous text
    ax2.cla()
    ax2.axis('off')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(-1.1, 1.2)

    # Text positions
    xpos, ypos, delypos = 0.1, 0.35, 0.1

    # Instruction.
    ax2.text(0.1, 1.2,
             r'Use the right/left arrow key to move along $h\left(\mathbf{x}\right)$.',
             fontsize=12, verticalalignment='center',
             horizontalalignment='left', color='gray', style='italic')
    ax2.text(0.1, 1.12,
             r'Use 0-4 to select function $f(\mathbf{x})$.',
             fontsize=12, verticalalignment='center',
             horizontalalignment='left', color='gray', style='italic')
    ax2.text(0.1, 1.04,
             r'Use ctrl+0-2 to select equality constraint $h\left(\mathbf{x}\right)$.',
             fontsize=12, verticalalignment='center',
             horizontalalignment='left', color='gray', style='italic')
    ax2.text(0.1, 0.96,
             "Use 'r' to reset current point.",
             fontsize=12, verticalalignment='center',
             horizontalalignment='left', color='gray', style='italic')

    ax2.text(0.1, 0.8, "minimize", fontsize=14, verticalalignment='top',
             horizontalalignment='left')
    ax2.text(0.3, 0.8, "$- x_1 x_2$", fontsize=14, verticalalignment='top',
             horizontalalignment='left')
    ax2.text(0.1, 0.7, "subject to", fontsize=14, verticalalignment='top',
             horizontalalignment='left')
    ax2.text(0.3, 0.7, "$4x_1 + 8x_2 - 10 = 0$", fontsize=14, verticalalignment='top',
             horizontalalignment='left')

    # Current point detials
    j = 0
    _xc = f"{np.array2string(xc.T[0], precision=3, floatmode='fixed')}"
    ax2.text(xpos, ypos - j * delypos,
             f"$\\mathbf{{x}}$ = " + _xc + r"$^\top$",
             fontsize=14)
    j += 1
    _fgv = funclass.grad(xc[0, 0], xc[1, 0]).T[0]
    _fgv /= np.linalg.norm(_fgv)
    _fg = f"{np.array2string(_fgv, precision=3, floatmode='fixed')}"
    ax2.text(xpos, ypos - j * delypos,
             f"$\\nabla f\\left(\\mathbf{{x}}\\right)$ = " + _fg + r"$^\top$",
             fontsize=14)
    j += 1
    _hgv = ecfunclass.normal(xc[0, 0], xc[1, 0]).T[0]
    print(_hgv)
    _hgv /= np.linalg.norm(_hgv)
    _hg = f"{np.array2string(_hgv, precision=3, floatmode='fixed')}"
    ax2.text(xpos, ypos - j * delypos,
             f"$\\nabla h\\left(\\mathbf{{x}}\\right)$ = " + _hg + r"$^\top$",
             fontsize=14)


# Handling key press events
def on_press(event):
    global funclass, ecfunclass, t

    # Close figure if escaped.
    if event.key == 'escape':
        plt.close(fig)
        return

    # Chekc if the solution needs to be updated.
    if event.key == 'right':
        # Compute the next step        
        t += 0.1
        ax.cla()
    elif event.key == 'left':
        # Compute the next step        
        t -= 0.1
        ax.cla()
    elif event.key == 'shift+right':
        # Compute the next step        
        t += 0.001
        ax.cla()
    elif event.key == 'shift+left':
        # Compute the next step        
        t -= 0.001
        ax.cla()
    elif event.key in ['r', 'R']:
        # Reset the plot
        reset_params()
        ax.cla()
    
    # Wrap the value of t
    if t > 4:
        t = 4
    elif t < -4:
        t = -4
    
    # Update
    update()

    # Draw the plot and text
    ax.cla()
    update_text()
    plot_contour()
    plot_const_func()
    fig.canvas.draw()


if __name__ == "__main__":
    # Function and Method ID
    funclass = aladaopt.RectangularHyperbola()
    ecfunclass = aladaopt.Linear2Ec()
    
    # Create the figure and the axis.
    fig = plt.figure(figsize=(13, 7.8))
    gs = gridspec.GridSpec(1, 3, width_ratios=(1.2, 0.1, 1))
    ax = fig.add_subplot(gs[0, 0])
    ax.equal_aspect = True
    ax.axis('equal')
    # ax.axis('off')
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    axins = inset_axes(ax2, width="80%", height="30%", loc=4, borderpad=1)
    axins.axis('off')
    fig.canvas.manager.set_window_title('ALADA Optimization Animations: Constrained Optimization')

    # Initialize the solution
    xc = None
    norms, tangs = None, None
    t = 0.
    reset_params()

    # Plot stuff
    ax.cla()
    update_text()
    plot_contour()
    plot_const_func()
    fig.canvas.draw()

    # Create the figure and the axis.
    fig.canvas.mpl_connect('key_press_event', on_press)
    plt.tight_layout(pad=3)
    plt.show()