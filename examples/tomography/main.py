# -*- coding: utf-8 -*-
import argparse
import json
import numpy as np
from qibolab import tomography


def rho_theory(i):
    rho = np.zeros((4, 4), dtype=complex)
    rho[i, i] = 1
    return rho


def extract(filename):
    with open(filename,"r") as r:
        raw = json.loads(r.read())
    return raw


state_file = "./data/states_181120.json"

measurement_files = [("./data/tomo_181120-00.json", rho_theory(0)),
                     ("./data/tomo_181120-01.json", rho_theory(1)),
                     ("./data/tomo_181120-10.json", rho_theory(2)),
                     ("./data/tomo_181120-11.json", rho_theory(3))]


parser = argparse.ArgumentParser()
parser.add_argument("--index", default=0, type=int)
parser.add_argument("--plot", action="store_true")


def plotfunc(tomography, rho_theory, width=0.8, depth=0.8):
    """Plots histograms of theory and estimated density matrices.

    Args:
        tomography (:class:`qibo.numpy.tomography.Tomography`): Tomography
            Qibo object that holds the estimated density matrices.
        rho_theory (np.ndarray): Theoretical (target) density matrix.
        width (float): Width of the histograms.
        depth (float): Depth of the histograms.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    _x, _y = np.meshgrid(np.arange(4), np.arange(4))
    x, y = _x.ravel(), _y.ravel()

    top_real_th = rho_theory.real.ravel()
    top_imag_th = rho_theory.imag.ravel()

    top_real_exp = tomography.linear.real.ravel()
    top_imag_exp = tomography.linear.imag.ravel()

    top_real_fit = tomography.fit.real.ravel()
    top_imag_fit = tomography.fit.imag.ravel()
    fidelity = tomography.fidelity(rho_theory)

    plt.style.use('default')
    ticks = [0.5, 1.5, 2.5, 3.5]
    tick_labels = ['|00>','|01>', '|10>', '|11>']
    fig = plt.figure(figsize=(12, 16))
    fig.suptitle("Fidelity: {:f}".format(fidelity))
    ax1 = fig.add_subplot(231, projection='3d')
    ax2 = fig.add_subplot(234, projection='3d')
    ax3 = fig.add_subplot(232, projection='3d')
    ax4 = fig.add_subplot(235, projection='3d')
    ax5 = fig.add_subplot(233, projection='3d')
    ax6 = fig.add_subplot(236, projection='3d')

    bottom = np.zeros_like(top_real_th)
    ax1.bar3d(x, y, bottom, width, depth, top_real_th, shade=True, color='C0')
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(tick_labels)
    ax1.set_yticks(ticks)
    ax1.set_yticklabels(tick_labels)
    ax1.set_zlim3d(-1, 1)
    ax1.set_title('Real part, Theory')

    ax2.bar3d(x, y, bottom, width, depth, top_imag_th, shade=True, color='C0')
    ax2.set_xticks(ticks)
    ax2.set_xticklabels(tick_labels)
    ax2.set_yticks(ticks)
    ax2.set_yticklabels(tick_labels)
    ax2.set_zlim3d(-1, 1)
    ax2.set_title('Imaginary part, Theory')

    ax3.bar3d(x, y, bottom, width, depth, top_real_exp, shade=True, color='C1')
    ax3.set_xticks(ticks)
    ax3.set_xticklabels(tick_labels)
    ax3.set_yticks(ticks)
    ax3.set_yticklabels(tick_labels)
    ax3.set_zlim3d(-1, 1)
    ax3.set_title('Real part, Linear')

    ax4.bar3d(x, y, bottom, width, depth, top_imag_exp, shade=True, color='C1')
    ax4.set_xticks(ticks)
    ax4.set_xticklabels(tick_labels)
    ax4.set_yticks(ticks)
    ax4.set_yticklabels(tick_labels)
    ax4.set_zlim3d(-1, 1)
    ax4.set_title('Imaginary part, Linear')

    ax5.bar3d(x, y, bottom, width, depth, top_real_fit, shade=True, color='C2')
    ax5.set_xticks(ticks)
    ax5.set_xticklabels(tick_labels)
    ax5.set_yticks(ticks)
    ax5.set_yticklabels(tick_labels)
    ax5.set_zlim3d(-1, 1)
    ax5.set_title('Real part, MLE_{}'.format(tomography.success))

    ax6.bar3d(x, y, bottom, width, depth, top_imag_fit, shade=True, color='C2')
    ax6.set_xticks(ticks)
    ax6.set_xticklabels(tick_labels)
    ax6.set_yticks(ticks)
    ax6.set_yticklabels(tick_labels)
    ax6.set_zlim3d(-1, 1)
    ax6.set_title('Imaginary part, MLE_{}'.format(tomography.success))

    plt.show()


def main(index, plot):
    """Perform tomography to estimate density matrix from measurements.

    Args:
        index (int): Which experimental json file to use.
            See ``measurement_files`` list defined above for available files.
        plot (bool): Plot histograms comparing the estimated density matrices
            with the theoretical ones. If ``False`` only the fidelity is
            calculated.
    """
    # Extract state data and define ``gate``
    state = extract(state_file)
    state = np.stack(list(state.values()))
    state = np.sqrt((state ** 2).sum(axis=1))

    # Extract tomography amplitudes
    filename, rho_theory = measurement_files[index]
    amp = extract(filename)
    amp = np.stack(list(amp.values()))
    amp = np.sqrt((amp ** 2).sum(axis=1))

    # Create tomography object
    tom = tomography.Tomography(amp, state)
    # Optimize denisty matrix by minimizing MLE
    tom.minimize()

    fidelity = tom.fidelity(rho_theory)
    print("Convergence: {}".format(tom.success))
    print("Fidelity: {}".format(fidelity))

    if plot:
        plotfunc(tom, rho_theory)


if __name__ == "__main__":
    args = vars(parser.parse_args())
    main(**args)
