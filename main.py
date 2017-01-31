import numpy as np
import matplotlib.pyplot as plt
def run():
    """high level support for doing this and that."""

    plt.ylabel('y')
    plt.show(block=False)
    plt.ion()

    #Step 1 - collect data
    points = np.genfromtxt('data.csv', delimiter=',')

    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        plt.plot(x, y, 'bo')
    axes = list(plt.axis())

    #Step 2 - find hyper parameters
    #How fast model convearge
    learning_rate = 0.0001
    #y= ax + b
    initial_b = -5
    initial_m = 6
    num_iterations = 20

    #Step 3 - train model
    print('Starting gradient descent at b = {0}, m = {1}, error = {2}'.format(initial_b, initial_m, compute_error(initial_b, initial_m, points)))

    b, m = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations, axes)

    print('Ending points at b = {0}, m = {1}, error = {2}'.format(b, m, compute_error(b, m, points)))
    plt.show(block=True)

def compute_error(b, m, points):
    """ Compute errors for the points"""
    total_error = 0.0

    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total_error = (y - (m * x + b)) ** 2

        return total_error / float(len(points))

def gradient_descent_runner(points, starting_b, starting_m, l_rate, n_iterations, axes):
    """ run gradient descent algorithm"""

    print(axes)

    b = starting_b
    m = starting_m
    line = None
    ax = plt.gca()
    ax.autoscale(False)
    for i in range(0, n_iterations):
        b, m = step_gradient(b, m, points, l_rate)
        y1 = axes[0] * m + b
        y2 = axes[1] * m + b
        if line is not None:
            line.remove()
        line, = plt.plot([axes[0], axes[1]], [y1, y2], 'r-', linewidth=5.0)
        plt.pause(0.1)
    return b, m

def step_gradient(b_current, m_current, points, learningRate):
    """ run gradient descent step"""

    b_gradient = 0
    m_gradient = 0

    N = len(points)

    for i in range(0, N):
        x = points[i, 0]
        y = points[i, 1]

        b_gradient += -(2.0 / N) * (y - (m_current * x + b_current))
        m_gradient += (2.0 / N) * x * (y - (m_current * x + b_current))

    new_b = b_current + (learningRate * b_gradient)
    new_m = m_current + (learningRate * m_gradient)

    return new_b, new_m

if __name__ == '__main__':
    run()
