import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from Optimizers import *
import matplotlib.colors as mcolors

def beale(x, y):
    return (1.5 - x + (x * y))**2 + (2.25 - x + (x * y**2))**2 + (2.625 - x + (x * y**3))**2

def abs_sum(x, y):
    return abs(x) + abs(y)

def inseparable_L1(x, y):
    return abs(x + y) + (abs(x - y) / 10)

def inseparable_L2(x, y):
    return (x + y)**2 + ((x - y)**2 / 10)

def abs_sum_div(x, y):
    return abs(x) / 10 + abs(y)

def rosenbrock(x, y, a=1, b=100):
    return (a - x)**2 + b * (y - x**2)**2

def optimize_function(initial_position, learning_rate, max_lr, num_steps, optimizer_type, function, increment, max_lr_scale_clamp, decrement, min_lr_scale_clamp, decr_start_step):
    x = torch.tensor(initial_position[0], requires_grad=True)
    y = torch.tensor(initial_position[1], requires_grad=True)

    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam([x, y], lr=learning_rate)
    elif optimizer_type == 'Adam-Upper-LR':
        optimizer = torch.optim.Adam([x, y], lr=max_lr)
    elif optimizer_type == 'AccAsroFinalScale':
        optimizer = AccAsroFinalScale([x, y], lr=learning_rate, increment=increment, max_lr_scale_clamp=max_lr_scale_clamp, decrement=decrement, min_lr_scale_clamp=min_lr_scale_clamp, decr_start_step=decr_start_step, num_iters=num_steps)
    else:
        raise ValueError('Unsupported optimizer type')

    positions = []
    for i in range(num_steps):
        optimizer.zero_grad()
        loss = function(x, y)
        loss.backward()
        optimizer.step()
        positions.append((x.item(), y.item()))
        
    return positions

# Define optimal points and functions
optimal_points = {
    "abs_sum": {"optimum": (0.0, 0.0), "start": (3.0, 7.0)},
    "inseparable_L1": {"optimum": (0.0, 0.0), "start": (-2.0, -4.0)},
    "inseparable_L2": {"optimum": (0.0, 0.0), "start": (5.0, 2.0)},
    "abs_sum_div": {"optimum": (0.0, 0.0), "start": (-4.0, -7.0)},
    "beale": {"optimum": (3.0, 0.5), "start": (-9.0, -9.0)},
    "rosenbrock": {"optimum": (1.0, 1.0), "start": (0.0, -6.0)}
}

functions = {
    "abs_sum": abs_sum,
    "inseparable_L1": inseparable_L1,
    "inseparable_L2": inseparable_L2,
    "abs_sum_div": abs_sum_div,
    "beale": beale,
    "rosenbrock": rosenbrock
}

chart_titles = {
    "abs_sum": '(a) Absolute Sum Function',
    "inseparable_L1": '(b) Inseparable L1 Loss' ,
    "inseparable_L2": '(c) Inseparable L2 Loss',
    "abs_sum_div": '(d) X-Scaled Absolute Sum' ,
    "beale": '(e) Beale Function',
    "rosenbrock": '(f) RosenBrock Function'
}

# Parameters
learning_rate = 0.01
max_lr = learning_rate * 20
min_lr = learning_rate / 10
increment = 2e-1
max_lr_scale_clamp = max_lr / learning_rate
decrement = 0
min_lr_scale_clamp = min_lr / learning_rate
decr_start_step = 1
num_steps_dict = {
    'rosenbrock': 5000,
    'beale': 20000,
    'abs_sum': 150,
    'default': 400
}

optimizers = ['Adam-Upper-LR', 'AccAsroFinalScale', 'Adam']
optimizer_colors = {
    'Adam': 'blue',
    'AccAsroFinalScale': 'red',
    'Adam-Upper-LR': 'yellow'
}
optimizers_labels = {
    'Adam': 'Adam - LR: ' + str(learning_rate),
    'AccAsroFinalScale': 'AccAsro LR: ' + str(learning_rate) + ' - ' + str(max_lr),
    'Adam-Upper-LR': 'Adam - LR: ' + str(max_lr)
}

fig, axes = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)
axes = axes.flatten()

for i, (func_name, func) in enumerate(functions.items()):
    num_steps = num_steps_dict.get(func_name, num_steps_dict['default'])
    initial_position = optimal_points[func_name]["start"]
    optimum_position = optimal_points[func_name]["optimum"]

    if func_name == 'rosenbrock':
        x = np.linspace(-2.5, 2.5, 400)
        y = np.linspace(-10, 10, 400)
    elif func_name == 'inseparable_L1':
        x = np.linspace(-7.5, 7.5, 400)
        y = np.linspace(-7.5, 7.5, 400)
    else:
        x = np.linspace(-10, 10, 400)
        y = np.linspace(-10, 10, 400)

    X, Y = np.meshgrid(x, y)
    Z = func(torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)).numpy()

    norm = mcolors.LogNorm(vmin=Z.min() + 1e-10, vmax=Z.max())
    log_levels = np.logspace(np.log10(Z.min() + 1e-10), np.log10(Z.max()), num=100)
    log_levels_contour_lins = np.logspace(np.log10(Z.min() + 1e-10), np.log10(Z.max()), num=20)

    ax = axes[i]
    contourf = ax.contourf(X, Y, Z, levels=log_levels, cmap='PuBuGn', norm=norm)
    contour = ax.contour(X, Y, Z, levels=log_levels_contour_lins, colors='black', linewidths=0.7, norm=norm)
    
    all_positions = {}
    for optimizer_type in optimizers:
        positions = optimize_function(
            initial_position,
            learning_rate,
            max_lr,
            num_steps,
            optimizer_type,
            func,
            increment,
            max_lr_scale_clamp,
            decrement,
            min_lr_scale_clamp,
            decr_start_step
        )
        positions = np.array(positions)
        all_positions[optimizer_type] = positions
        color = optimizer_colors.get(optimizer_type, 'black')
        ax.plot(positions[:, 0], positions[:, 1], label=optimizers_labels[optimizer_type], color=color, linewidth=2)
        ax.plot(positions[-1, 0], positions[-1, 1], 'o', color=color)

    if func_name == 'rosenbrock':
        ax_inset = inset_axes(ax, width="30%", height="30%", loc='lower left')
    else:
        ax_inset = inset_axes(ax, width="30%", height="30%", loc='upper left')

    ax_inset.set_facecolor('teal')

    # Zoom in on the vicinity of the optimum position in the inset
    zoom_radius = 0.5 
    ax_inset.set_xlim(optimum_position[0] - zoom_radius, optimum_position[0] + zoom_radius)
    ax_inset.set_ylim(optimum_position[1] - zoom_radius, optimum_position[1] + zoom_radius)

    for optimizer_type, positions in all_positions.items():
        color = optimizer_colors.get(optimizer_type, 'black')  # Default to black if color not specified
        ax_inset.plot(positions[:, 0], positions[:, 1], label=optimizers_labels[optimizer_type], color=color)
        ax_inset.plot(positions[-1, 0], positions[-1, 1], 'o', color=color)  # Mark the end point with optimizer color

    ax_inset.plot(initial_position[0], initial_position[1], 'o', color='#FFA500')
    ax_inset.plot(optimum_position[0], optimum_position[1], 'X', color='purple', markersize=9)

    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    ax_inset.set_xlabel('')
    ax_inset.set_ylabel('')
    ax_inset.legend().set_visible(False)

    # Plot the start and optimum positions
    ax.plot(initial_position[0], initial_position[1], 'o', color='#FFA500', label='Start')
    ax.plot(optimum_position[0], optimum_position[1], 'X', color='purple', markersize=9, label='Optimum')
    
    ax.set_title(chart_titles[func_name], fontdict={'fontsize': 16, 'fontname': 'Times New Roman'})

    if func_name == 'abs_sum':
        ax.legend(fontsize=13, loc = 'lower left') 
    else:
        ax.legend(fontsize=13)

    ax.tick_params(axis='both', which='major', labelsize=5)

# plt.savefig("./Function Optimization Tests.png", dpi=600)
plt.savefig("./Function Optimization Tests.pdf", dpi=600)

# plt.show()