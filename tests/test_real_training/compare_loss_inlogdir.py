import tensorflow as tf
import matplotlib.pyplot as plt
import os
import glob
import seaborn as sns
import numpy as np
from matplotlib.ticker import FuncFormatter

def read_logdir(logpath):
    # Finds the event file in the specified directory.
    is_diffusion = False
    if os.path.exists(os.path.join(logpath, 'loss_validation')):
        event_files = glob.glob(os.path.join(logpath, 'loss_validation', 'events.out.tfevents.*'))
    elif os.path.exists(os.path.join(logpath, 'loss_valid_loss')):
        event_files = glob.glob(os.path.join(logpath, 'loss_valid_loss', 'events.out.tfevents.*'))
        is_diffusion = True
    else:
        raise ValueError(f"Could not find event files in {logpath}")
    
    if not event_files:
        print("No event files found.")
        return
    
    event_file_path = event_files  # Select the first event file.
    result = {'steps': [], 'loss': []}

    for event_file_path in event_files:
        try:
            # Iterates over the events in the file.
            for e in tf.compat.v1.train.summary_iterator(event_file_path):
                for val in e.summary.value:
                    # Checks if the value has the property 'simple_value'.
                    if val.HasField('simple_value'):
                        # print(e.step, val.tag, val.simple_value)
                        result['steps'].append(e.step)
                        result['loss'].append(val.simple_value) if  not is_diffusion else result['loss'].append(np.sqrt(val.simple_value))
                    else:
                        print(f"Value for step {e.step} has no 'simple_value'")
        except Exception as e:
            print(f"Failed to read event file {event_file_path}: {str(e)}")

    # Sorts the results by step.
    result['steps'], result['loss'] = zip(*sorted(zip(result['steps'], result['loss'])))

    return result

def compare_loss():
    logpaths = {'LSTM+Aug.': 'logdir/traj_train/LSTM/pos/real/OptimLayer/run09',
                'Diffusion+Aug.': 'logdir/traj_train/Diffusion',
                'PhyTune+Aug.': 'logdir/traj_train/PhyTune/pos/real/OptimLayer/run19',
                'MLP+Aug.': 'logdir/traj_train/PureMLP/pos/real/OptimLayer/run03',
                'MLP+GS (ours)':'logdir/traj_train/MLP/pos/real/OptimLayer/run02',
                'Skip+GS. (ours)': 'logdir/traj_train/Skip/pos/real/OptimLayer/run00',
                'MNN+GS (ours)': 'logdir/traj_train/MNN/pos/real/OptimLayer/run44',
                } 

    # Setting up a seaborn style
    sns.set(style="whitegrid")

    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plotting each model's loss curve
    for name, logpath in logpaths.items():
        result = read_logdir(logpath)
        if len(result['steps']) > 40:
            # diffusion
            div = 5
            N = len(result['steps'])
            ax.plot(result['steps'][0:N:N//div], result['loss'][0:N:N//div], label=name, marker='o', markersize=5, linestyle='-')
            print(result['loss'][0:N:N//div])
        else:
            ax.plot(result['steps'], result['loss'], label=name, marker='o', markersize=5, linestyle='-')

    # Log scale for y-axis

    # Set limits and labels with increased font sizes
    ax.set_xlim(0, 1500)
    ax.set_ylabel('Validation L1 Loss (m)', fontsize=14)
    ax.set_xlabel('Steps', fontsize=14)

    # Improve legend placement
    ax.legend(fontsize=12)

    # ax.set_yscale('log')
    # # add more y ticks and grids, each grid has a text label
    # ax.yaxis.grid(True, which='minor', linestyle='--', linewidth=0.5)
    # ax.yaxis.set_tick_params(which='minor', labelsize=12)
    # ax.yaxis.set_tick_params(which='major', labelsize=12)
    # ax.minorticks_on()

    # tick_locations = np.logspace(-2, 0, num=8)  # Example: 10^-3, 10^-2, 10^-1, 10^0

    # # Set custom ticks
    # ax.set_yticks(tick_locations)
    # # ax.set_yticks(np.logspace(-3, 0, num=12), minor=True)  # More granular minor ticks

    # # Formatting function for ticks to appear as y.z 10^x
    # def format_func(value, tick_number):
    #     # Splitting the exponent and mantissa
    #     exponent = np.floor(np.log10(value))
    #     mantissa = value / 10**exponent
    #     return f'{mantissa:.1f}x$10^{{{int(exponent)}}}$'
    
    # ax.yaxis.set_major_formatter(FuncFormatter(format_func))

    # Show the plot
    plt.show()

if __name__ == '__main__':
    compare_loss()