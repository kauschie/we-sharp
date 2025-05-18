import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg') # Use TkAgg backend for interactive plots
import pandas as pd
import numpy as np

def parse_log(log_file):
    """
    Parse the log file and return a DataFrame.
    lines that contain batch size appear in the format:
    2025-02-22 09:17:05,237 - batch_size: 8

    lines that contain gradient accumulation appear in the format:
    2025-02-22 09:17:05,238 - grad_accum_every: 8
    lines with loss appear in the format:
    2025-02-22 09:17:25,480 - Step 12: Training Loss: 51.43777322769165



    """
    with open(log_file, 'r') as f:
        lines = f.readlines()
        data = []
        batch_size = 1
        grad_accum = 1
        increment_amount = batch_size * grad_accum
        samples_processed = 0
        for line in lines:
            if 'batch_size:' in line:
                batch_size = int(line.split('batch_size:')[1].strip())
                increment_amount = batch_size * grad_accum
                print(f'Batch size: {batch_size}')
            elif 'grad_accum_every:' in line:
                grad_accum = int(line.split('grad_accum_every:')[1].strip())
                increment_amount = batch_size * grad_accum
                print(f'Gradient accumulation: {grad_accum}')
            elif 'Step' in line:
                step = int(line.split('Step')[1].split(':')[0].strip())
                loss = float(line.split('Training Loss:')[1].strip())
                data.append([step, loss])
                samples_processed += increment_amount


    df = pd.DataFrame(data, columns=['step', 'loss'])
    print(f'Trained on {samples_processed} samples')

    return df

def smooth_loss(df, window=5):
    """
    Smooth the loss using a rolling window.
    """
    smoothed_loss = []
    for i in range(len(df['loss'])):
        if i < window:
            smoothed_loss.append(np.mean(df['loss'][:i+1]))
        else:
            smoothed_loss.append(np.mean(df['loss'][i-window:i+1]))
    df[f'smoothed_loss_{window}'] = smoothed_loss

    return df


def plot_loss(df1, df2, df3, label='Unknown'):
    """
    Plot the loss curves for Coarse, Semantic, and Fine training.
    """
    plt.figure(figsize=(8, 5))

    # Coarse
    plt.scatter(df1['step'], df1['loss'], alpha=0.5, s=3, color='#FFB6B6', label='Training Loss (Coarse)')
    plt.plot(df1['step'], df1['smoothed_loss_500'], label='Smoothed Loss (Coarse)', color='#E60000')

    # Semantic
    plt.scatter(df2['step'], df2['loss'], alpha=0.5, s=3, color='#ADD8E6', label='Training Loss (Semantic)')
    plt.plot(df2['step'], df2['smoothed_loss_500'], label='Smoothed Loss (Semantic)', color='#0077B6')

    # Fine
    plt.scatter(df3['step'], df3['loss'], alpha=0.5, s=3, color='#B8E994', label='Training Loss (Fine)')
    plt.plot(df3['step'], df3['smoothed_loss_500'], label='Smoothed Loss (Fine)', color='#2E8B57')

    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title(f'{label} Training Loss vs Step')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{label}_training_loss.svg', dpi=300)
    plt.savefig(f"{label}_training_loss.png")
    plt.show()


if __name__ == '__main__':
    df_course = parse_log('../p1_logs/coarse/coarse_training.log')
    df_sem = parse_log("../p1_logs/sem/semantic_training.log")
    df_fine = parse_log("../p1_logs/fine/fine_training.log")
    df_course = smooth_loss(df_course, window=500)
    df_sem = smooth_loss(df_sem, window=500)
    df_fine = smooth_loss(df_fine, window=500)

    # print(df.head())
    plot_loss(df_sem, df_course, df_fine, label='All')