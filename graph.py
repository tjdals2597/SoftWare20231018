import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_descriptive_stats(ax, data):
    a = data[:, 0]
    b = data[:, 1]
    ax.bar(['Mean', 'Median'], [np.mean(a), np.median(a)], color='blue', alpha=0.7, label='Variable 1')
    ax.bar(['Mean', 'Median'], [np.mean(b), np.median(b)], color='green', alpha=0.7, label='Variable 2')
    ax.legend()
    ax.set_title('Descriptive Statistics: Mean and Median')

def plot_correlation(ax, data):
    sns.heatmap(np.corrcoef(data.T), annot=True, ax=ax)
    ax.set_title('Correlation Analysis')

def plot_histogram(ax, data):
    a = data[:, 0]
    b = data[:, 1]
    ax.hist(a, bins=15, color='blue', alpha=0.7, label='Variable 1')
    ax.hist(b, bins=15, color='green', alpha=0.7, label='Variable 2')
    ax.legend()
    ax.set_title('Histogram of Variables')

def plot_scatter(ax, data):
    a = data[:, 0]
    b = data[:, 1]
    ax.scatter(a, b, alpha=0.7)
    ax.set_xlabel('Variable 1')
    ax.set_ylabel('Variable 2')
    ax.set_title('Scatter Plot of Variable 1 vs Variable 2')

np.random.seed(0); data = np.random.randn(100, 2)
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

plt.tight_layout(); plt.show()
