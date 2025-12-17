import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

def plot_regression_results(y_true, y_pred, title, save_path=None):
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
    r2 = r2_score(y_true, y_pred)
    
    print(f'{title} \n\t- MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}')

    if save_path:
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
        plt.xlabel('True T')
        plt.ylabel('Predicted T')
        plt.title(f'{title} \n- MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}')
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
    
