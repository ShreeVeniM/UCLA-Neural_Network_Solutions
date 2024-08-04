import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import mean_squared_error, r2_score

def save_plot(fig, filename, folder='charts'):
    try:
        os.makedirs(folder, exist_ok=True)
        fig.savefig(os.path.join(folder, filename), dpi=300)
        plt.close(fig)
    except Exception as e:
        print(f"Error saving plot: {e}")

def model_evaluate(model, Xtest, ytest, folder='charts'):
    try:
        ypred = model.predict(Xtest)
        mse = mean_squared_error(ytest, ypred)
        r2 = r2_score(ytest, ypred)

        # Plotting
        fig, ax = plt.subplots()
        sns.regplot(x=ytest, y=ypred, ax=ax)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title('Actual vs Predicted')
        save_plot(fig, 'actual_vs_predicted.png', folder)

        return mse, r2
    except Exception as e:
        print(f"Error in model evaluation: {e}")
        return None, None
