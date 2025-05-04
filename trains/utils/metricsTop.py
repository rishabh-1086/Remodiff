import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

__all__ = ['MetricsTop']

class MetricsTop():
    def __init__(self, train_mode):
        self.metrics_dict = {
            'FIRST_IMPRESSION': self.__eval_first_impression_regression,
        }

    def __multiclass_acc(self, y_pred, y_true):
        """
        Compute the multiclass accuracy w.r.t. groundtruth

        :param preds: Float array representing the predictions, dimension (N,)
        :param truths: Float/int array representing the groundtruth classes, dimension (N,)
        :return: Classification accuracy
        """
        return np.sum(np.round(y_pred) == np.round(y_true)) / float(len(y_true))

    def concordance_correlation_coefficient(self, y_true, y_pred):
        mean_true = np.mean(y_true)
        mean_pred = np.mean(y_pred)
        var_true = np.var(y_true)
        var_pred = np.var(y_pred)
        covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
        ccc = (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2)
        return ccc

    def __eval_first_impression_regression(self, y_pred, y_true):
        y_pred = y_pred.cpu().detach().numpy()#.reshape(-1, 1)
        y_true = y_true.cpu().detach().numpy().squeeze()#reshape(-1, 1)

        mse_loss = mean_squared_error(y_true, y_pred)
        l1_loss = mean_absolute_error(y_true, y_pred)
        ccc = self.concordance_correlation_coefficient(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        pcc, _ = pearsonr(y_true, y_pred)

        eval_results = {
            "MSE":  round(mse_loss, 4),
            "MAE": round(l1_loss, 4),
            "Accuracy": round(1-l1_loss, 4),
            "CCC": round(ccc, 4),
            "r2": round(r2, 4),
            "PCC": round(pcc, 4),
        }
        return eval_results
    
    def getMetics(self, datasetName):
        return self.metrics_dict[datasetName.upper()]