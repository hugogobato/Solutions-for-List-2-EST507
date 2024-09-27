import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import scipy.stats as stats
from scipy.stats import shapiro, anderson, jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
from statsmodels.stats.diagnostic import het_breuschpagan, het_white

def model_diagnostics(model, X, y):
    cov_beta = model.cov_params()
    std_errors = np.array(model.bse)
    correlation_matrix = cov_beta / (std_errors[:, None] @ std_errors[None, :])
    
    print("Correlation matrix of estimated betas:")
    print(correlation_matrix)

    plt.scatter(model.fittedvalues, model.resid)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Fitted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted Plot')
    plt.show()
    
    plt.scatter(model.fittedvalues, np.sqrt(np.abs(model.resid)))
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Fitted Values')
    plt.ylabel('√|Standardized Residuals|')
    plt.title('Scale-Location Plot')
    plt.show()
    
    sm.qqplot(model.resid, line='45')
    plt.title('Q-Q Plot')
    plt.show()
    
    shapiro_test = shapiro(model.resid)
    print(f'Shapiro-Wilk test statistic: {shapiro_test.statistic:.4f}, p-value: {shapiro_test.pvalue:.4f}')
    
    if shapiro_test.pvalue < 0.05:
        print("Reject the null hypothesis: Residuals are not normally distributed.")
    else:
        print("Fail to reject the null hypothesis: Residuals appear to be normally distributed.")
    
    anderson_test = anderson(model.resid, dist='norm')
    print(f'Anderson-Darling test statistic: {anderson_test.statistic:.4f}')
    print("Critical values and significance levels:")
    
    for i in range(len(anderson_test.critical_values)):
        sl, cv = anderson_test.significance_level[i], anderson_test.critical_values[i]
        print(f'Significance level: {sl}%, Critical value: {cv}')
        if anderson_test.statistic > cv:
            print(f"Reject the null hypothesis at the {sl}% significance level: Residuals are not normally distributed.")
        else:
            print(f"Fail to reject the null hypothesis at the {sl}% significance level.")
    
    jb_test = jarque_bera(model.resid)
    print(f'Jarque-Bera test statistic: {jb_test.statistic:.4f}, p-value: {jb_test.pvalue:.4f}')
    
    if jb_test.pvalue < 0.05:
        print("Reject the null hypothesis: Residuals are not normally distributed.")
    else:
        print("Fail to reject the null hypothesis: Residuals appear to be normally distributed.")
    
    sm.graphics.influence_plot(model, criterion="cooks")
    plt.title('Residuals vs Leverage Plot')
    plt.show()   
    
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    print(vif_data)
    
    sm.graphics.plot_partregress_grid(model)
    plt.tight_layout()
    plt.show()
    
    bp_test = het_breuschpagan(model.resid, model.model.exog)
    print(f'Breusch-Pagan test p-value: {bp_test[1]:.4f}')
    if bp_test[1] < 0.05:
        print("Result: Reject the null hypothesis. There is evidence of heteroscedasticity in the residuals.")
    else:
        print("Result: Fail to reject the null hypothesis. There is no evidence of heteroscedasticity in the residuals.")
    
    white_test = het_white(model.resid, model.model.exog)
    print(f'White test p-value: {white_test[1]:.4f}')
    if white_test[1] < 0.05:
        print("Result: Reject the null hypothesis. There is evidence of heteroscedasticity in the residuals.")
    else:
        print("Result: Fail to reject the null hypothesis. There is no evidence of heteroscedasticity in the residuals.")