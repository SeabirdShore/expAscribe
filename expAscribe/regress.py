import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from bayes_opt import BayesianOptimization


def piecewise_linear_regression_3d(data, candidate_knots_x, candidate_knots_y):
    best_knot_x = None
    best_knot_y = None
    best_rss = np.inf 
    best_model = None

    for knot_x in candidate_knots_x:
        for knot_y in candidate_knots_y:
            data['D_x'] = (data['X'] > knot_x).astype(int)
            data['D_y'] = (data['Y'] > knot_y).astype(int)
            data['kink_x'] = (data['X'] - knot_x) * data['D_x']
            data['kink_y'] = (data['Y'] - knot_y) * data['D_y']
            
            X_vars = sm.add_constant(data[['X', 'Y', 'D_x', 'D_y', 'kink_x', 'kink_y']])
            
            model = sm.OLS(data['Z'], X_vars).fit()
            
            rss = np.sum(model.resid ** 2)
            
            if rss < best_rss:
                best_rss = rss
                best_knot_x = knot_x
                best_knot_y = knot_y
                best_model = model
                
    return best_knot_x, best_knot_y, best_model



def piecewise_linear_regression(data, candidate_knots):
    best_knot = None
    best_rss = np.inf
    best_model = None

    for knot in candidate_knots:

        data['D'] = (data['X'] > knot).astype(int)
        data['kink'] = (data['X'] - knot) * data['D']
        
        X_vars = sm.add_constant(data[['X', 'D', 'kink']])
        
        model = sm.OLS(data['Y'], X_vars).fit()
        
        rss = np.sum(model.resid ** 2)
        
        if rss < best_rss:
            best_rss = rss
            best_knot = knot
            best_model = model
            
    return best_knot, best_model



def kink_regression_2d(file_path,xl,xh,iplot=True):
    data = pd.read_csv(file_path)
    candidate_knots_x = np.linspace(xl, xh, 20)
    best_knot_x, best_model = piecewise_linear_regression(data, candidate_knots_x)
    if iplot:
        x_vals = np.linspace(xl, xh, 100)
        df_predict = pd.DataFrame({
            'X': x_vals,
            'D': (x_vals > best_knot_x).astype(int),
            'kink': (x_vals - best_knot_x) * (x_vals > best_knot_x)
        })
        predictions = best_model.get_prediction(sm.add_constant(df_predict))
        pred_summary = predictions.summary_frame(alpha=0.05)  # 95% CI
        y_vals = pred_summary['mean']
        lower_ci = pred_summary['mean_ci_lower']
        upper_ci = pred_summary['mean_ci_upper']
        plt.style.use('_mpl-gallery-nogrid')
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hexbin(data['X'], data['Y'], gridsize=10,label='Samples')
        plt.plot(x_vals, y_vals, color='#2B6688', label='Piecewise Linear Fit')
        plt.fill_between(x_vals, lower_ci, upper_ci, color='grey', alpha=0.2, label='95% CI')
        plt.axvline(best_knot_x, color='#193E8F', linestyle='--', label='Estimated Knot')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Piecewise Linear Fit with Estimated Knot')
        plt.legend()
        plt.show()
    return best_knot_x, best_model




def kink_regression_3d(file_path,xl,xh,yl,yh,iplot=True,responsive_surface_design=True):
    data = pd.read_csv(file_path)
    candidate_knots_x = np.linspace(xl, xh, 20)  
    candidate_knots_y = np.linspace(yl, yh, 20)
    best_knot_x, best_knot_y, best_model = piecewise_linear_regression_3d(data, candidate_knots_x, candidate_knots_y)
    x_vals = np.linspace(1.5*xl, 1.5*xh, 100)
    y_vals = np.linspace(1.5*yl, 1.5*yh, 100)
    X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
    data_grid = pd.DataFrame({
        'X': X_grid.ravel(),
        'Y': Y_grid.ravel(),
        'D_x': (X_grid.ravel() > best_knot_x).astype(int),
        'D_y': (Y_grid.ravel() > best_knot_y).astype(int),
        'kink_x': (X_grid.ravel() - best_knot_x) * (X_grid.ravel() > best_knot_x),
        'kink_y': (Y_grid.ravel() - best_knot_y) * (Y_grid.ravel() > best_knot_y)
    })
    min_idx = data['Z'].idxmin()
    min_point = data.loc[min_idx]
    min_x, min_y, min_z = min_point['X'], min_point['Y'], min_point['Z']
    Z_pred = best_model.predict(sm.add_constant(data_grid))
    Z_grid = Z_pred.values.reshape(X_grid.shape)
    intercept = best_model.params['const']
    coef_X = best_model.params['X']
    coef_Y = best_model.params['Y']
    coef_D_x = best_model.params['D_x']
    coef_D_y = best_model.params['D_y']
    coef_kink_x = best_model.params['kink_x']
    coef_kink_y = best_model.params['kink_y']

    D_x = int(min_x > best_knot_x)
    D_y = int(min_y > best_knot_y)
    kink_x = (min_x - best_knot_x) * D_x
    kink_y = (min_y - best_knot_y) * D_y

    min_point_pred = (intercept +
                    coef_X * min_x +
                    coef_Y * min_y +
                    coef_D_x * D_x +
                    coef_D_y * D_y +
                    coef_kink_x * kink_x +
                    coef_kink_y * kink_y)
    if iplot:
        fig = plt.figure(figsize=(14, 8))

        ax = fig.add_subplot(121, projection='3d')
        ax.scatter(data['X'], data['Y'], data['Z'], alpha=0.5, label='Data', color='blue')

        ax.plot_surface(X_grid, Y_grid, Z_grid, color='gray', alpha=0.5, label='Piecewise Linear Fit')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Regression Kink Design')
        ax.legend()
        plt.show()

        if responsive_surface_design:
            fig2 = plt.figure(figsize=(14, 6))
            ax2 = fig2.add_subplot(122)
            ax2.scatter(data['X'], data['Y'], alpha=0.5, label='Total Sample', color='#427AB2')

            contour_plot = ax2.contour(X_grid, Y_grid, Z_grid, levels=[max(min_point_pred * 1.1, min_z)], colors='black', linewidths=2)

            contour_path = contour_plot.get_paths()[0]
            inside_mask = contour_path.contains_points(data[['X', 'Y']].values)

            ax2.scatter(data.loc[inside_mask, 'X'], data.loc[inside_mask, 'Y'], color='#F8D5E4', label='Likely Optm Points', edgecolor='black')
            ax2.scatter(min_x, min_y, color='#CC88B0', s=100, label='Min Z Point', edgecolor='black')
            ax2.scatter(best_knot_x, best_knot_y, color='#96D1C6', s=100, label='Kink Point', edgecolor='black')

            ax2.plot([], [], color='black', linewidth=2, label='O(minz) in Kink Eq')
            ax2.legend()
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_title('Response Surface Projection')
            ax2.legend()
            plt.show()
            return best_knot_x, best_knot_y, best_model, pd.concat([data.loc[inside_mask, 'X'], data.loc[inside_mask, 'Y'],data.loc[inside_mask, 'Z']],axis=1)
    return best_knot_x, best_knot_y, best_model

def bayes_opt(df, ploy_dim=2, init_p=5, iters=25, verbose_=2, mode='maximize'):
    X_data = df[['X', 'Y']]
    Z_data = df['Z']

    poly = PolynomialFeatures(degree=ploy_dim)  
    X_poly = poly.fit_transform(X_data.values)

    model = LinearRegression()
    model.fit(X_poly, Z_data)

    def regression_function(x, y):
        poly_features = poly.transform([[x, y]])
        
        z = model.predict(poly_features)[0]
        return z

    pbounds = {
        'x': (df['X'].min(), df['X'].max()),
        'y': (df['Y'].min(), df['Y'].max())
    }

    def bayesian_optimize():
        def black_box_function(x, y):
            if mode == 'maximize':
                return regression_function(x, y)  
            elif mode == 'minimize':
                return -regression_function(x, y)

        optimizer = BayesianOptimization(
            f=black_box_function,
            pbounds=pbounds,
            random_state=42,
            verbose=verbose_,
            allow_duplicate_points=True
        )
        optimizer.maximize(init_points=init_p, n_iter=iters)

        return optimizer.max
        
    optimal_result = bayesian_optimize()
    return optimal_result