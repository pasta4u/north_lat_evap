#based on https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import scipy.stats
import os, glob, json, joblib
from compile_level1 import compile_for_gapfilling

#Add ERA5 Land evaporation as baseline for testing models
with open(os.path.join('..','metadata','wind_dir_sectors.json')) as file:
    wind_dir_sectors = json.load(file)

with open(os.path.join('..','metadata','level2_predictor_lists.json')) as file:
    predictor_lists = json.load(file)

def train_and_test(fig_path,
                    model_path,
                    model_id,
                    data_in,
                    response_variable,
                    baseline_variable,
                    baseline_coeff=1,
                    regression_model='RF'):
    '''
    data_in: dataframe including response variable and predictors used to gapfill
    response_variable: str, name of response_variable matching column name in data_in
    regression_model: either RF (random forest), or LR (Linear Regression)
    '''

    info = []

    #plot input data
    data_in.plot(subplots=True,figsize=(16,10))
    plt.savefig(os.path.join(fig_path,f'{response_variable}_indata.png'))

    data_in.dropna(how='any',axis=0,inplace=True)
    #plot pairplot
    pplot = sns.pairplot(data=data_in)
    pplot.savefig(os.path.join(fig_path,f'{response_variable}_indata_pairplot.png'))

    #plot correlations
    corr = data_in.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig,ax = plt.subplots(figsize=(16,10))
    sns.heatmap(corr, mask=mask, annot=True, cmap=plt.cm.Reds, ax=ax,vmin=-1,vmax=1)
    fig.savefig(os.path.join(fig_path,f'{response_variable}_correlation.png'))
    plt.close('all')

    #predictors for training and testing
    features = data_in.copy()

    # in dataframe with predictors:
    # removing rows with na-values in any of the columns
    # before dropping response_variable
    features.dropna(inplace=True)

    #response variable, target
    labels = features[response_variable].to_numpy()

    #dropping label (response variable) from features
    features.drop(response_variable, axis = 1, inplace = True)

    #saving names for later use
    feature_list = list(features.columns)

    n = len(features.columns)
    info.append(f'Building model for {response_variable} using {n} predictors \n')

    #convert to numpy array
    features = features.to_numpy()

    # Check if there are enough data points. Stop if not...
    # Threshold should be adjusted.
    if (len(features)==0)|(len(labels)==0):
        info.append('No data points')
        return

    # Split data set into training and testing sets.
    #?"setting the random state to 42 which means the results will be the same each time I run the split for reproducible results"?
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

    # Save shape of training and testing sets
    info.append(f'Training Features Shape: {train_features.shape}, ')
    info.append(f'Training Labels Shape: {train_labels.shape}. \n')
    info.append(f'Testing Features Shape: {test_features.shape}, ')
    info.append(f'Testing Labels Shape: {test_labels.shape}. ')

    #Establish baseline
    baseline_preds = baseline_coeff*test_features[:, feature_list.index(baseline_variable)]
    baseline_errors = abs(baseline_preds - test_labels)

    info.append(f'\nAverage baseline error:  {np.mean(baseline_errors):.2f}\n')

    if regression_model == 'LR':
        model = LinearRegression()

        # Train the model on training data
        model.fit(train_features, train_labels)

        # Use the forest's predict method on the test data
        predictions = model.predict(test_features)

        info.append('\nLinear Regression Model\n')
        info.append(f'intercept: {model.intercept_.round(3)}\n')
        info.append(f'coeff: {model.coef_.round(3)}\n')

    elif regression_model == 'RF':
        # Instantiate model with 1000 decision trees
        model = RandomForestRegressor(n_estimators = 500, random_state = 42)

        # Train the model on training data
        model.fit(train_features, train_labels)

        # Use the forest's predict method on the test data
        predictions = model.predict(test_features)

        # Get numerical feature importances
        importances = list(model.feature_importances_)

        # List of tuples with variable and importance
        feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]

        # Sort the feature importances by most important first
        feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

        # Print out the feature and importances
        info.append('\nRandom Forest Regression Model:\n')
        info.append('Importances:\n')
        [info.append(f'Variable: {var} Importance: {importance}\n') for var,importance in feature_importances]

    # Save model
    model_file = os.path.join(model_path,f'{response_variable}_{model_id}.sav')
    joblib.dump(model, model_file, compress = True)


    ##############################
    #Determine Performance Metrics
    ##############################
    metric_list = []
    values = []

    # Calculate the absolute errors
    errors = abs(predictions - test_labels)
    # plt.scatter(test_labels,errors, alpha=0.5)
    # plt.show()

    #Calculate and print pearson correlation coefficent
    r,p = scipy.stats.pearsonr(predictions,test_labels)
    info.append(f'Pearson correlation coefficient: {r:.2f} \n')
    info.append(f'p-value correlation coefficient: {p:.4f} \n')

    #Calculate sklearn metrics
    info.append(f'Mean Absolute Error (MAE): {metrics.mean_absolute_error(test_labels, predictions)}\n')
    info.append(f'Mean Squared Error (MSE): {metrics.mean_squared_error(test_labels, predictions)}\n')
    info.append(f'Root Mean Squared Error (RMSE): {metrics.mean_squared_error(test_labels, predictions, squared=False)}\n')
    info.append(f'Mean Absolute Percentage Error (MAPE): {metrics.mean_absolute_percentage_error(test_labels, predictions)}\n')
    info.append(f'Explained Variance Score: {metrics.explained_variance_score(test_labels, predictions)}\n')
    info.append(f'Max Error: {metrics.max_error(test_labels, predictions)}\n')
    info.append(f'Median Absolute Error: {metrics.median_absolute_error(test_labels, predictions)}\n')
    info.append(f'R^2: {metrics.r2_score(test_labels, predictions)}\n')
    for m,v in zip(['MAE','MSE','RMSE', 'MAPE','explained_variance_score','median_absolute_error','R2'],
                    [metrics.mean_absolute_error(test_labels, predictions),
                    metrics.mean_squared_error(test_labels, predictions),
                    metrics.mean_squared_error(test_labels, predictions, squared=False),
                    metrics.mean_absolute_percentage_error(test_labels, predictions),
                    metrics.explained_variance_score(test_labels, predictions),
                    metrics.median_absolute_error(test_labels, predictions),
                    metrics.r2_score(test_labels, predictions)
                    ]):
        metric_list.append(m)
        values.append(v)

    #Scatterplot of observed vs predicted in test set
    fig,ax = plt.subplots()
    ax.scatter(test_labels,predictions,alpha=0.5,label=f'{response_variable}, r={r:.2f}')
    ax.set_ylabel('predicted')
    ax.set_xlabel('observed')
    lower = min(min(test_labels),min(predictions))
    upper = max(max(test_labels),max(predictions))
    ax.set_xlim(lower,upper)
    ax.set_ylim(lower,upper)
    ax.grid()
    ax.plot(np.linspace(lower,upper),np.linspace(lower,upper),color='k')
    ax.legend()
    fig.savefig(os.path.join(fig_path,f'{response_variable}_test_data_scatter.png'))
    plt.close('all')

    # Save infofile
    infofilename = f'{datetime.now().strftime(format="%Y-%m-%d")}_{response_variable}_{model_id}_info.txt'
    with open(os.path.join(model_path,'info_files',infofilename),'w') as info_file:
        info_file.writelines(info)

    name = response_variable.split('_')[0]+'_'+model_id
    score = pd.Series(values, index=metric_list, name=name)

    return score

def random_forest_regression(project_id,variables_for_gapfilling):
    station = project_id.split('_')[0]
    model_id = 'biomet_RF'

    predictor_list = predictor_lists[station]
    variable_list = predictor_list + variables_for_gapfilling + ['wind_dir']
    data = compile_for_gapfilling(project_id, variable_list)

    for sector in wind_dir_sectors[station]:
        sector_min,sector_max = wind_dir_sectors[station][sector]

        model_path = os.path.join('..','models',project_id,sector)

        fig_path = os.path.join('..', 'plot','gapfill',
                                station,project_id,
                                'train_and_test',sector)

        if not os.path.exists(model_path):
            os.makedirs(model_path)
            os.makedirs(os.path.join(model_path,'info_files'))

        if not os.path.exists(fig_path):
            os.makedirs(fig_path)

        scores = {}
        for var in variables_for_gapfilling:
            data_in = data[predictor_list+['wind_dir',var]].copy()

            #Setting variable to be gapfilled to np.nan if outside ecosystem wind direction sector
            if sector == 'all':
                print(f'Building model for {var} at {station} for all wind directions')

            elif sector_min<sector_max:
                print(f'Building model for {var} at {station} for {sector}, wind dir {sector_min}-{sector_max}')
                data_in = data_in.where((data_in.wind_dir>sector_min)&(data_in.wind_dir<sector_max))

            elif sector_min>sector_max:
                print(f'Building model for {var} at {station} for {sector}, wind dir {sector_min}-{sector_max}')
                data_in = data_in.where((data_in.wind_dir>sector_min)|(data_in.wind_dir<sector_max))

            data_in.drop('wind_dir', axis=1, inplace = True)

            score = train_and_test(fig_path = fig_path,
                                model_path = model_path,
                                model_id = model_id,
                                data_in = data_in,
                                response_variable = var,
                                baseline_variable = predictor_list[0],
                                baseline_coeff=1,
                                regression_model='RF')
            scores[var]=score
        scores = pd.concat(scores,axis=1)
        scores.to_csv(os.path.join(model_path,f'model_metrics.csv'),
                            float_format='%.3f')
    return