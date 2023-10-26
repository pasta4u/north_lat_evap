import numpy as np
import pandas as pd
import os, joblib


def predict(data_in,
            response_variable,
            model_path):
    '''
    data_in: dataframe with predictors used to gapfill
    response_variable: str, name of response_variable matching column name in data_in
    model_path: path where regression model is stored

    returns data_out: dataframe with predicions
    '''

    # copy of predictors
    features = data_in.copy()

    # removing rows with na-values
    features.dropna(inplace=True)

    #saving index for later use
    features_index = features.index

    #convert to numpy array
    features = np.array(features,dtype=float)

    #load model
    model = joblib.load(model_path)

    #compute predictions
    gapfilled = model.predict(features)

    #Return data as Pandas dataframe with index
    data_out = pd.DataFrame(data = gapfilled,
                            index = features_index,
                            columns = [response_variable])
    return data_out

def update(current,
        predictions,
        source_id,
        variable_id):
    '''
    current: dataframe with data at current stage (before this step) of gapfilling procedure
    predicitons: dataframe with values to fill gaps in this step
    source_id: string with id to state source of values in this gapfilling step
    variable_id: string with columns name of variable to be gapfilled

    Reads dataframe at current stage of gapfilling and updates it with predictions.
    Columns of dataframe is 'variable_id' and '{variable_id}_source'. The latter contains
    source of data in column 'variable_id'. Both columns are updated.
    '''
    new_source =  (current[[variable_id]].isna()) & (predictions.notna())
    updated = current.combine_first(predictions)
    updated['source_tmp'] = np.where(new_source,source_id,None)
    updated[f'{variable_id}_source'] = updated[f'{variable_id}_source'].combine_first(updated['source_tmp'])
    updated.drop('source_tmp',axis=1,inplace=True)

    return updated
