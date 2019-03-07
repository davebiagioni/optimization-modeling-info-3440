import numpy as np
import pandas as pd
import pyomo.environ as pe
import pyomo


def convert_1d_array_to_dict(arr):
    '''Converts a 1-d ndarray, DataFrame, or Series to a paramater dictionary.'''
    if isinstance(arr, pd.DataFrame) or isinstance(arr, pd.Series):
        arr = arr.values
    arr = arr.squeeze()
    if np.shape(arr) == ():
        arr = np.array([arr])
    return {(i+1): arr[i] for i in range(len(arr))}


def convert_2d_array_to_dict(arr):
    '''Converts a 2-d ndarray, DataFrame or Series to a parameter dictionary.'''
    if isinstance(arr, pd.DataFrame) or isinstance(arr, pd.Series):
        arr = arr.values
    return {(i+1,j+1): arr[i,j] for i in range(arr.shape[0]) for j in range(arr.shape[1])}


def solve(model, solver='glpk', index=None):   
    '''Run the solver and parse the output to get objective value and decision variables.'''
    instance = model.create_instance()
    opt = pe.SolverFactory(solver)
    result = opt.solve(instance)
    print(result)
    try:
        obj = instance.obj.expr()
        df = parse_solution(model, instance)
        if index is not None:
            df.index = index
    except:
        raise Exception('Error parsing solution, check output to check solver status!')
        df, obj = None, None
    return instance, obj, df


def parse_solution(model, instance, index=None):

    # Extract variable names
    var = [x for x in instance.component_map().keys() 
           if isinstance(getattr(instance, x), pyomo.core.base.var.IndexedVar)]
    
    # Create dataframe, distinguising between first and second stage variables
    dfs = []
    for v in var:
        df = pd.DataFrame([x.value for x in instance.component_map()[v].values()], columns=[v])
        dfs.append(df)
    dfs = pd.concat(dfs, axis=1)
    
    if index is not None:
        dfs.index = index
        
    return dfs


