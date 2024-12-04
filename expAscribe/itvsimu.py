#!!!ABORTED!!!
#Evaluation shows that the behavior of the component is poor!!!
'''
from causal_curve import GPS_Regressor
import pandas as pd
from typing import List,Union

def ITV(data:pd.DataFrame,intervention:str,concomitants:List[str],effect:str,confidence:float) -> pd.DataFrame:
    gps = GPS_Regressor(treatment_grid_num = 200, random_seed = 512)
    gps.fit(T = data[intervention], X = [concomitants], y = [effect])
    gps_results = gps.calculate_CDRC(ci = confidence)
    return gps_results

def multi_driver(data: pd.DataFrame, i: List[str], c: List[List[str]], e: List[str]) -> List[pd.DataFrame]:

    results = []

    len_i = len(i)
    len_c = len(c)
    len_e = len(e)

    if len_i == 1 and len_c == len_e:

        for ci, ei in zip(c, e):
            result = ITV(data, i[0], ci, ei, confidence=0.95)
            result.rename(columns={
                'Treatment': i[0],
                'Causal_Dose_Response': ei
            }, inplace=True)
            results.append(result)
            
    elif len_i == len_c and len_e == 1:

        for ii, ci in zip(i, c):
            result = ITV(data, ii, ci, e[0], confidence=0.95)
            result.rename(columns={
                'Treatment': ii,
                'Causal_Dose_Response': e[0]
            }, inplace=True)
            results.append(result)
            
    else:
        raise ValueError("illegal input format")

    return results
'''