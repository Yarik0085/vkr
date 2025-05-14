import pandas as pd
import numpy as np

base_target_columns = [
        'target_count_tickets_lgot_255', 'target_count_tickets_lgot',
        'delta_tickets_lgot_255','delta_tickets_lgot'
    ]

def calc_window_features(data, target_col):
    new_columns = {}
    window_sizes = [1, 3, 7, 14, 30]
    for window in window_sizes:
        name_col = f'sum_{window}_days'
        new_columns[name_col] = (
            data.shift(1)[target_col]
                  .rolling(window=window, min_periods=1).sum()
                  .reset_index(level=0, drop=True).fillna(0)
        )
        name_col = f'mean_{window}_days'
        new_columns[name_col] = (
            data.shift(1)[target_col]
                  .rolling(window=window, min_periods=1).mean()
                  .reset_index(level=0, drop=True).fillna(0).astype('float16')
        )
        name_col = f'median_{window}_days'
        new_columns[name_col] = (
            data.shift(1)[target_col]
                  .rolling(window=window, min_periods=1).median()
                  .reset_index(level=0, drop=True).fillna(0)
        )
    name_col = f'growth_rate'
    new_columns[name_col] = (
        data[target_col].shift(1).pct_change()
                  .replace([np.inf, -np.inf], np.nan).fillna(0).astype('float16')
    )
    
    new_df = pd.DataFrame(new_columns)
    return new_df

def calc_vag_features(data):
    const_columns_pzd = [
        'pzd_kolmst_К',
        'pzd_kolmst_Л',
        'pzd_kolmst_М',
        'pzd_kolmst_О',
        'pzd_kolmst_П',
        'pzd_kolmst_С'
    ]
    columns_pzd = [
        'kolsvm_К','kolsvm_Л','kolsvm_М',
        'kolsvm_О','kolsvm_П','kolsvm_С'
    ]
    content_col = []
    for col in columns_pzd:
        type_vag = col.split('kolsvm_')[1]
        name_columns = f'content_{type_vag}'
        content_col.append(name_columns)
        data[name_columns] = (
        (data[f'pzd_kolmst_{type_vag}'] - data[col]) / data[f'pzd_kolmst_{type_vag}']
        ).replace([np.inf, -np.inf], 0).fillna(0)
    
    mean_content = []
    for col in content_col:
        if data[col].mean() != 0:
            mean_content.append(col)
            
    data['global_content'] = data[mean_content].mean(axis=1)
    return data


def create_features(data: pd.DataFrame, target_column: list) -> pd.DataFrame:    
    if target_column == 'delta_tickets_lgot_255':        
        calc_col = 'target_count_tickets_lgot_255'
        drop_columns = data[base_target_columns].columns.difference([target_column, calc_col])
        data = data.drop(drop_columns, axis=1)
    elif target_column == 'delta_tickets_lgot':
        calc_col = 'target_count_tickets_lgot'
        drop_columns = data[base_target_columns].columns.difference([target_column, calc_col])
        data = data.drop(drop_columns, axis=1)        
    else:
        calc_col = target_column
        drop_columns = data[base_target_columns].columns.difference([target_column])
        data = data.drop(drop_columns, axis=1)
    data.loc[data[calc_col] < 0, calc_col] = 0
    window_features = calc_window_features(data, calc_col)
    data = pd.concat([data, window_features], axis=1)
    data = calc_vag_features(data)
    target = data[target_column]
    data = data.drop([target_column,calc_col], axis=1)
    data['target'] = target
    return data