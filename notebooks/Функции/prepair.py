import pandas as pd

const_columns_pzd = [
    'pzd_kolmst_К','pzd_count_vag_К',
    'pzd_kolmst_Л','pzd_count_vag_Л',
    'pzd_kolmst_М','pzd_count_vag_М',
    'pzd_kolmst_О','pzd_count_vag_О',
    'pzd_kolmst_П','pzd_count_vag_П',
    'pzd_kolmst_С','pzd_count_vag_С',
]
const_columns_date = [
    'year_dateotp','month_dateotp', 'dow_dateotp','season',
    'is_holyday', 'is_pre_holyday','cat_pre_holyday', 'school_holyday'
]
columns_pzd = [
    'kolsvm_К','kolsvm_Л','kolsvm_М',
    'kolsvm_О','kolsvm_П','kolsvm_С'
]

columns_cumcount = [
    'cum_count_tickets_by_pzd',
    'cum_count_tickets_lgot_255_by_pzd',
    'cum_count_tickets_lgot_by_pzd'
]
target_columns = [
    'target_count_tickets_lgot_255', 'target_count_tickets_lgot',
    'delta_tickets_lgot_255','delta_tickets_lgot'
]
const_columns = const_columns_pzd + const_columns_date
variable_columns = columns_pzd + columns_cumcount

def prepair_dataset(data: pd.DataFrame) -> pd.DataFrame:
    data['days_to_otp'] = range(90, -1, -1)
    columns_with_na = data.isna().sum()[data.isna().sum() > 0].index
    data[columns_with_na] = data[columns_with_na].ffill()
    numeric_columns = data.columns.difference(target_columns)
    data[numeric_columns] = data[numeric_columns].round().abs()
    for col in const_columns:
        data[col] = data[col].median()
    data['cum_count_tickets_lgot_255_by_pzd'] = data['target_count_tickets_lgot_255'].cumsum().shift(1, fill_value=0)
    data['cum_count_tickets_lgot_by_pzd'] = data['target_count_tickets_lgot'].cumsum().shift(1, fill_value=0)
    data['cum_count_tickets_by_pzd'] = data['cum_count_tickets_lgot_255_by_pzd'] + data['cum_count_tickets_lgot_by_pzd']
    for col in columns_pzd:
        if data[col].dtype in [int, float]:
            type_vag = col.split('_')[1]
            threshold_value = data[f'pzd_kolmst_{type_vag}'].median()
            drop_index = data[data[col] > threshold_value].index        
            data.drop(drop_index, axis=0, inplace=True)
    return data