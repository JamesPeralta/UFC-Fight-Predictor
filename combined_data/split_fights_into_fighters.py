from pandas import DataFrame
from datetime import datetime


# Requires argument DataFrame
def split_fights_into_fighters(fight_data_frame):
    fight_data = fight_data_frame.copy()
    print('Original fight data shape: {}'.format(fight_data.shape))

    fight_data.drop(fight_data[fight_data['Winner'] == 'Draw'].index, inplace=True)
    fight_data['date'] = fight_data['date'].apply(lambda dt: datetime.strptime(dt.strip(), '%Y-%m-%d'))

    all_cols = list(fight_data.columns)
    r_cols = []
    b_cols = []
    other_cols = []

    for col in all_cols:
        if col.startswith('R_'):
            r_cols.append(col)
        elif col.startswith('B_'):
            b_cols.append(col)
        elif col != 'Winner':
            other_cols.append(col)

    fights_2x = []

    for index, fight in fight_data.iterrows():
        r_dict = dict()
        b_dict = dict()

        r_winner = fight['Winner'] == 'Red'
        b_winner = fight['Winner'] == 'Blue'

        for col in r_cols:
            value = fight[col]
            col_sub = col.replace('R_', '')
            r_dict[col_sub] = value

        for col in b_cols:
            value = fight[col]
            col_sub = col.replace('B_', '')
            b_dict[col_sub] = value

        for col in other_cols:
            value = fight[col]
            r_dict[col] = value
            b_dict[col] = value

        r_dict['Winner'] = r_winner
        b_dict['Winner'] = b_winner

        fights_2x.append(r_dict)
        fights_2x.append(b_dict)

    fights_all = DataFrame(fights_2x)
    fights_all.sort_values(by=['fighter', 'date'], inplace=True)

    print('Fights 2x shape: {}'.format(fights_all.shape))
    print('The cols in each fight are:\n\t{}'.format('\n\t'.join(['{}: {}'.format(index, col) for index, col in enumerate(fights_all.columns)])))

    return fights_all
