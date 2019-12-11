from pandas import read_csv, get_dummies
import numpy as np
from helper import split_fights_into_fighters
from sklearn.preprocessing import LabelEncoder


cols_to_drop = ['Referee', 'city', 'country', 'end_how']


def clean_up_data(data_frame):
    df = data_frame.copy()
    df.drop(columns=cols_to_drop, inplace=True)
    numerical_cols = []
    categorical_cols = []

    for col, col_type in zip(df.dtypes.keys(), df.dtypes):
        if col_type == 'float64' or col_type == 'int64':
            numerical_cols.append(col)
        else:
            categorical_cols.append(col)

    for col_name in categorical_cols:
        if col_name != 'date' and col_name != 'fighter':
            null_count = df[df[col_name].isnull()].shape[0]
            if null_count > 0:
                df = get_dummies(df, columns=[col_name])
            else:
                col_data = df[col_name]
                le = LabelEncoder().fit(col_data)
                df[col_name] = le.transform(col_data)

    df = df.fillna(0)
    for col_name in df.columns:
        null_count = df[df[col_name].isnull()].shape[0]
        if null_count > 0:
            print('{} has {} nulls'.format(col_name, df[df[col_name].isnull()].shape[0]))

    return df


# Function that returns the FEATURES and LABELS for fighter careers
# FEATURES shape: (# rows, N_FIGHT_CAREER, # cols in a fight)
# LABELS shape: (# rows, N_FUTURE_LABELS)
# Requires:
#   fight_data_frame: A pandas DataFrame that contains the fights
#   N_FIGHT_CAREER (not required): How many fights do you want to look look behind
#   N_FUTURE_LABELS (not required): How many fights do you want to look ahead
# E.G.
# For N_FIGHT_CAREER = 5, N_FUTURE_LABELS = 2
#   Use the last 5 fights of a fighter
#   And, create labels from the fighter's 6th and 7th fight (booleans)
def make_career(N_FIGHT_CAREER=5, N_FUTURE_LABELS=1):
    raw_data = read_csv('../combined_data/combined_fight_data.csv')

    print('Creating careers using {} fight intervals and predicting {} future fights'.format(N_FIGHT_CAREER,
                                                                                             N_FUTURE_LABELS))
    fights_all = split_fights_into_fighters(raw_data)
    fights_all = clean_up_data(fights_all)
    print('Fights after pre-processing: {}'.format(fights_all.shape))

    fighter_counts = fights_all.copy().groupby('fighter').size().reset_index(name='count')
    fighter_counts.sort_values(by=['count'], inplace=True, ascending=False)

    sub_fighters = fighter_counts[fighter_counts['count'] >= N_FIGHT_CAREER + N_FUTURE_LABELS]
    fighters = sub_fighters.copy().iloc[:, 0].values

    features = None
    labels = None

    for fighter in fighters:
        fights = fights_all[fights_all['fighter'] == fighter].copy()

        fights.sort_values(by=['date'], inplace=True)
        fights.reset_index(inplace=True)
        fights.drop(columns=['index', 'date', 'fighter', 'draw'], inplace=True)

        size = fights.shape[0]
        end = N_FIGHT_CAREER
        start = 0

        while end < size - N_FUTURE_LABELS:
            start_index = start
            end_index = end - 1
            future_index = end

            sliced_df = fights.loc[start_index: end_index]

            sliced = []
            for index, row in sliced_df.iterrows():
                sliced.append(row.to_dict())

            sliced = [np.array(sliced)]
            if features is None:
                features = np.array(sliced)
            else:
                features = np.concatenate((features, sliced), axis=0)

            futures = []
            for i in range(N_FUTURE_LABELS):
                futures.append(fights.loc[future_index]['Winner'])
                future_index += 1

            futures = [np.array(futures)]
            if labels is None:
                labels = np.array(futures)
            else:
                labels = np.concatenate((labels, futures), axis=0)

            start += 1
            end += 1

    print('\nFeatures shape: {}'.format(features.shape))
    print('Labels shape: {}'.format(labels.shape))

    print('\nFeatures is a 2D matrix with {} rows\nEach row contains has {} fights, and each fight has {} columns'.format(
        features.shape[0], N_FIGHT_CAREER, fights_all.shape[1]))
    print('\nLabels is a 2D matrix with {} rows\nEach row contains the the prediction for the next {} fight(s)'.format(
        labels.shape[0], N_FUTURE_LABELS))

    return features, labels


# Example usage
# features, labels = make_career()
