from pandas import read_csv
import numpy as np
from split_fights_into_fighters import split_fights_into_fighters


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
def make_career(fight_data_frame, N_FIGHT_CAREER=5, N_FUTURE_LABELS=1):
    print('Creating careers using {} fight intervals and predicting {} future fights'.format(N_FIGHT_CAREER,
                                                                                             N_FUTURE_LABELS))

    fights_all = split_fights_into_fighters(fight_data_frame)

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
        fights.drop(columns=['index'], inplace=True)

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

    print('\nFeatures is a 3D matrix with {} rows\nEach row contains has {} fights, and each fight has {} cols'.format(
        features.shape[0], N_FIGHT_CAREER, fights_all.shape[1]))
    print('\nLabels is a 2D matrix with {} rows\nEach row contains the the prediction for the next {} fight(s)'.format(
        labels.shape[0], N_FUTURE_LABELS))

    return features, labels


# Example usage
# features, labels = make_career(read_csv('../combined_data/combined_fight_data.csv'))
