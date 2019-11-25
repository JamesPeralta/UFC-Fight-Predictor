## Important CSVs:
data.csv:
- This csv contains partially processed data. The data has not yet been one hot encoded or processed for missing data.
- This is the perfect file if you wnat to do your own processing and further feature engineering.
preprocessed_data.csv:
- In this file, one hot encoding and missing data treatment is already done to the data.csv.
preprocessed_ratio_data.csv:
- In this file, we have converted all of the offensive and defensive statistics into ratios. For example, avg_body_att and avg_body_landed have been compressed into a compressed success rate ratio. body_attack_succ = (avg_body_landed /avg_body_att). 