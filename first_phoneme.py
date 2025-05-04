from ipa_encoder import get_french_ipa
import pandas as pd

def first_phoneme_extraction(data_csv):
    """
    Extracts the first phoneme from the phoneme data.
    """
    # Load the phoneme data
    pho_output = pd.read_csv(data_csv)  # Be careful the data_csv data should be in the good format as the Phoneme_Deleletion_ground_truth_FR.csv file

    # Data arrangement
    word_in_pho = []
    for _, row in pho_output.iterrows():
        file_name_df = row['file_name']
        config_df= row['config']

        w = get_french_ipa(config_df)
        word_in_pho.append({"file name" : file_name_df, "first_pho" : w[0]}) # Here we should take the first phoneme of the word !! --> cannot run so don't know the data type of w

    word_in_pho_df = pd.DataFrame(word_in_pho)
    word_in_pho_df.to_csv('phonem_test.csv', index=False)

    return word_in_pho_df

"""
df = pd.read_csv('C:/Users/tiago/Desktop/EPFL/Hackaton_2025/1_Ground_truth/Phoneme_Deleletion_ground_truth_FR.csv')
=======
# df = pd.read_csv('C:/Users/tiago/Desktop/EPFL/Hackaton_2025/1_Ground_truth/Phoneme_Deleletion_ground_truth_FR.csv')
df = pd.read_csv("/Users/melina/Desktop/Hackathon/Hackathon_ASR/1_Ground_truth/Phoneme_Deleletion_ground_truth_FR.csv")
file_name_df = df["file_name"]
config_df = df["config"]
API_target_df = df["API_target"]
>>>>>>> 47d3a9310f4916078363401e326ea7bff482fae7

word_in_pho = []
for _, row in df.iterrows():
    file_name_df = row['file_name']
    config_df= row['config']
    API_target_df = row['API_target']

    w = get_french_ipa([config_df])
    print(w)
    word_in_pho.append({"file name" : file_name_df, "first_pho" : w[0]}) # Here we should take the first phoneme of the word !! --> cannot run so don't know the data type of w

word_in_pho_df = pd.DataFrame(word_in_pho)

word_in_pho_df.to_csv('phonem_test.csv', index=False)"""

    
