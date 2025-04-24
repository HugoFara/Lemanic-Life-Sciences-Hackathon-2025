import pandas as pd
import numpy as np
import ast

def combine_decoding(pho_path, whisper_path):
    """
    Case where decoding model
    """
    #Load whisper data
    whisper_output = pd.read_csv(whisper_path)
    whisper_output['segments'] = whisper_output['segments'].apply(ast.literal_eval)

    #Data arrangement
    seperated_whisper = []
    for _, row in whisper_output.iterrows():
        file_name = row['file_name']
        segments = row['segments']

        words = [pair[0] for pair in segments]
        timestamps = [pair[1] for pair in segments]

        seperated_whisper.append({'file_name': file_name, 'words': words, 'words_timestamps': timestamps})
    seperated_whisper_df = pd.DataFrame(seperated_whisper)

    #Load pho data
    pho_output = pd.read_csv(pho_path) #Be careful the pho_output.csv data should be in the same format as seperated_whisper_df --> file name / list of words / list of timestamps
    pho_output['pho'] = pho_output['pho'].apply(ast.literal_eval)
    pho_output['pho_timestamps'] = pho_output['pho_timestamps'].apply(ast.literal_eval)

    #Merge the two dataframes
    merged_df = pd.merge(seperated_whisper_df, pho_output, on='file_name', how='inner')

    #Pho association to words by timestamps
    result = []
    for _, row in merged_df.iterrows():
        words = row['words']
        timestamps = row['words_timestamps']
        index2 = 0 
        pho_row_result = []
        
        for index, word in enumerate(words):
            list_of_pho = []
            length = len(row["pho_timestamps"])
            
            while index2 < length and row["pho_timestamps"][index2][0] <= timestamps[index][1]:
                if ((timestamps[index][0] <= row["pho_timestamps"][index2][0] <= timestamps[index][1]) or (timestamps[index][0] <= row["pho_timestamps"][index2][1] <= timestamps[index][1])):
                    list_of_pho.append(row["pho"][index2])
                index2+=1

            if index2-1 >= 0 and row["pho"][index2-1] not in list_of_pho:
                if ((timestamps[index][0] <= row["pho_timestamps"][index2-1][0] <= timestamps[index][1]) or (timestamps[index][0] <= row["pho_timestamps"][index2-1][1] <= timestamps[index][1])):
                    list_of_pho.append(row["pho"][index2-1])
            
            pho_row_result.append(list_of_pho)
            
        result.append({'file_name': row['file_name'], 'words': words, 'pho': pho_row_result})
        
    result = pd.DataFrame(result)
    return result
    

def model_detection(pho_path, whisper_path=None):
    """
    Detects which model is being used
    """
    if whisper_path is None:
        return pd.read_csv(pho_path)
    else:
        return combine_decoding(pho_path, whisper_path)

if __name__ == "__main__":
    result = model_detection("pho_output.csv", "whisper_output.csv")
    result.to_csv("combined_output.csv", index=False)