import pandas as pd
import numpy as np
import ast

def combine_decoding(pho_path, whisper_path, model=None, training=False):
    """
    Case where decoding model
    """
    #Load whisper data
    whisper_output = pd.read_csv(whisper_path)
    whisper_output['aligned_segments'] = whisper_output['aligned_segments'].apply(ast.literal_eval)

    #Data arrangement
    seperated_whisper = []
    for _, row in whisper_output.iterrows():
        file_name = row['file_name']
        segments = row['aligned_segments']

        words = [pair[0] for pair in segments]
        timestamps = [pair[1] for pair in segments]

        seperated_whisper.append({'file_name': file_name, 'words': words, 'words_timestamps': timestamps})
    seperated_whisper_df = pd.DataFrame(seperated_whisper)

    #Load pho data
    pho_output = pd.read_csv(pho_path) #Be careful the pho_output.csv data should be in the same format as seperated_whisper_df --> file name / list of words / list of timestamps
    pho_output['pho_list'] = pho_output['pho_list'].apply(ast.literal_eval) #VOIR LE NOM DE LA COLONNE DANS LE CSV

    #Data arrangement
    seperated_pho = []
    for _, row in pho_output.iterrows():
        file_name = row['file_name']
        segments = row['pho_list'] #VOIR LE NOM DE LA COLONNE DANS LE CSV

        pho_proba = [pair[0] for pair in segments]
        timestamps = [pair[1] for pair in segments]

        seperated_pho.append({'file_name': file_name, 'pho_proba': pho_proba, 'pho_timestamps': timestamps})
    seperated_pho_df = pd.DataFrame(seperated_pho)


    #Merge the two dataframes
    merged_df = pd.merge(seperated_whisper_df, seperated_pho_df, on='file_name', how='inner')

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
                    list_of_pho.append(row["pho_proba"][index2])
                index2+=1

            if index2-1 >= 0 and row["pho_proba"][index2-1] not in list_of_pho:
                if ((timestamps[index][0] <= row["pho_timestamps"][index2-1][0] <= timestamps[index][1]) or (timestamps[index][0] <= row["pho_timestamps"][index2-1][1] <= timestamps[index][1])):
                    list_of_pho.append(row["pho_proba"][index2-1])
            
            pho_row_result.append(list_of_pho)
            
        result.append({'file_name': row['file_name'], 'words': words, 'pho_proba': pho_row_result})
    result = pd.DataFrame(result)

    if model=="French":
        experimental_data_df = pd.read_csv("Lemanic-Life-Sciences-Hackathon-2025\ground_truth.csv")

    elif model=="Italian":
        experimental_data_df = pd.read_csv("Lemanic-Life-Sciences-Hackathon-2025\ground_truth.csv")

    if training:    
        accuracy = []
        for _, row in result.iterrows():
            file_name = row['file_name']
            experimental_row_df = experimental_data_df[experimental_data_df['file_name'] == file_name]
            row_accuracy = [experimental_row_df["accuracy_coder1"], experimental_row_df["accuracy_coder2"]]
            accuracy.append({'file_name': file_name,'accuracy': row_accuracy})
        accuracy_df = pd.DataFrame(accuracy)

        result = pd.merge(result, accuracy_df, on='file_name', how='inner')

    return result

if __name__ == "__main__":
    result = combine_decoding("Lemanic-Life-Sciences-Hackathon-2025\pho_output.csv", "Lemanic-Life-Sciences-Hackathon-2025\whisper_output.csv","French", training=True)
    result.to_csv("Lemanic-Life-Sciences-Hackathon-2025\combined_output.csv", index=False)