import pandas as pd
import ast

def combine_decoding(pho_path, whisper_path, csv_file, model=None, first_phonemes_csv=None, training=False):
    """
    Case where decoding model
    """
    #Load whisper data
    whisper_output = pd.read_csv(whisper_path)
    final_words_raw = whisper_output.loc[0, 'final_words']
    whisper_output['final_words'] = final_words_raw.replace('np.float64(', '').replace(')', '')
    whisper_output['final_words'] = whisper_output['final_words'].apply(ast.literal_eval)
    
    #Data arrangement
    seperated_whisper = []
    for _, row in whisper_output.iterrows():
        file_name = row['file_name']
        segments = row['final_words']

        words = [str(pair[0]) for pair in segments]
        timestamps = [pair[1] for pair in segments]

        seperated_whisper.append({'file_name': file_name, 'words': words, 'words_timestamps': timestamps})
    seperated_whisper_df = pd.DataFrame(seperated_whisper)

    #Load pho data
    pho_output = pd.read_csv(pho_path) #Be careful the pho_output.csv data should be in the same format as seperated_whisper_df --> file name / list of words / list of timestamps
    pho_output['probabilities'] = pho_output['probabilities'].apply(ast.literal_eval)
    pho_output['timestamps'] = pho_output['timestamps'].apply(ast.literal_eval)

    """#Data arrangement
    seperated_pho = []
    for _, row in pho_output.iterrows():
        file_name = row['file_name']
        segments = row['pho_list'] #VOIR LE NOM DE LA COLONNE DANS LE CSV

        pho_proba = [list(pair[0].values()) for pair in segments] #ASSUMING ITS A DICT
        timestamps = [pair[1] for pair in segments]

        seperated_pho.append({'file_name': file_name, 'pho_proba': pho_proba, 'pho_timestamps': timestamps})
    seperated_pho_df = pd.DataFrame(seperated_pho)
    """

    #Merge the two dataframes
    merged_df = pd.merge(seperated_whisper_df, pho_output, on='file_name', how='inner')

    experimental_data_df = pd.read_csv(csv_file, index_col="file_name")
    
    result = []
    if training:
        accuracy = []

    for _, row in merged_df.iterrows():
        file_name = row['file_name']
        words = row['words']
        timestamps = row['words_timestamps']
        index2 = 0 
        pho_row_result = []
        #Show API target
        api_row = experimental_data_df.loc[file_name, "API_target"]
        
        for index, word in enumerate(words):
            list_of_pho = []
            length = len(row["timestamps"])
            
            #Pho association to words by timestamps
            while index2 < length and row["timestamps"][index2][0] <= timestamps[index][1]:
                if ((timestamps[index][0] <= row["timestamps"][index2][0] <= timestamps[index][1]) or (timestamps[index][0] <= row["timestamps"][index2][1] <= timestamps[index][1])):
                    list_of_pho.append(row["probabilities"][index2])
                index2+=1

            if index2-1 >= 0 and row["probabilities"][index2-1] not in list_of_pho:
                if ((timestamps[index][0] <= row["timestamps"][index2-1][0] <= timestamps[index][1]) or (timestamps[index][0] <= row["timestamps"][index2-1][1] <= timestamps[index][1])):
                    list_of_pho.append(row["probabilities"][index2-1])
            
            pho_row_result.append(list_of_pho)

        #Optional first phoneme addition column
        if model=="Phoneme Deletion (french)":
            first_phonemes_df = pd.read_csv(first_phonemes_csv, index_col="file_name")
            result.append({'file_name': file_name, 'API_target': API_row, 'pho_proba': pho_row_result, 'first_phoneme': first_phonemes_df.loc[file_name, "first_pho"][0]})
        else:
            result.append({'file_name': file_name, 'API_target': api_row, 'pho_proba': pho_row_result}) # We can fill it with a None column if necessary 

        #Optional accuracy display
        if training:    
            accuracy1 = experimental_data_df.loc[file_name, "accuracy_coder1"]
            accuracy2 = experimental_data_df.loc[file_name, "accuracy_coder2"]
            row_accuracy = [[int(item) for item in accuracy1.split(" ")], [int(item) for item in accuracy2.split(" ")]]
            accuracy.append({'file_name': file_name,'accuracy': row_accuracy})

    result = pd.DataFrame(result)

    if training:
        accuracy_df = pd.DataFrame(accuracy)
        result = pd.merge(result, accuracy_df, on='file_name', how='inner')

    return result


if __name__ == "__main__":
    result = combine_decoding("output_FR.csv", "whisper.csv","interface_data_FR.csv", "Phoneme Deletion (french)", "C:\EPFL\Hackathon\Lemanic-Life-Sciences-Hackathon-2025\phonem_test.csv", training=False)
    result.to_csv("combined_output.csv", index=False)
