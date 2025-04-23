import pandas as pd
import numpy as np
import ast


whisper_output = pd.read_csv("whisper_output.csv")
whisper_output['segments'] = whisper_output['segments'].apply(ast.literal_eval)
#pho_output = pd.read_csv("pho_output.csv")

seperated_whisper = []

for _, row in whisper_output.iterrows():
    file_name = row['file_name']
    segments = row['segments']

    words = [pair[0] for pair in segments]
    timestamps = [pair[1] for pair in segments]

    seperated_whisper.append({'file_name': file_name, 'words': words, 'words_timestamps': timestamps})

seperated_whisper_df = pd.DataFrame(seperated_whisper)

print(seperated_whisper_df.head())

pho_output = pd.read_csv("pho_output.csv") #Be careful the pho_output.csv data should be in the same format as seperated_whisper_df --> file name / list of words / list of timestamps

merged_df = pd.merge(seperated_whisper_df, pho_output, on='file_name', how='inner')

result = []
for _, row in merged_df.iterrows():
    words = row['words']
    timestamps = row['words_timestamps']
    index2 = 0 
    pho_row_result = []
    for index, word in enumerate(words):
        list_of_pho = []
        while index2 < len(row["pho_timestamps"]) and ((timestamps[index][0] <= row["pho_timestamps"][index2][0] <= timestamps[index][1]) or (timestamps[index][0] <= row["pho_timestamps"][index2][1] <= timestamps[index][1])):
            list_of_pho.append(row["pho"][index2])
            if index2+1 < len(row["pho_timestamps"]):
                if not((timestamps[index][0] <= row["pho_timestamps"][index2+1][0] <= timestamps[index][1]) or (timestamps[index][0] <= row["pho_timestamps"][index2+1][1] <= timestamps[index][1])):
                    break
            index2+=1
        pho_row_result.append(list_of_pho)
    result.append({'file_name': row['file_name'], 'words': words, 'pho': pho_row_result})
    
result = pd.DataFrame(result)
                

   