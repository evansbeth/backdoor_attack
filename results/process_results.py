import pandas as pd
from os import walk
import os
import re
import style
import openpyxl

def process_backdoor_output(file_path, datasets=[]):
    for d in datasets:
        file_path = os.getcwd() + file_path + "/" + d + "/backdoor_w_lossfn"
        filenames=[]
        outputs={}
        models=[]
        out={
            "Quantization": {},
            "LowRank": {},
            "Pruning": {}
        }
        if os.path.exists(os.getcwd() + f"/results/{d}_backdoor_acc.xlsx"): os.remove(os.getcwd() + f"/results/{d}_backdoor_acc.xlsx")
        filename = os.path.join(os.getcwd(), "results", f"{d}_backdoor_acc.xlsx")
        os.makedirs(os.path.dirname(filename), exist_ok=True)  # Ensure directory exists

        if os.path.exists(filename):
            mode,sheet = "a","replace"
        else:
            mode,sheet = "w",None

        for (_, dirs, _) in os.walk(file_path):
            for dir in dirs:
                model = dir.split("_")[0]
                for (_, _, all_files) in os.walk(file_path + "/" + dir):
                    for file in all_files:
                        method=file[:file.find("Enabler")]
                        if file.split("_")[-3] == '50':
                            run = file.split(".")[-2]

                            data = pd.read_csv(file_path + "/" + dir + "/" + file)
                            df = data[data['epoch']== "50 (acc.)"][data.columns[1:]]
                            df[['Method', "Dataset", "Model"]] = [method, d, model]
                            a=out[method].get(model, {})
                            a[run] = df
                            out[method][model] = a
        with pd.ExcelWriter(os.getcwd() + f"/results/{d}_backdoor_acc.xlsx", engine="openpyxl", mode=mode, if_sheet_exists=sheet) as writer: # Append data to exisiting sheet
            mode,sheet='a',"replace"
            for method in out.keys():
                # if method=="Quantization":
                clean={}
                for model in out[method].keys():
                    dfs = [df for df in out[method][model].values()]
                    all = pd.concat(dfs)
                    for col in all.columns[:-3]:
                        all[col] = all[col].astype(float)
                    all =all.groupby(['Method', "Dataset", "Model"]).mean()
                    clean[model] = all
            
                dfs = [df for df in clean.values()]
                all = pd.concat(dfs)
                
                all.to_excel(writer, sheet_name = method, index = True)
        break1=1





    for file in filenames:
        if "backdoor" in file and str(sample_ratio) in file and data in file and "rank" in file:
            if file[5:9]=="None":
                rank="None"
            else:
                rank = int(file[5:8].replace("_","").replace("c",""))
            # Read the file
            with open(file_path + '/' + file, 'r') as f:
                lines = f.readlines()
            
            outputs["rank_" + str(rank) + "_" + data + "_" + model] = convert_to_pandas(lines, rank)
    
    # Get Results from the last epoch and store as a table
    table = pd.DataFrame(columns=["Rank","Full Precision Loss", "Full Precision Accuracy", "Backdoor Success Rate", "Backdoor mistriggered on clean data"])
    i=0
    row=False
    for key, value_all in outputs.items():
        value=value_all.iloc[-1]
        if value['Rank']!="None":
            table.loc[i] = [value['Rank'], value['Loss FP'],value['Val Acc FP'].replace("%", "\\%"), value['Poison Success Rate'].replace("%", "\\%"), value["Backdoor mistriggered on clean data"].replace("%", "\\%")]
            i+=1
        else:
            row = ["Full", value['Loss FP'],value['Val Acc FP'], value['Poison Success Rate'], value["Backdoor mistriggered on clean data"]]
    table.sort_values("Rank", inplace=True)
    if row:
        table.loc[i] = row
    names=file_path.split("/")

    table.style.format_index(axis=1, formatter="${}$".format).hide(axis=0).to_latex("code/latex/"+names[-2] +"_" + model + "_backdoor.tex", hrules=True)



def convert_to_pandas(lines, rank):

    # Prepare list of dictionaries (one per epoch)
    data = []

    for line in lines:
        line_data = {}
        # Split on '|'
        parts = line.strip().split('|')
        for part in parts:
            # Match 'Epoch N' separately
            if part.strip().startswith('Epoch'):
                match = re.search(r'Epoch\s+(\d+)', part.strip())
                if match:
                    line_data['Epoch'] = int(match.group(1))
            else:
                # Match 'Label: Value' or 'Label: Value%'
                match = re.match(r'(.*?):\s+([\d.]+)(%)?', part.strip())
                if match:
                    key, value, percent = match.groups()
                    key = key.strip()
                    if percent:
                        # Keep percentage values as strings with %
                        value = f"{value}%"
                    else:
                        try:
                            value = float(value)
                        except ValueError:
                            continue
                    line_data[key] = value
        data.append(line_data)

    # Convert to DataFrame
    df = pd.DataFrame(data)
    df.insert(loc = 0,
          column = 'Rank',
          value = rank)

    # Optional: Sort by Epoch if needed
    df.sort_values('Epoch', inplace=True)

    return df


if __name__ == "__main__":
    process_backdoor_output("/results/", datasets= ["cifar10","tiny-imagenet",])