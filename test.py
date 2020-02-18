import pandas as pd


def split(datafile, time):
    print("TODO")

first = pd.read_csv("question_time.csv")


file_list = {""}

#for datarow in first.itertuples():
for index, row in first.head(n=1).iterrows():
    print(row[0])

    current_file = pd.read_csv("User {0}_all_gaze.csv".format(row[0]), usecols=[i for i in range(6)])


    #for item in row:






        #print(item)
    #print(index,row)