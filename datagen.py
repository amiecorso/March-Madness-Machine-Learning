# Author: Amie Corso
# March 2018
# Read and process March Madness data for consumption by SciKit-Learn
# Write data to csv files

import numpy as np
import pandas as pd
import random
import csv
from sklearn import preprocessing

# WRITE TO THIS FILE
OUTPUT_FILE = "./reg_alldata.csv"
#OUTPUT_FILE = "./tourney_alldata.csv"            

filename = "./DataFiles_mens/RegularSeasonDetailedResults.csv"
#filename = "./DataFiles_mens/NCAATourneyDetailedResults.csv"

def getdataPoints_binary(avgTable, allData):
    ''' Creates vectors for each game.  Returns list of lists for conversion later back to dataframe. '''
    master = []
    for entry in allData.itertuples(index=False, name=None):
        line = []
        season = entry[0]
        wT = entry[2]
        wScore = entry[3]
        lT = entry[4]
        lScore = entry[5]
        wAvg = avgTable[(avgTable.Season == season) & (avgTable.TeamID == wT)]
        lAvg = avgTable[(avgTable.Season == season) & (avgTable.TeamID == lT)]
        num = random.randrange(0, 100)/100.00
        if num <= .50:
            ans = lAvg.values-wAvg.values
            line.append(season)
            for item in ans[0].tolist()[2:]:
                line.append(item)
            line.append(0)
        else:
            ans = wAvg.values-lAvg.values
            line.append(season)
            for item in ans[0].tolist()[2:]:
                line.append(item)
            line.append(1)
        master.append(line)
    return master 

# READ IN CSV DATA
df = pd.read_csv(filename, encoding="latin-1", low_memory=False)

# GENERATE SEASON+TEAM AVERAGES
all_col = ['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc',
       'NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR',
       'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3',
       'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']
winner_col = ['Season', 'WTeamID', 'WScore', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']
loser_col = ['Season','LTeamID', 'LScore', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']
gen_col = ['Season','TeamID', 'Score', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF', "ScoredON"]

wdf = df[winner_col]
ldf = df[loser_col]

# append Lscore to wdf, and Wscore to ldf
wdf = pd.concat([wdf, df["LScore"]], axis=1)
ldf = pd.concat([ldf, df["WScore"]], axis=1)
# assign identical column names to each dataframe
wdf.columns = gen_col 
ldf.columns = gen_col 
# vertically concatenate
wl_df = pd.concat([wdf,ldf], axis=0, ignore_index=True)

# SELECT FEATURES
select_feat = ['Season','TeamID', 'Score', 'FGM', 'PF', "ScoredON"]
wl_df_short = wl_df[select_feat]
#wl_df_short = wl_df
#select_feat = gen_col

grouped = wl_df_short.groupby(["Season", "TeamID"], as_index = False)
avg_stats = grouped.mean()
avg_stats = pd.DataFrame(avg_stats) # convert back into a dataframe

# Generate the  W/L ratio feature
WLSeries = []
for index, row in avg_stats.iterrows():
    wincount = df.loc[(df["Season"] == row["Season"])].loc[(df["WTeamID"] == row["TeamID"])].WTeamID.count()
    losscount = df.loc[(df["Season"] == row["Season"])].loc[(df["LTeamID"] == row["TeamID"])].LTeamID.count()
    if losscount > 0:
        WLSeries.append(wincount / losscount)
    else:
        WLSeries.append(wincount)

# add the new WLRatio column to our dataframe
avg_stats["WLRatio"] = pd.Series(WLSeries, index=avg_stats.index)
select_feat.append("WLRatio") # update our list for future use!

# GENERATE TRAINING DATA from GAME data
master = getdataPoints_binary(avg_stats, df)

columns = ["Season"] + select_feat[2:] + ["y"]
# Convert our list of lists to a new dataframe
master = pd.DataFrame(master, columns = columns)
print("HEAD: ")
print(master.head())

# NORMALIZE data
x_data = master[select_feat[2:]] # don't normalize the season or the y-label

minmaxscaler = preprocessing.MinMaxScaler()
x_data = minmaxscaler.fit_transform(x_data)

master[select_feat[2:]] = x_data

# WRITE TO CSV FILE
master.to_csv(OUTPUT_FILE, index=False)
