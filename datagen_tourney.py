# Author: Amie Corso
# March 2018
# Read and process March Madness data for consumption by SciKit-Learn - TOURNAMENT
# Write data to csv file

import numpy as np
import pandas as pd
import random
import csv
from sklearn import preprocessing

OUTPUT_FILE = "./tourney_alldata.csv"
            
#gen_avgs = "./DataFiles_mens/NCAATourneyDetailedResults.csv"
gen_avgs = "./DataFiles_mens/RegularSeasonDetailedResults.csv"
gen_gamedata = "./DataFiles_mens/NCAATourneyDetailedResults.csv"


def getdataPoints_binary(avgTable, allData):
    ''' Creates vectors for each game.  Returns list of lists for conversion later back to dataframe. '''
    master = []
    #print(avgTable.to_string())
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
#        print(num)
        if num <= .50:
            ans = lAvg.values-wAvg.values
            #lol janky
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


# READ IN CSV DATA - still generating Reg-season averages!!! even though training on tourney GAMES
df = pd.read_csv(gen_avgs, encoding="latin-1", low_memory=False)

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
#select_feat = ['Season','TeamID', 'Score', 'FGM', 'FTM', 'DR', 'Ast', 'TO', 'Blk', 'PF', "ScoredON"]
select_feat = ['Season','TeamID', 'Score', 'FGM', 'PF', "ScoredON"]
wl_df_short = wl_df[select_feat]
#wl_df_short = wl_df
#select_feat = gen_col

grouped = wl_df_short.groupby(["Season", "TeamID"], as_index = False)
avg_stats = grouped.mean()
avg_stats = pd.DataFrame(avg_stats) # convert back into a dataframe
print(avg_stats.head()) # check it out

# Generate the  W/L ratio feature
WLSeries = []
for index, row in avg_stats.iterrows():
    wincount = df.loc[(df["Season"] == row["Season"])].loc[(df["WTeamID"] == row["TeamID"])].WTeamID.count()
    losscount = df.loc[(df["Season"] == row["Season"])].loc[(df["LTeamID"] == row["TeamID"])].LTeamID.count()
    if losscount > 0:
        WLSeries.append(wincount / losscount)
    else:
        WLSeries.append(wincount)

# add the new column to our dataframe
avg_stats["WLRatio"] = pd.Series(WLSeries, index=avg_stats.index)
select_feat.append("WLRatio") # update our list for future use!

# GENERATE TRAINING DATA from GAME data
df = pd.read_csv(gen_gamedata, encoding="latin-1", low_memory=False) # for each TOURNAMENT game
master = getdataPoints_binary(avg_stats, df)

columns = ["Season"] + select_feat[2:] + ["y"]
print("DATAFRAME MASTER")
master = pd.DataFrame(master, columns = columns)
print(master.head())

# NORMALIZE data
x_data = master[select_feat[2:]]

minmaxscaler = preprocessing.MinMaxScaler()
x_data = minmaxscaler.fit_transform(x_data)

master[select_feat[2:]] = x_data

# HAVE NORMALIZED DATAFRAME with SEASON as first column, y as last column
# split data into relevant sets to run KNN

# WRITE TO CSV FILE
master.to_csv(OUTPUT_FILE, index=False)

