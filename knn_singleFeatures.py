# Read and process March Madness data for consumption by SciKit-Learn

'''
NOTES:
   Most useful categories from adeshpande's model: (in order most to least)
   wins
   strength of schedule
   location
   SRS simple rating system (??)
   SPG average steals per game
   APG average assists per game
   TOP average turnovers per game
   RPG average rebounds per game
   3PG average 3's per game
   Tourney appearances
   PPG points per game scored
   PPGA points per game allowed
   Powerconf
   
  
KAGGLE COLUMNS - detailed results file
Season, DayNum, WTeamID, WScore, LTeamID, LScore, WLoc, and NumOT
WFGM - field goals made (by the winning team)
WFGA - field goals attempted (by the winning team)
WFGM3 - three pointers made (by the winning team)
WFGA3 - three pointers attempted (by the winning team)
WFTM - free throws made (by the winning team)
WFTA - free throws attempted (by the winning team)
WOR - offensive rebounds (pulled by the winning team)
WDR - defensive rebounds (pulled by the winning team)
WAst - assists (by the winning team)
WTO - turnovers committed (by the winning team)
WStl - steals (accomplished by the winning team)
WBlk - blocks (accomplished by the winning team)
WPF - personal fouls committed (by the winning team) 

additional features?
- total average score per game
- total average score ALLOWED

Feature                       Val Acc                       Test Acc                      
Score                         0.6205997392438071            0.5992585727525487            
FGM                           0.6010430247718384            0.5990732159406859            
ScoredON                      0.5947103743713913            0.585356811862836             
Ast                           0.5773887129819333            0.559406858202039             
DR                            0.5587632706276774            0.5670064874884152            
TO                            0.5460979698267834            0.574050046339203             
Blk                           0.5341776867200596            0.5338276181649676            
FTM                           0.530638852672751             0.5445783132530121            
PF                            0.5278450363196125            0.5243744207599629            
FTA                           0.5276587818960701            0.5154772937905469            
FGA                           0.5200223505308251            0.4904541241890639            
Stl                           0.516297262059974             0.5177015755329009            
FGM3                          0.5092195939653567            0.5151065801668211            
OR                            0.5067982864593034            0.5082483781278962            
FGA3                          0.49916185509405847           0.518628359592215             
'''
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import dataprocess

outfile = "knn_output"
filename = "./DataFiles_mens/RegularSeasonDetailedResults.csv"
#filename = "./DataFiles_mens/short_RegularSeasonDetailedResults.csv"

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

with open(outfile, "w") as out:
    out.write("{: <30}{: <30}{: <30}".format("Feature", "Val Acc", "Test Acc") + '\n')

# TRY WITH ONLY ONE FEATURE
for column in gen_col[2:]:
    wl_df_short = wl_df[['Season', 'TeamID', column]]
    #wl_df = wl_df[gen_col[:3]]
#    print(wl_df.head())

    grouped = wl_df_short.groupby(["Season", "TeamID"], as_index = False)
    avg_stats = grouped.mean()
    avg_stats = pd.DataFrame(avg_stats) # convert back into a dataframe
    print(avg_stats.head())

    # GENERATE TRAINING DATA from GAME data

    train_x, train_y, validate_x, validate_y, test_x, test_y = dataprocess.getdataPoints_binary(avg_stats, df)

    # NORMALIZE
    train_x = np.array(train_x)
    validate_x = np.array(validate_x)
    test_x = np.array(test_x)

    train_x = train_x / train_x.max(axis=0)
    validate_x = validate_x / validate_x.max(axis=0)
    test_x = test_x / test_x.max(axis=0)
    # back to lists
    train_x = train_x.tolist()
    validate_x = validate_x.tolist()
    test_x = test_x.tolist()

#    print("TRAIN DATA")
#    print(train_x[:10])

    # create and fit nearest-neighbor classifier
    knn = KNeighborsClassifier()
    knn.fit(train_x, train_y)

    val_predict = knn.predict(validate_x)
    val_acc = accuracy_score(validate_y, val_predict)

    test_predict = knn.predict(test_x)
    test_acc = accuracy_score(test_y, test_predict)
    
    with open(outfile, "a") as out:
        out.write("{: <30}{: <30}{: <30}".format(column, val_acc, test_acc) + '\n')
