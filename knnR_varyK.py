# Author: Amie Corso
# run KNN algorithm on March Madness Data

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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import csv

outfile = "OUTPUT_reg"
dictout = "./plotfiles/reg_alld_allf"
regdata = "./reg_alldata.csv"
tourneydata = "./tourney_alldata.csv"

k_vals = [1, 2, 4, 8, 16, 32, 60, 100, 200, 400, 800, 1000, 2000, 4000, 8000, 16000, 32000, 40000, 50000, 60000, 70000, 75000, 76000, 76636]

dfR = pd.read_csv(regdata)
dfT = pd.read_csv(tourneydata)
#print(dfR.head())

years = ["2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017"]

with open(outfile, "w") as out:
    out.write("Features: ['Season','TeamID', 'Score', 'FGM', 'FTM', 'DR', 'Ast', 'TO', 'Blk', 'PF', 'ScoredON'] \n")
    out.write("Train years: all seasons excluding validate and test\n")
    out.write("{: <15}{: <30}{: <30}{: <30}{: <30}".format("K", "Val years", "Test years", "Val Acc", "Test Acc") + '\n')

results = {"kvals":k_vals} # store {year : [accuracies]} results for plotting
for i in range(len(years)): # we're going to test on each year
    test_years = [years[i]]
    val_years = [years[i - 1]]
    train_years = years # with regular season data, we can always use all the years

    train_x = dfR.loc[dfR['Season'].isin(train_years)]
    print("train_years = ", train_x.Season.unique())
    train_y = train_x["y"]
    train_x = train_x.drop(columns=["Season", "y"])
    train_x = train_x.as_matrix()
    train_y = train_y.as_matrix()

    val_x = dfT.loc[dfT['Season'].isin(val_years)]
    print("val_years = ", val_x.Season.unique())
    val_y = val_x["y"]
    val_x = val_x.drop(columns=["Season", "y"])
    val_x = val_x.as_matrix()
    val_y = val_y.as_matrix()

    test_x = dfT.loc[dfT['Season'].isin(test_years)]
    print("test_years = ", test_x.Season.unique())
    test_y = test_x["y"]
    test_x = test_x.drop(columns=["Season", "y"])
    test_x = test_x.as_matrix()
    test_y = test_y.as_matrix()
    print("NUM TRAINING POINTS: ", len(train_x))
    print("NUM VAL POINTS: ", len(val_x))
    print("NUM TEST POINTS: ", len(test_x))

    accuracies = []
    for k in k_vals:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(train_x, train_y)

        val_predict = knn.predict(val_x)
        val_acc = accuracy_score(val_y, val_predict)

        test_predict = knn.predict(test_x)
        test_acc = accuracy_score(test_y, test_predict)
        
        accuracies.append(test_acc) # collect accuracies for each value of k for this year
        # WRITE REPORT TO FILE
        with open(outfile, "a") as out:
            out.write("{: <15}{: <30}{: <30}{: <30.4f}{: <30.4f}".format(k, str(val_years[0]), str(test_years[0]), val_acc, test_acc) + '\n')
    results[years[i]] = accuracies # create dictionary entry

years.insert(0, "kvals") 
# write dictionary to file
with open(dictout, "w") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=years)
    writer.writeheader()
    writer.writerow(results)


