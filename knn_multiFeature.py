# Author: Amie Corso
# run KNN algorithm on March Madness Data
# TODO:
#  correct this for running actual tests on tournament data
# why are we seeing such high initial accuracy??
# experiment with removing additional features?

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
csvfile = "./reg_alldata.csv"
#csvfile = "./reg_alldata_short.csv"

k_vals = [1, 2, 5, 10, 20, 40]
df = pd.read_csv(csvfile)
print(df.head())


sets = [ [("train_1", ["2003"]), ("val_1", ["2004", "2005"]), ("test_1", ["2006", "2007"])],
         [("train_2", ["2003", "2004", "2005"]), ("val_2", ["2006", "2007"]), ("test_2", ["2008", "2009"])], 
         [("train_3", ["2003", "2004", "2005", "2006", "2007", "2008"]), ("val_3", ["2009", "2010"]), ("test_3", ["2011", "2012"])], 
         [("train_4", ["2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011"]), ("val_4", ["2012", "2013"]), ("test_4", ["2014", "2015"])],
         [("train_5", ["2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013"]), ("val_5", ["2014", "2015"]), ("test_5", ["2016", "2017"])]]


with open(outfile, "w") as out:
    out.write("Features: ['Season','TeamID', 'Score', 'FGM', 'FTM', 'DR', 'Ast', 'TO', 'Blk', 'PF', 'ScoredON'] \n")
    out.write("{: <30}{: <30}{: <30}{: <30}{: <30}".format("Train years", "Val years", "Test years", "Val Acc", "Test Acc") + '\n')

#for i in range(len(sets)):
for i in range(len(sets)):
    train_x = df.loc[df['Season'].isin(sets[i][0][1])]
    train_y = train_x["y"]
    train_x = train_x.drop(columns=["Season", "y"])
    train_x = train_x.as_matrix()
    train_y = train_y.as_matrix()

    val_x = df.loc[df['Season'].isin(sets[i][1][1])]
    val_y = val_x["y"]
    val_x = val_x.drop(columns=["Season", "y"])
    val_x = val_x.as_matrix()
    val_y = val_y.as_matrix()

    test_x = df.loc[df['Season'].isin(sets[i][2][1])]
    test_y = test_x["y"]
    test_x = test_x.drop(columns=["Season", "y"])
    test_x = test_x.as_matrix()
    test_y = test_y.as_matrix()

    for k in k_vals:
        knn = KNeighborsClassifier(n_neighbors = k)
        knn.fit(train_x, train_y)

        val_predict = knn.predict(val_x)
        val_acc = accuracy_score(val_y, val_predict)

        test_predict = knn.predict(test_x)
        test_acc = accuracy_score(test_y, test_predict)

        # WRITE REPORT TO FILE
        with open(outfile, "a") as out:
            out.write("K = {}\n".format(k))
            out.write("{: <30}{: <30}{: <30}{: <30.4f}{: <30.4f}".format(str(sets[i][0][1][0]) + " - " + str(sets[i][0][1][-1]), 
                                                                    str(sets[i][1][1][0]) + " - " + str(sets[i][1][1][-1]),
                                                                    str(sets[i][2][1][0]) + " - " + str(sets[i][2][1][-1]),
                                                                   val_acc, test_acc) + '\n')

