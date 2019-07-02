# Author: Amie Corso
# run KNN algorithm on March Madness Data

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors # check that our data isn't dirty
from sklearn.metrics import accuracy_score
import csv

outfile = "./OUTPUT_Ryracc"
dictout = "./plotfiles/reg_yracc"
regdata = "./reg_alldata.csv"
tourneydata = "./tourney_alldata.csv"

KVAL = 40

dfR = pd.read_csv(regdata)
dfT = pd.read_csv(tourneydata)

years = ["2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017"]

with open(outfile, "w") as out:
    out.write("Features: ['Season','TeamID', 'Score', 'FGM', 'FTM', 'DR', 'Ast', 'TO', 'Blk', 'PF', 'ScoredON'] \n")
    out.write("{: <30}{: <30}".format("Test year", "Test Acc") + '\n')
results = {"years":years} 

accuracies = []

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

    test_x = dfT.loc[dfT['Season'].isin(test_years)]
    print("test_years = ", test_x.Season.unique())
    test_y = test_x["y"]
    test_x = test_x.drop(columns=["Season", "y"])
    test_x = test_x.as_matrix()
    test_y = test_y.as_matrix()
    
    '''
    # check whether points are their own neighbors!!!
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(train_x, train_y)
    neighs = nn.kneighbors(test_x)
    #print("NEIGHS", neighs)
    # it seems they are not
    '''

    knn = KNeighborsClassifier(n_neighbors = KVAL)
    knn.fit(train_x, train_y)

    test_predict = knn.predict(test_x)
    test_acc = accuracy_score(test_y, test_predict)
    accuracies.append(test_acc)

    # WRITE REPORT TO FILE
    with open(outfile, "a") as out:
        out.write("{: <30}{: <30.4f}".format(str(test_years[0]), test_acc) + '\n')
    
results["accuracies"] = accuracies

with open(dictout, "w") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["years", "accuracies"])
    writer.writeheader()
    writer.writerow(results)
