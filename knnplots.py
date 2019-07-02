# Author: Amie Corso
# March 2018

# Plots for KNN-model, March Madness Machine Learning Project


import matplotlib.pyplot as plt
#from knn_tourney_all import k_vals as k_vals1
import csv
import ast

def get_avgs(dic):
    sums = [0 for i in range( len(list(dic.values())[0]))]
    for key in dic:
        val_list = dic[key]
        for i in range(len(val_list)):
            sums[i] += val_list[i]
    for i in range(len(sums)):
        sums[i] = sums[i] / (len(dic.keys()))
    return sums

# Plot: T - K  v.  Accuracy
file1 = "./plotfiles/tourn_alld_allf"
with open(file1, "r") as f:
    dr = csv.DictReader(f)
    dd = {} # dictionary is the first and only item in dr
    years = dr.fieldnames[1:]
    for d in dr:
        kvals = ast.literal_eval(d["kvals"])
        for key in years:
            dd[key] = ast.literal_eval(d[key])

for year in years:
    plt.plot(kvals, dd[year], label=year)
plt.plot(kvals, get_avgs(dd), label="AVERAGE", linestyle="--", color="black", linewidth=3)

plt.xlabel("K")
plt.ylabel("Accuracy")
plt.suptitle("Tournament prediction accuracy by year\n Tournament training data")
plt.legend(labels=years + ["AVERAGE"], loc="upper left", bbox_to_anchor=(1.04,1))

plt.savefig("./figs/tourn_all_lgK", format="pdf", bbox_inches="tight")
plt.clf()

# Plot: T - K v. Accuracy (smaller K vals)
file2 = "./plotfiles/tourn_all_smallk"

with open(file2, "r") as f:
    dr = csv.DictReader(f)
    dd = {} # dictionary is the first and only item in dr
    years = dr.fieldnames[1:]
    for d in dr:
        kvals = ast.literal_eval(d["kvals"])
        for key in years:
            dd[key] = ast.literal_eval(d[key])

for year in years:
    plt.plot(kvals, dd[year], label=year)
plt.plot(kvals, get_avgs(dd), label="AVERAGE", linestyle="--", color="black", linewidth=3)

plt.xlabel("K")
plt.ylabel("Accuracy")
plt.suptitle("Tournament prediction accuracy by year\n Tournament training data")
plt.legend(labels=years + ["AVERAGE"], loc="upper left", bbox_to_anchor=(1.04,1))

plt.savefig("./figs/tourn_all_smK", format="pdf", bbox_inches="tight")
plt.clf()

# Plot: R - K v. Accuracy
file3 = "./plotfiles/reg_alld_allf"

with open(file3, "r") as f:
    dr = csv.DictReader(f)
    dd = {}
    years = dr.fieldnames[1:]
    for d in dr:
        kvals = ast.literal_eval(d["kvals"])
        for key in years:
            dd[key] = ast.literal_eval(d[key])

for year in years:
    plt.plot(kvals, dd[year], label=year)
plt.plot(kvals, get_avgs(dd), label="AVERAGE", linestyle="--", color="black", linewidth=3)

plt.xlabel("K")
plt.ylabel("Accuracy")
plt.suptitle("Tournament prediction accuracy by year\n Regular training data")
plt.legend(labels=years + ["AVERAGE"], loc="upper left", bbox_to_anchor=(1.04,1))

plt.savefig("./figs/reg_all_varyK", format="pdf", bbox_inches='tight')
plt.clf()

# plot: R - K v. Accuracy (Small K) 
file3 = "./plotfiles/reg_all_smK"

with open(file3, "r") as f:
    dr = csv.DictReader(f)
    dd = {}
    years = dr.fieldnames[1:]
    for d in dr:
        kvals = ast.literal_eval(d["kvals"])
        for key in years:
            dd[key] = ast.literal_eval(d[key])

for year in years:
    plt.plot(kvals, dd[year], label=year)
plt.plot(kvals, get_avgs(dd), label="AVERAGE", linestyle="--", color="black", linewidth=3)


plt.xlabel("K")
plt.ylabel("Accuracy")
plt.suptitle("Tournament prediction accuracy by year\n Regular training data")
plt.legend(labels=years + ["AVERAGE"] , loc="upper left", bbox_to_anchor=(1.04,1))

plt.savefig("./figs/reg_all_smK", format="pdf", bbox_inches='tight')
plt.clf()



# Plot: T and R - Accuracy by year
file4 = "./plotfiles/tourn_yracc"

file6 = "./plotfiles/reg_yracc"

with open(file4, "r") as f:
    dr = csv.DictReader(f)
    for d in dr:
        years = ast.literal_eval(d["years"])
        accsT = ast.literal_eval(d["accuracies"])
with open(file6, "r") as f:
    dr = csv.DictReader(f)
    for d in dr:
        accsR = ast.literal_eval(d["accuracies"])


plt.plot(years, accsT, label="Tourney trained")
plt.plot(years, accsR, label="Reg-season trained")
plt.legend(labels=["Tourney trained", "Reg-season trained"] , loc="upper left", bbox_to_anchor=(1.04,1))

plt.xlabel("Year")
plt.ylabel("Test Accuracy")
plt.suptitle("Tournament prediction accuracy by year")

plt.savefig("./figs/yracc", format="pdf", bbox_inches='tight')
plt.clf()


# plot: R - vary Features 
file7 = "./plotfiles/reg_varyFeat"

with open(file7, "r") as f:
    dr = csv.DictReader(f)
    dd = {}
    years = dr.fieldnames[1:]
    for d in dr:
        numfeats = ast.literal_eval(d["numfeats"])
        for key in years:
            dd[key] = ast.literal_eval(d[key])

for year in years:
    plt.plot(numfeats, dd[year], label=year)
plt.plot(numfeats, get_avgs(dd), label="AVERAGE", linestyle="--", color="black", linewidth=3)


plt.xlabel("Num Features")
plt.ylabel("Accuracy")
plt.suptitle("Tournament prediction accuracy by year\n Regular training data")
plt.legend(labels=years + ["AVERAGE"] , loc="upper left", bbox_to_anchor=(1.04,1))

plt.savefig("./figs/reg_varyFeat", format="pdf", bbox_inches='tight')
plt.clf()
