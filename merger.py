import glob
files = (glob.glob("results/merge/*csv"))
fin_res =""
for i in range(len(files)):
    each = files[i]
    with open(each, 'r') as f1:
        res = f1.read()
        if i:
            ind = res.find("Shape")+6
            res = res[ind:]
        fin_res+=res


with open('results/merge/merged.csv', 'w') as f2:
    f2.write(fin_res)
