import glob, os
hhids=[]
for file in glob.glob("/home/baihong/Documents/sh/data_predicted2/*"):
    print(file)
    id=file.split("_")[3]
    print(id)
    hhids.append(id)
# hhids.sort()
print(hhids)
# Alist=[]
# for i in range(0,70,10):
#     Alist.append(hhids[i])
# print(Alist)
