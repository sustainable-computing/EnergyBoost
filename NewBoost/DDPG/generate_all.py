hhids_old = [26,59,77,86,93,94,101,114,171,187,379, 1283, 2532, 1800, 499, 2365, 1192, 2980, 1697, 1790]
hhids = [2575, 2472, 1103, 2072, 86, 2755, 1507, 93, 370, 2018, 1642, 1403, 26, 59, 1086, 890, 1953, 1718, 1801, 503, 1463, 1500, 187, 94, 585, 2859, 2337, 484, 2953, 1169, 114, 624, 1632, 101, 252, 2401, 1792, 2378, 2557, 1415, 171, 2787, 974, 545, 871, 1700, 2814, 2965, 2986, 1202, 77, 2818, 2829, 2945, 1714, 946, 434, 744, 2233, 2925]
scenarios = [(0.2,6.4,2),(0.2,13.5,5),(0.04,6.4,2),(0.04,13.5,5),(0.08,6.4,2),(0.08,13.5,5),(0.1,6.4,2),(0.1,13.5,5)]
#scenarios = [(0.2,6.4,2),(0.2,13.5,5),(0.04,6.4,2),(0.04,13.5,5)]


for hh in hhids:
    if hh in hhids_old:
        continue
    for sc in scenarios:
        #for i in range(0, 24*357, 72):
        f_suffix = str(hh)+"_"+str(int(sc[0]*100))+"_"+str(int(sc[1]*10))
        f_name = "process_"+f_suffix+".sh"
        
        ##################### Generate ##########################
        
        f = open(f_name, "w+")
        f.write("#!/bin/bash\n#SBATCH --gres=gpu:1        # request GPU \"generic resource\"\n#SBATCH --cpus-per-task=6  # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.\n#SBATCH --mem=32000M        # memory per node\n#SBATCH --time=00-07:00      # time (DD-HH:MM)\n#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID\n#SBATCH --error=%N-%j.err   # %N for node name, %j for jobID\n#SBATCH --job-name={}\n\nmodule load python/3.6.3\nmodule load scipy-stack\nvirtualenv --no-download ~/ENV\nsource ~/ENV/bin/activate\npip install torch --no-index\npip install tqdm\n\npython main.py {} {} {} ../data/added_hhdata_{}_2.csv".format("DDPG"+f_suffix, sc[0], sc[1], sc[2], hh))
        f.close()
        
        ############################################################
                


