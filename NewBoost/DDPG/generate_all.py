hhids = [26,59,77,86,93,94,101,114,171,187]
scenarios = [(0.2,6.4,2),(0.2,13.5,5),(0.04,6.4,2),(0.04,13.5,5),(0.08,6.4,2),(0.08,13.5,5),(0.1,6.4,2),(0.1,13.5,5)]
#scenarios = [(0.2,6.4,2),(0.2,13.5,5),(0.04,6.4,2),(0.04,13.5,5)]


for hh in hhids:
    for sc in scenarios:
        #for i in range(0, 24*357, 72):
        f_suffix = str(hh)+"_"+str(int(sc[0]*100))+"_"+str(int(sc[1]*10))
        f_name = "process_"+f_suffix+".sh"
        
        ##################### Generate ##########################
        
        f = open(f_name, "w+")
        f.write("#!/bin/bash\n#SBATCH --gres=gpu:1        # request GPU \"generic resource\"\n#SBATCH --cpus-per-task=6  # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.\n#SBATCH --mem=32000M        # memory per node\n#SBATCH --time=00-14:00      # time (DD-HH:MM)\n#SBATCH --output=%N-%j.out  # %N for node name, %j for jobID\n#SBATCH --error=%N-%j.err   # %N for node name, %j for jobID\n#SBATCH --job-name={}\n\nmodule load python/3.6.3\nmodule load scipy-stack\nmodule load cuda\nvirtualenv --no-download ~/ENV\nsource ~/ENV/bin/activate\npip install torch --no-index\npip install tqdm\npip install --no-index tensorflow_gpu\n\npython main.py {} {} {} ../data/added_hhdata_{}_2.csv".format("DDPG"+f_suffix, sc[0], sc[1], sc[2], hh))
        f.close()
        
        ############################################################
                


