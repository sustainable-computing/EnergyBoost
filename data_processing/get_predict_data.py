import csv,os,sys
from sklearn.externals import joblib

state=[]
homeid= sys.argv[1].split("_")[4]
clf_hl = joblib.load('saved_models/hl_rf_{}.pkl'.format(homeid))
clf_ac = joblib.load('saved_models/ghi_rf_{}.pkl'.format(homeid))
def init_ground_truth(datafile):
    print("init_ground_truth")


    if not os.path.exists(datafile):
        print("No datafile was found. Run generatepower.py first.")
        raise ValueError

    with open(datafile, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        row_count = sum(1 for _ in reader)

    #with open("processed_hhdata_26_result.csv", 'r') as csvfile:
    with open(datafile, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        row_number = 0

        for row in reader:

            row_number += 1

            if row_number == 1:
                continue
            #print(row)

            state.append(row)

            print("\rEnvironment setup progress: %5.2f%%" % (row_number * 100 / row_count), end='')

    print("\rEnvironment setup finished. Total %i lines data." % row_count)


#localhour	use	temperature	cloud_cover	wind_speed	GH	is_weekday	month	hour	use_hour	use_week	ac	ac_hour	ac_week
#  0         1        2         3           4       5      6          7      8         9           10       11    12      13

def predict_day(start):
    global state
    for i in range(24):
        global_index = start+i
        #print("write the line", global_index)
        if i==0:
            use = state[global_index][1]
            ac = state[global_index][11]
        else:
            #predict based on previous data
            use = clf_hl.predict([[state[global_index][5], state[global_index-1][1], state[global_index][10], state[global_index][2], state[global_index][3], state[global_index][4], state[global_index][6], state[global_index][7], state[global_index][8]]])[0]
            ac = clf_ac.predict([[state[global_index][1], state[global_index][2], state[global_index][3], state[global_index][4], state[global_index][6], state[global_index-1][11], state[global_index][13], state[global_index][7], state[global_index][8]]])[0]
            #update
            state[global_index][1] = use
            state[global_index][11] = ac
        writer.writerow([state[global_index][0],use,state[global_index][2],state[global_index][3],state[global_index][4],state[global_index][5],state[global_index][6],state[global_index][7],state[global_index][8],state[global_index][9],state[global_index][10],ac,state[global_index][12],state[global_index][13]])






init_ground_truth(sys.argv[1])
#print(state[2][1])
directory="data_predicted4"
if not os.path.exists(directory):
    os.makedirs(directory)
csvfile = open("data_predicted2/predicted_hhdata_{}_2.csv".format(homeid), 'w', newline='')
writer = csv.writer(csvfile, delimiter=',')
start_point=0
end_point=8616
writer.writerow(["localhour", "use", "temperature",	"cloud_cover", "wind_speed", "GH", "is_weekday", "month", "hour", "use_hour", "use_week", "ac", "ac_hour", "ac_week"])
for i in range (start_point,end_point,24):
    predict_day(i)
#print(state[2][1])
print("================FINISH================")
