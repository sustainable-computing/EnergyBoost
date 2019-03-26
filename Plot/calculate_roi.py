import pandas as pd
import glob, os
import csv
import pathlib


# Read in csv file
os.chdir("roi_table_all/")
for file in glob.glob("*.csv"):
    #create csv file
    pathlib.Path("ROI_Calculation").mkdir(parents=True, exist_ok=True)
    csvfile = open('ROI_Calculation/{}'.format(file),'w', newline='')
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(["No solar no incentive MPC", "No solar no incentive RBC", "No solar with incentive MPC", "No solar with incentive RBC", "no battery MPC", "no battery RBC"])
    print("=======================================")
    print("Starting for file---",file)
    panel_number=file.split(".")[0][5]
    print("Panel number is: ", panel_number)
    sb=file.split("b")[1]
    print("Sell pack price: ", sb)
    battery_size=file.split("b")[2].split(".")[0]
    print("Battery size is: ", battery_size)

    # Calulate the total cost
    if panel_number == 2:
        panel_cost = 16000
        incentive = 6250
    else:
        panel_cost = 30000
        incentive = 11500

    if battery_size == 64:
        battery_cost = 6500
    else:
        battery_cost = 9000

    inverter_cost = 65
    Y = 20 #year
    i=0.02

    nosolar_incentive_cost = panel_cost + battery_cost + inverter_cost - incentive
    nosolar_noincentive_cost = panel_cost + battery_cost + inverter_cost
    nobattery_cost = panel_cost + inverter_cost


    #Read in the bills
    table = pd.read_csv(file)
    nsp=table['no_solar'][0:68].values
    nb_fg=table['NoB FG'][0:68].values
    nb_tg=table['NoB TG'][0:68].values  # must be multiplied by -1
    rbc_fg=table['RBC FG'][0:68].values
    rbc_tg=table['RBC TG'][0:68].values # must be multiplied by -1
    mpc_tg=table['MPC TG'][0:68].values # must be multiplied by -1
    mpc_fg=table['MPC FG'][0:68].values

    for k in range(68):
        r_mpc=-mpc_tg[k]
        r_rbc=-rbc_tg[k]
        r_nb=-nb_tg[k]
        rev_mpc=0
        rev_rbc=0
        rev_nb=0
        for j in range(1,Y+1):
            rev_mpc=rev_mpc+r_mpc/((1+i)**j)
            rev_rbc=rev_rbc+r_rbc/((1+i)**j)
            rev_nb=rev_nb+r_nb/((1+i)**j)

        sp_mpc=mpc_fg[k]*Y
        sp_rbc=rbc_fg[k]*Y
        sp_nb=nb_fg[k]*Y

        #Comparing with No solar without incentive cost
        roi_mpc = (nsp[k]*Y-sp_mpc+rev_mpc-nosolar_noincentive_cost)/nosolar_noincentive_cost
        roi_rbc = (nsp[k]*Y-sp_rbc+rev_rbc-nosolar_noincentive_cost)/nosolar_noincentive_cost

        #Comparing with No solar with incentive cost
        roi_mpc_incen = (nsp[k]*Y-sp_mpc+rev_mpc-nosolar_incentive_cost)/nosolar_incentive_cost
        roi_rbc_incen = (nsp[k]*Y-sp_rbc+rev_rbc-nosolar_incentive_cost)/nosolar_incentive_cost


        #Comparing with No nobattery_cost
        roi_mpc_nb=(sp_nb-rev_nb-sp_mpc+rev_mpc-nobattery_cost)/nobattery_cost
        roi_rbc_nb=(sp_nb-rev_nb-sp_rbc+rev_rbc-nobattery_cost)/nobattery_cost

        #write in
        writer.writerow([roi_mpc, roi_rbc, roi_mpc_incen, roi_rbc_incen, roi_mpc_nb, roi_rbc_nb])
    csvfile.close()
