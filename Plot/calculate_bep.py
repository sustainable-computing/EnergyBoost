import pandas as pd
import glob, os
import csv
import pathlib



# Read in csv file
os.chdir("roi_table_all/")
for file in glob.glob("*.csv"):
    #create csv file
    pathlib.Path("BEP_Calculation").mkdir(parents=True, exist_ok=True)
    csvfile = open('BEP_Calculation/{}'.format(file),'w', newline='')
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
        rev_mpc=-mpc_tg[k]
        rev_rbc=-rbc_tg[k]
        rev_nb=-nb_tg[k]
        sp_mpc=mpc_fg[k]
        sp_rbc=rbc_fg[k]
        sp_nb=nb_fg[k]

        #Comparing with No solar without incentive cost
        year_mpc = nosolar_noincentive_cost/(nsp[k] + rev_mpc*16/19 - sp_mpc)
        year_rbc = nosolar_noincentive_cost/(nsp[k] + rev_rbc*16/19 - sp_rbc)

        #Comparing with No solar with incentive cost
        year_mpc_incen = nosolar_incentive_cost/(nsp[k] + rev_mpc*16/19 - sp_mpc)
        year_rbc_incen = nosolar_incentive_cost/(nsp[k] + rev_rbc*16/19 - sp_rbc)


        #Comparing with No nobattery_cost
        year_mpc_nb = nobattery_cost/(sp_nb-rev_nb*16/19+rev_mpc*16/19-sp_mpc)
        year_rbc_nb = nobattery_cost/(sp_nb-rev_nb*16/19+rev_rbc*16/19-sp_rbc)

        #write in
        writer.writerow([year_mpc, year_rbc, year_mpc_incen, year_rbc_incen, year_mpc_nb, year_rbc_nb])
    csvfile.close()
