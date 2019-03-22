import numpy as np
# parameter
e = 4464
f = -0.1382
g = -1519
h = -0.4305
m = 5963
n = -0.6531
o = 321.4
p = 0.03168
q = 1471
s = 214.3
t = 0.6111
u = 0.3369
v = -2.295
# funtion of get fractional degradation
def get_fractional_degradation(bt):
    """
    Get get fractional degradation of battery

    :param bt: net energy to withdraw from the battery during interval t [Kwh]

    :return: None
    """



    NomIch = 0.125 # Nominal charge current
    NomId = 0.25  # Nominal discharge current
    NomSoC = 0.5 # Nominal state of charge_mode
    NomDoD = 1.0 # Nominal depth of discharge
    B = 5 #Battery capacity
    qt = 5 * 0.5 # Amount of energy in the battery at the start
    # Determin charge of discharge
    if bt > 0:
        Id = bt/(B*1) # time interval differnece is 1
        Ich = NomIch
    else:
        Ich = bt/(B*1)
        Id = NomId

    #Calculate average State of Charge
    SoC = 100 * (qt - 0.5*bt)/B

    #Calculate Depth of Discharge
    DoD = 100 * bt /B

    # Functions
    nCL1 = (e * np.exp (f * Id) + g * np.exp(h * Id))/ (e * np.exp (f * NomId) + g * np.exp(h * NomId))
    nCL2 = (m * np.exp (n * Ich) + o * np.exp(p * Ich))/ (m* np.exp (n* NomIch) + o * np.exp(p * NomIch))
    nCL3 = get_CL4(DoD, SoC)/get_CL4(NomDoD, NomSoC)
    nCL = nCL1 * nCL2 * nCL3
    Fractional_D = (0.5/3650)/ nCL
    return Fractional_D

def get_CL4(DoD, SoC):
    CL4 = q + ((u/(2*v)*(s + 100*u) - 200*t)*DoD + s * SoC + t * (DoD ** 2) + u * DoD * SoC + v * (SoC ** 2))
    return CL4



if __name__ == '__main__':
    blife= 30 # battery life 30 years
    action = 0.2
    degradation = get_fractional_degradation(action)
    print("This is degradation: ",degradation)
    update_blife = 30 * (1-degradation)
    print("Updated life is: ",update_blife)
