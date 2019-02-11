import pandas as pd
import pvlib
import csv
import time as t
import glob
import os


class EnvironmentToAC:

    def __init__(self):
        self.angle_selector = ['apparent_', '']
        # Panel global position
        self.latitude = 30.2671500
        self.longitude = -97.7430600
        self.altitude = 165
        self.time_zone = 'US/Central'

        # self.latitude = 26.3059
        # self.longitude = -98.1716
        # self.altitude = 45.4
        # self.time_zone = 'Etc/GMT+6'
        self.select = 1
        # Panel parameters
        self.panel_tilt = 60
        self.panel_azimuth = 180
        self.panel_albedo = 0.2
        self.module_number = 10
        self.parallel_number = 4
        # Environment setup
        self.wind_speed = 10
        self.reference_irradiance = 1000
        self.sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
        self.sandia_module = self.sandia_modules['Canadian_Solar_CS5P_220M___2009_']
        self.sapm_inverters = pvlib.pvsystem.retrieve_sam('CECinverter')
        self.sapm_inverter = self.sapm_inverters['ABB__MICRO_0_25_I_OUTD_US_208_208V__CEC_2014_']
        self.sapm_model = {
                              'a': -3.56,
                              'b': -0.075,
                              'deltaT': 3.0
                          }

    def compute(self, time, fahrenheit_temperature, ghi):
        #time = pd.to_datetime(time[:-3]).tz_localize(self.time_zone,ambiguous=False)
        time = pd.to_datetime(time[:-3]).tz_localize(self.time_zone,ambiguous=True)
        # time = pd.to_datetime([time, time + pd.Timedelta(hours=1)]).tz_localize(self.time_zone)

        # Convert the fahrenheit to celsius
        temperature = fahrenheit_to_celsius(float(fahrenheit_temperature))

        ghi = max(0.0, float(ghi))

        # Determine site pressure from altitude.
        pressure = pvlib.atmosphere.alt2pres(self.altitude)

        # method = 'nrel_numpy' uses an implementation of the NREL SPA algorithm
        # Source from I. Reda and A. Andreas, Solar position algorithm for solar radiation applications.
        # Solar Energy, vol. 76, no. 5, pp. 577-589, 2004.
        # Return: apparent_elevation, apparent_zenith, azimuth, elevation, equation_of_time, zenith
        solar_position = pvlib.solarposition.get_solarposition(time=time,
                                                               latitude=self.latitude,
                                                               longitude=self.longitude,
                                                               altitude=self.altitude,
                                                               pressure=pressure,
                                                               method='nrel_numpy',
                                                               temperature=temperature)

        # direct normal irradiance extraterrestrial radiation
        dni_extra = pvlib.irradiance.extraradiation(datetime_or_doy=time,
                                                    solar_constant=1366.1,
                                                    method='spencer',
                                                    epoch_year=time.year)

        # Compute the airmass at sea-level on sun zenith angle
        # Fritz Kasten and Andrew Young. “Revised optical air mass tables and approximation formula”.
        # Applied Optics 28:4735-4738
        # Model comparisons: https://pvpmc.sandia.gov/PVLIB_Matlab_Help/html/pvl_relativeairmass_help.html
        airmass = pvlib.atmosphere.relativeairmass(zenith=solar_position[self.angle_selector[self.select] + 'zenith'],
                                                   model='kastenyoung1989')

        absolute_airmass = pvlib.atmosphere.absoluteairmass(airmass_relative=airmass,
                                                            pressure=pressure)

        # Calculates the angle of incidence of the solar vector on a surface.
        # This is the angle between the solar vector and the surface normal.
        # Assume modules tilted 37 degrees (approximately latitude tilt)
        aoi = pvlib.irradiance.aoi(surface_tilt=self.panel_tilt,
                                   surface_azimuth=self.panel_azimuth,
                                   solar_zenith=solar_position[self.angle_selector[self.select] + 'zenith'],
                                   solar_azimuth=solar_position['azimuth'])

        '''
        # Interpolates the monthly Linke turbidity values found in LinkeTurbidities.mat to daily values.
        # Return turbidity
        linke_turbidity = pvlib.clearsky.lookup_linke_turbidity(time=time,
                                                                latitude=self.latitude,
                                                                longitude=self.longitude,
                                                                filepath=None,
                                                                interp_turbidity=True)
        '''

        # Estimate Direct Normal Irradiance from Global Horizontal Irradiance using the DISC model.
        modeled_dni = pvlib.irradiance.disc(ghi=ghi,
                                            zenith=solar_position[self.angle_selector[self.select] + 'zenith'],
                                            datetime_or_doy=time,
                                            pressure=pressure)

        dni = modeled_dni['dni'][0]
        dhi = (ghi - pvlib.tools.cosd(angle=solar_position[self.angle_selector[self.select] + 'zenith']) * dni)[0]

        '''
        # Determine clear sky GHI, DNI, and DHI from Ineichen/Perez model.
        # P. Ineichen and R. Perez, “A New airmass independent formulation for the Linke turbidity coefficient”,
        # Solar Energy, vol 73, pp. 151-157, 2002.
        # R. Perez et. al., “A New Operational Model for Satellite-Derived Irradiances: Description and Validation”,
        # Solar Energy, vol 73, pp. 307-317, 2002.
        clear_sky = pvlib.clearsky.ineichen(apparent_zenith=solar_position['apparent_zenith'],
                                            airmass_absolute=absolute_airmass,
                                            linke_turbidity=linke_turbidity,
                                            altitude=self.altitude,
                                            dni_extra=dni_extra)
        '''

        # Determine diffuse irradiance from the sky on a tilted surface
        total_irradiance = pvlib.irradiance.total_irrad(surface_tilt=self.panel_tilt,
                                                        surface_azimuth=self.panel_azimuth,
                                                        apparent_zenith=solar_position[self.angle_selector[self.select]
                                                                                       + 'zenith'],
                                                        azimuth=solar_position['azimuth'],
                                                        dni=dni,
                                                        ghi=ghi,
                                                        dhi=dhi,
                                                        dni_extra=dni_extra,
                                                        airmass=airmass,
                                                        albedo=self.panel_albedo,
                                                        surface_type=None,
                                                        model='isotropic',
                                                        model_perez='allsitescomposite1990')

        # Estimate cell and module temperatures per the Sandia PV Array Performance Model (SAPM, SAND2004-3535)
        # King, D. et al, 2004, "Sandia Photovoltaic Array Performance Model", SAND2004-3535,
        # Sandia National Laboratories, Albuquerque, NM Web Link
        cell_module_temperature = pvlib.pvsystem.sapm_celltemp(poa_global=total_irradiance['poa_global'],
                                                               wind_speed=self.wind_speed,
                                                               temp_air=temperature,
                                                               model=self.sapm_model)

        # Calculates the SAPM effective irradiance using the SAPM spectral loss and
        # SAPM angle of incidence loss functions.
        effective_irradiance = pvlib.pvsystem.sapm_effective_irradiance(poa_direct=total_irradiance['poa_direct'],
                                                                        poa_diffuse=total_irradiance['poa_diffuse'],
                                                                        airmass_absolute=absolute_airmass,
                                                                        aoi=aoi,
                                                                        module=self.sandia_module,
                                                                        reference_irradiance=self.reference_irradiance)

        # The Sandia PV Array Performance Model (SAPM) generates 5 points on a PV module’s I-V curve
        # (Voc, Isc, Ix, Ixx, Vmp/Imp) according to SAND2004-3535.
        sapm = pvlib.pvsystem.sapm(effective_irradiance=effective_irradiance,
                                   temp_cell=cell_module_temperature['temp_cell'],
                                   module=self.sandia_module)

        # Converts DC power and voltage to AC power using Sandia’s Grid-Connected PV Inverter model.
        # SAND2007-5036, “Performance Model for Grid-Connected Photovoltaic Inverters
        # by D. King, S. Gonzalez, G. Galbraith, W. Boyson
        ac_power = pvlib.pvsystem.snlinverter(v_dc=sapm['v_mp'],
                                              p_dc=sapm['p_mp'],
                                              inverter=self.sapm_inverter)

        ac = max(0.0, ac_power[0]) * self.module_number * self.parallel_number / 1000
        dc = max(0.0, (sapm['v_mp'] * self.module_number *
                       sapm['i_mp'] * self.parallel_number)[0]) / 1000

        return [ac, dc]


def fahrenheit_to_celsius(fahrenheit):
    return (fahrenheit - 32) * 5 / 9


def get_time(second):
    hour = int(second // 3600)
    if hour > 99:
        hour = "99"
        minute = "99"
        second = "99"
    else:
        hour = str(hour)
        hour = ' ' * (2 - len(hour)) + hour
        minute = str(int((second // 60) % 60))
        minute = ' ' * (2 - len(minute)) + minute
        second = str(int(second % 60))
        second = ' ' * (2 - len(second)) + second

    if hour != ' 0':
        return_time = hour + 'h ' + minute + 'm ' + second + 's'
    elif minute != ' 0':
        return_time = minute + 'm ' + second + 's'
    else:
        return_time = second + 's'
    print(return_time)
    return return_time


def main():
    if not os.path.exists('data_ac_4'):
        os.makedirs('data_ac_4')
    start_time = t.time()
    model = EnvironmentToAC()
    for filename in glob.glob('data_by_home/processed*'):
        basename=filename.split("/")[1].split(".")[0]
        homenumber=basename.split("_")[2]
        # directory=('{}_{}'.format(homenumber,model.parallel_number))
        # if not os.path.exists(directory):
        #     os.makedirs(directory)
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            row_count = sum(1 for _ in reader)
        with open(filename, 'r') as csvfile:
            basename=filename.split("/")[1].split(".")[0]
            print('data_ac_4/{}_{}.csv'.format(basename,model.parallel_number))
            with open(('data_ac_4/{}_{}.csv'.format(basename,model.parallel_number)), 'w', newline='') as result:

                reader = csv.reader(csvfile, delimiter=',')
                writer = csv.writer(result, delimiter=',')
                row_number = 0

                for row in reader:

                    row_number += 1

                    if row_number == 1:
                        writer.writerow(row + ['AC', 'DC'])
                        continue

                    writer.writerow(row + model.compute(row[1], row[3], row[6]))

                    print("\rProgress: %5.2f%%" % (row_number * 100 / row_count), end='')

    print("\nProgram finished in " + get_time(t.time() - start_time))


def test():
    model = EnvironmentToAC()
    print(model.compute('2016/11/6 10:00:00-6', 48.85, 233.541))


if __name__ == "__main__":
    #test()
    main()
