import pandas as pd
import numpy as np
from os.path import isfile, join
import os
import matplotlib.pyplot as plt
import math
import scipy

class trPL_measurement_series:
    def __init__(self, TRPL_folderpath, BG, importPL = False, importSPV = False, thickness = None, alpha = None, TRPL_denoise = False, retime = True, mode = "HySprint", Nc = 2e18, Nv = 2e18, kT = 27.7*1e-3,  lambda_laser = 705e-9, spot_diameter =  2.72e-04, BD_ratio = 0.21, which = None):
       
        self.lambda_laser = lambda_laser
        print("Lambda Laser set to: {:2e}".format(self.lambda_laser))
        self.spot_diameter = spot_diameter #m (measured)
        print("Spot diameter set to: {:2e}".format(self.spot_diameter))
        self.spot_area = math.pi*(spot_diameter/2)**2 #m2
        #Film params
        self.thickness = thickness
        print("Film thickness set to: {:2e}".format(self.thickness))
        self.TRPL_folderpath = TRPL_folderpath

        ## TODO: Generation profile calculation ##
        #if not(thickness == None):  
        #    self.thickness = thickness
        #elif not(alpha == None):
         #   self.
        
        if (mode == "HySprint" or mode == "auto"):
            self.BD_ratio = BD_ratio
        else:
            self.BD_ratio = 1.0
        print("Beam dump ratio set to: {:2e}".format(self.BD_ratio))

        #Physics
        self.BG = BG
        self.Nc = Nc #cm-3
        self.Nv = Nv #cm-3
        print("BG, Nc and Nv set to:"+str(BG)+" "+ str(Nc)+ " "+ str(Nv))
        self.kT = kT
        self.hc = 1.98645E-25
        self.ni = np.sqrt(self.Nc*self.Nv*np.exp(-self.BG/(self.kT)))
        
        self.mode = mode

        self.TRPL_denoise = TRPL_denoise
        self.retime = retime 
        
        self.which = which

        self.Q2_colors = ['#ffc67c', '#f98b2c', '#F05D23', '#c71d06', '#942a00', 'black'] 
        self.S_colors = ['#33d1a3', '#04d288', '#098e68', '#046a55', '#054639', 'black']
        self.B2_colors = ['#abc6e5', '#81ADC8', '#62769c', '#465970', '#303d4d', 'black']  

        if (importPL):
            self.TRPL_reprates_Hz = []
            self.TRPL_powers = []
            self.TRPL_integration_times_seconds = []
            self.TRPLs_files, self.TRPLs_ts, self.TRPLs_n, self.TRPLs_subsMean, self.TRPLs_raw, self.TRPLs_noise = self.TRPL_folder_read()
            self.calculate_N0s()
    
        if (importSPV):
            self.SPV_reprates_Hz = []
            self.SPVs_files, self.SPVs= self.SPV_folder_read()
        

    def calculate_N0s(self):
        photon_energy = self.hc / self.lambda_laser #J
        n0s = []
        fluences = []
        for p, rep in zip(self.TRPL_powers, self.TRPL_reprates_Hz):
            power_per_pulse = p*self.BD_ratio/rep #J
            PowerDensity_per_pulse = power_per_pulse/self.spot_area
            photons_per_pulse = PowerDensity_per_pulse/photon_energy #m-2
            fluences.append(PowerDensity_per_pulse)
            pump_carrierDensity = photons_per_pulse/self.thickness #m-3
            pump_carrierDensity_cm = 1e-6*pump_carrierDensity #cm-3 (includes the beamdump ratio)
            n0s.append(pump_carrierDensity_cm)

        self.n0s = np.array(n0s)
        self.fluences = np.array(fluences)

        return None
    
    # def calculateGs(self, alpha, photon_flux):
    #     if (alpha == None):
    #         alpha = 1e7 # 1e5 cm-1 expressed in m-1
    #     self.alpha = alpha

    #     z = 1e-9*np.arange(1500) #m
    #     G = alpha*photon_flux*np.exp(-alpha*z) #m-3.s-1

    #     df = self.PLpowers
    #     G_total = np.empty((len(self.TRPLs_ts),))

    #     for i, t in enumerate((self.TRPLs_ts)):
    #         G_total[i] = np.trapezoid(G[z<=t], x = z[z<=t])

    #     self.Gs = G_total

    def TRPL_readout_function(self, filepath, integration_time_seconds, reprate, TRPL_denoise = False, retime = False, mode = "HySprint"):
        """
        Imports a TRPL data file.  

        Parameters
        ----------
        filepath : Path to the TRPL data file. 
        
        integration_time_seconds : Integration time of the measurement. 
        
        reprate: repetition rate of all the experiments. 
        
        TRPL_denoise: TRPL_denoise values? True: Automatic denoising (careful, check the procedure). False: No denoising. 
                                Value: The value is removed from the counts.
                                
        
        retime: shift time axis so that max(counts) is at t = 0.
        
        
        Returns
        -------
        time vector, TRPL normalised vector, TRPL (denoised) vector, TRPL raw data vector, Noise level
        
        """

        if(mode == "HySprint"):
            df = pd.read_csv(filepath, skiprows=8, header = None, encoding='latin-1', delimiter = '\t')
            binsize_seconds = 1e-9*df.iloc[0].astype('float64')[0]
            self.TRPL_binsize.append(binsize_seconds)
            TRPL = df.iloc[2:,0].to_numpy(dtype = None).astype('int')
        elif(mode == "auto"):
            df = pd.read_csv(filepath, delimiter = '\t')
            df = df.drop(df.columns[0:1], axis=1)
            t = 1e-12*df["bins [ps]"].to_numpy(dtype = None).astype('int')
            binsize_seconds = t[1]-t[0]
            self.TRPL_binsize.append(binsize_seconds)
            TRPL = df["counts [#]"].to_numpy(dtype = None).astype('int')
            t = np.transpose(t)
        elif(mode == "wannsee"):
            df = pd.read_csv(filepath, skiprows=11, header = None, on_bad_lines='skip')
            binsize_seconds = 1e-12*(df.iloc[1, 0].astype('float64') - df.iloc[0, 0].astype('float64'))
            self.TRPL_binsize.append(binsize_seconds)
            TRPL = df.iloc[0:,1].to_numpy(dtype = None).astype('int')
            t = 1e-12*(df.iloc[0:,0].to_numpy(dtype = None).astype('float64'))

        TRPL = np.transpose(TRPL)
        #Remove Uniform noise
        if(not(TRPL_denoise)):
            noise = 0
            #noise = np.mean(TRPL[0][(np.argmax(TRPL[0][:])-noise_start_i):(np.argmax(TRPL[0][:])-noise_end_i)])
        elif(TRPL_denoise < 0):  
            noise = np.mean(np.trim_zeros(TRPL, trim='b')[TRPL_denoise:]) #takes the zeros out of the end of the data
            #noise = np.mean(TRPL[TRPL_denoise:])
        elif(TRPL_denoise > 0):
            noise = np.mean(TRPL[:TRPL_denoise])
            
        TRPL_denoise = TRPL[:] - noise
        TRPL_denoise = self.rate_calculation_function(TRPL_denoise, integration_time_seconds, binsize_seconds, reprate)
        #TRPL = rate_calculation_function(TRPL, integration_time_seconds, binsize, reprate)
            
        #Normalize
        TRPL_n = TRPL_denoise/np.amax(TRPL_denoise)
        
        if (retime):
            if(mode == "HySprint"):
                t_TRPL = binsize_seconds*(np.arange(len(TRPL_n))-np.argmax(TRPL_n))
            elif(mode == "auto" or mode == "wannsee"):
                t_TRPL = t-t[np.argmax(TRPL)]
        else:
            if(mode == "HySprint"):
                t_TRPL = binsize_seconds*(np.arange(len(TRPL_n)))
            else:
                t_TRPL = t

        return t_TRPL, TRPL_n, TRPL_denoise, TRPL, noise

    def TRPL_folder_read(self):
        """
        Imports a folder of TRPL files.  

        Parameters
        ----------
        folderpath : Path to the folder containing data. 
        
        integration_time_seconds : Integration time of the measurement. 
        
        reprate_Hz: repetition rate of all the experiments. 
        
        TRPL_denoise: TRPL_denoise values? True: Automatic denoising (careful, check the procedure). False: No denoising. 
                                Value: The value is removed from the counts.
                                
        
        retime: shift time axis so that max(counts) is at t = 0.
        
        
        Returns
        -------
        file names, time array, TRPL normalised array, TRPL (denoised), TRPL raw data, Noise levels
        
        """
        
        if(self.mode == "wannsee"):
            files = [f for f in os.listdir(self.TRPL_folderpath) if (isfile(join(self.TRPL_folderpath, f)) and f.endswith(".csv") and (not(f.startswith("._"))) and not(f.endswith(".Wfm.csv")))]
            files.sort()
        else:
            files = [f for f in os.listdir(self.TRPL_folderpath) if (isfile(join(self.TRPL_folderpath, f)) and f.endswith(".dat") and (not(f.startswith("._"))))]
            files.sort()

        print(files)
        
        if ((self.TRPL_reprates_Hz == []) or (self.TRPL_integration_times_seconds == []) or (self.TRPL_powers == [])):
            self.TRPL_sample = []
            self.TRPL_NDs = []
            self.TRPL_cps = []
            self.TRPL_PLs = []
            self.TRPL_measNum = []
            self.TRPL_binsize = []
        else:
            print("Manual input of params")


        fpar = os.path.splitext(files[0])[0]
        meas_ps = fpar.split("_")
        if(self.which == "LPtrPL"):
            self.TRPL_sample.append(meas_ps[0])
            self.TRPL_reprates_Hz.append(float(1e3*float(meas_ps[1][:-3])))
            self.TRPL_NDs.append(meas_ps[3])
            self.TRPL_cps.append(float(meas_ps[5][:-3]))
            self.TRPL_integration_times_seconds.append(float(meas_ps[1][:-1]))
            self.TRPL_PLs.append(1e-6*float(meas_ps[4][3:-2]))
        else:
            if(self.mode == "HySprint"):
                self.TRPL_sample.append(meas_ps[0])
                meas_ps = meas_ps[1].split("-")
                self.TRPL_reprates_Hz.append(float(1e3*float(meas_ps[0][:-3])))
                self.TRPL_NDs.append(meas_ps[1])
                self.TRPL_powers.append(float(1e-6*float(meas_ps[2][:-2])))
                self.TRPL_integration_times_seconds.append(float(meas_ps[3][:-1]))
            elif(self.mode == "wannsee"):
                meas_ps = meas_ps[0].split("-")
                # print(meas_ps)
                self.TRPL_measNum.append(meas_ps[0])
                self.TRPL_sample.append(meas_ps[2])
                self.TRPL_reprates_Hz.append(float(1e3*float(meas_ps[-1][:-3])))
                self.TRPL_NDs.append(meas_ps[-2])
                self.TRPL_powers.append(1)
                self.TRPL_integration_times_seconds.append(float(meas_ps[3][:-1]))
            else:
                self.TRPL_sample.append(meas_ps[0])
                self.TRPL_reprates_Hz.append(float(1e3*float(meas_ps[2][:-3])))
                self.TRPL_NDs.append(meas_ps[3])
                self.TRPL_cps.append(float(meas_ps[4][:-3]))
                self.TRPL_powers.append(float(1e-6*float(meas_ps[5][:-2])))
                self.TRPL_integration_times_seconds.append(float(meas_ps[1][:-1]))
        
        ts, TRPLs_n, TRPL_subsMean, TRPL_raw, noise = self.TRPL_readout_function(join(self.TRPL_folderpath, files[0]), self.TRPL_integration_times_seconds[0], self.TRPL_reprates_Hz[0], self.TRPL_denoise, self.retime, self.mode)
        
        Noise = [noise]
        
        for i, f in enumerate(files):
            if(i == 0):
                continue
            fpar = os.path.splitext(f)[0]
            meas_ps = fpar.split("_")
            if(self.which == "LPtrPL"):
                self.TRPL_sample.append(meas_ps[0])
                self.TRPL_reprates_Hz.append(float(1e3*float(meas_ps[1][:-3])))
                self.TRPL_NDs.append(meas_ps[3])
                self.TRPL_cps.append(float(meas_ps[5][:-3]))
                self.TRPL_integration_times_seconds.append(float(meas_ps[1][:-1]))
                self.TRPL_PLs.append(1e-6*float(meas_ps[4][3:-2]))
            else:
                if(self.mode == "HySprint"):
                    self.TRPL_sample.append(meas_ps[0])
                    meas_ps = meas_ps[1].split("-")
                    self.TRPL_reprates_Hz.append(float(1e3*float(meas_ps[0][:-3])))
                    self.TRPL_NDs.append(meas_ps[1])
                    self.TRPL_powers.append(float(1e-6*float(meas_ps[2][:-2])))
                    self.TRPL_integration_times_seconds.append(float(meas_ps[3][:-1]))
                elif(self.mode == "wannsee"):
                    meas_ps = meas_ps[0].split("-")
                    # print(meas_ps)
                    self.TRPL_measNum.append(meas_ps[0])
                    self.TRPL_sample.append(meas_ps[2])
                    self.TRPL_reprates_Hz.append(float(1e3*float(meas_ps[-1][:-3])))
                    self.TRPL_NDs.append(meas_ps[-2])
                    self.TRPL_powers.append(1)
                    self.TRPL_integration_times_seconds.append(float(meas_ps[3][:-1]))
                else:
                    self.TRPL_sample.append(meas_ps[0])
                    self.TRPL_reprates_Hz.append(float(1e3*float(meas_ps[2][:-3])))
                    self.TRPL_NDs.append(meas_ps[3])
                    self.TRPL_cps.append(float(meas_ps[4][:-3]))
                    self.TRPL_powers.append(float(1e-6*float(meas_ps[5][:-2])))
                    self.TRPL_integration_times_seconds.append(float(meas_ps[1][:-1]))
        
            t, trpl_n, trpl_subsMean, trpl_raw, noise = self.TRPL_readout_function(join(self.TRPL_folderpath, f), self.TRPL_integration_times_seconds[i], self.TRPL_reprates_Hz[i], self.TRPL_denoise, self.retime, self.mode)
            
            ts = np.c_[ts, t]
            TRPLs_n = np.c_[TRPLs_n, trpl_n]
            TRPL_subsMean = np.c_[TRPL_subsMean, trpl_subsMean]
            TRPL_raw = np.c_[TRPL_raw, trpl_raw]
            Noise.append(noise)
       
        return files, ts, TRPLs_n, TRPL_subsMean, TRPL_raw, Noise

    def rate_calculation_function(self, count, integration_time_seconds, binsize_seconds, reprate, tau_COUNT_APD = 45e-9):
        """
        Corrects nonlinearities of counts for higher Rates. 

        Parameters
        ----------
        count : Measured Counts, array like. 
        
        integration_time_seconds : Integration time of the measurement. 
        
        binsize: binsize in ns. 
        
        tau_COUNT_APD: Value for the dead time of the APD, found on the Laser Components COUNT HySprint. 
            
        Returns
        -------
        Measured Rate
        
        """
        rate_measured = count/(binsize_seconds*integration_time_seconds*reprate)
        
        return rate_measured

    def SPV_folder_read(self, setup = "Dittrich"):
        if(self.mode == "wannsee"):
            setup = "wannsee"
            files = [f for f in os.listdir(self.TRPL_folderpath) if (isfile(join(self.TRPL_folderpath, f)) and f.endswith(".Wfm.csv"))]
        elif(setup == "Dittrich"):
            files = [f for f in os.listdir(self.TRPL_folderpath) if (isfile(join(self.TRPL_folderpath, f)) and f.endswith("txt"))]
        
        files.sort()

        if (self.SPV_reprates_Hz == []):
            self.SPV_NDs = []
            self.SPV_samples = []
            self.SPV_oscilloRecord = []
            self.SPV_measNum = []
        

        fpar = os.path.splitext(files[0])[0]
        meas_ps = fpar.split("_")
        # print(meas_ps)
        # print(meas_ps[0][-1])
        if(self.mode == "wannsee"):
            self.SPV_oscilloRecord.append((meas_ps[0][-1]))
            self.SPV_measNum.append((meas_ps[0][:-1]))
            self.SPV_NDs.append(meas_ps[2])
            self.SPV_samples.append(meas_ps[1])
        else:
            print("Not incoding sample params for SPV")
        
        dfs = self.import_SPV(join(self.TRPL_folderpath, files[0]), setup = setup)

        for i, f in enumerate(files):
            if(i == 0):
                continue
            df = self.import_SPV(join(self.TRPL_folderpath, f), setup = setup)
            dfs = pd.concat([dfs, df], axis = 1)

            fpar = os.path.splitext(f)[0]
            meas_ps = fpar.split("_")
            # print(meas_ps)
            # print(meas_ps[0][-1])
            if(self.mode == "wannsee"):
                self.SPV_oscilloRecord.append((meas_ps[0][-1]))
                self.SPV_measNum.append((meas_ps[0][:-1]))
                self.SPV_NDs.append(meas_ps[2])
                self.SPV_samples.append(meas_ps[1])
            else:
                print("Not incoding sample params")
            

        
        return files, dfs

    def import_SPV(self, filepath, setup = None):
        if (setup == "Dittrich"):
            df = pd.read_csv(filepath, skiprows=0, sep = "\t", comment='#')
            names = [os.path.basename(filepath)[:-4]+": "+str(n) for n in df.columns]
            df.columns = names
        elif(setup == "wannsee"):
            df = pd.read_csv(filepath, skiprows=0, header = None, sep = ",")
            cols = ["time [s]", "voltage [V]"]
            names = [os.path.basename(filepath)+": "+cols[n] for n in range(len(df.columns))]
            df.columns = names
        return df

    def TRPL_savedata(self, filename, selection = None,  sep=',', col_filenames = True):

        """
        Saves the trPL arrays in the specified order. 

        Parameters - They come from the output of the TRPL_folder_read function
        ----------
        *arrays: number of arrays to be saved next to time values, array, array, array, like.

        times: time values, array like.
        
        selection : slection of measurements to be saved, array-like
        
        colnames : prefix for column names corresponding to the *arrays, array-like
        
        filename : name of the .csv file to be saved, path-like
        
        files: file names corresponding to the times, array-like.
            
        Returns
        -------
        data: Dataframe that was used to save the .csv file.
        
        """

        if (selection == None):
            selection = [i for i in range(len(self.TRPLs_files))]

        data = pd.DataFrame()
        l = len(self.TRPLs_ts[:,selection[0]])

        for i, s in enumerate(selection):
            if (i == 0):
                how = 'right'        
            elif (len(self.TRPLs_ts[:,s]) > l):
                how = 'right'
            else:
                how = 'left'
            
            if (col_filenames):
                data = data.join(pd.Series(self.TRPLs_ts[:,s]).rename("time: "+self.TRPLs_files[s]), how = how)
                data = data.join(pd.Series(self.TRPLs_subsMean[:,s]).rename("trPL denoised: "+self.TRPLs_files[s]), how = how)
                data = data.join(pd.Series(self.TRPLs_n[:,s]).rename("trPL normalised: "+self.TRPLs_files[s]), how = how)
                data = data.join(pd.Series(self.TRPLs_raw[:,s]).rename("trPL raw: "+self.TRPLs_files[s]), how = how)
            else:
                data = data.join(pd.Series(self.TRPLs_ts[:,s]).rename("time"), how = how)
                data = data.join(pd.Series(self.TRPLs_subsMean[:,s]).rename("trPL denoised"), how = how)
                data = data.join(pd.Series(self.TRPLs_n[:,s]).rename("trPL normalised"), how = how)
                data = data.join(pd.Series(self.TRPLs_raw[:,s]).rename("trPL raw"), how = how)
        
            # for j, array in enumerate(arrays):
            #     if (col_filenames):
            #         #data[] = pd.Series()
            #         data = data.join(pd.Series(array[:, selection].transpose()[i]).rename(colnames[j]+": "+files[selection[i]]), how = how)
            #     else:
            #         data = data.join(pd.Series(array[:, selection].transpose()[i]).rename(colnames[j]), how = how)
            #         #data[colnames[j]] = pd.Series(array[:, selection].transpose()[i])
            # #print(elements)

            l = len(self.TRPLs_ts[:,s])

        data.to_csv(filename, index=None, sep = sep)
        return data

    def correction_function(self, count, integration_time_seconds, binsize_nanoseconds, reprate, tau_COUNT_APD = 45e-9):
        """
        Corrects nonlinearities of counts for higher Rates. 

        Parameters
        ----------
        count : Measured Counts, array like. 
        
        integration_time_seconds : Integration time of the measurement. 
        
        binsize: binsize in ns. 
        
        tau_COUNT_APD: Value for the dead time of the APD, found on the Laser Components COUNT HySprint. 
            
        Returns
        -------
        Corrected Counts
        
        """
        rate_measured = count/(1e-9*binsize_nanoseconds*integration_time_seconds*reprate)
        rate_actual = rate_measured/(1-rate_measured*tau_COUNT_APD)
        
        return rate_actual*(binsize_nanoseconds*integration_time_seconds*reprate)

    def plot_trPL(self, selection = None, **kwargs):
        if selection == None:
            selection = range(len(self.TRPLs_files))
        #Plot series
        fig, ax = plt.subplots(1,2, figsize = (kwargs.get("xplotsize", 20), kwargs.get("xplotsize", 10)))

        for i, (sPL) in enumerate(selection):
                ax[0].scatter(1e6*(self.TRPLs_ts[:,sPL]), self.TRPLs_subsMean[:,sPL], label = self.TRPL_sample[sPL]+" "+str(self.TRPL_powers[sPL]), alpha = 1)
                ax[1].scatter(1e6*(self.TRPLs_ts[:,sPL]), self.TRPLs_n[:,sPL], label = self.TRPL_sample[sPL]+" "+str(self.TRPL_powers[sPL]), alpha = 1)

        axisTicks_fontsize = kwargs.get("axisTicks_fontsize", 20)
        axis_fontsize = kwargs.get("axis_fontsize", 25)

        ax[0].set_title('trPL: Not normalised', fontsize=25)
        ax[0].set_xscale("linear")
        ax[0].set_ylabel("Count Rate [#]", fontsize=axis_fontsize) 
        ax[0].set_xlabel("Time [$\mu$s]", fontsize=axis_fontsize) 

        ax[1].set_title('trPL: normalised', fontsize=25)
        ax[1].set_xscale("linear")
        ax[1].set_ylabel("Normalised Count Rate [#]", fontsize=axis_fontsize) 
        ax[1].set_xlabel("Time [$\mu$s]", fontsize=axis_fontsize) 

        for a in ax:
            a.set_yscale(kwargs.get("yscale", "log"))
            a.set_xscale(kwargs.get("xscale", "linear"))
            if ("xlim" in kwargs):
                a.set_xlim(kwargs.get("xlim"))
            if ("ylim" in kwargs):
                a.set_xlim(kwargs.get("ylim"))

            axis_fontsize = 29
            axisTicks_fontsize = 20
            a.legend(fontsize = 15)
            a.tick_params(axis='x', labelsize=axisTicks_fontsize)
            a.tick_params(axis='y', labelsize=axisTicks_fontsize)
            a.grid(visible = True, which = 'both', linestyle = '--')

            w_frame = 2
            a.spines["bottom"].set_linewidth(w_frame)
            a.spines["top"].set_linewidth(w_frame)
            a.spines["left"].set_linewidth(w_frame)
            a.spines["right"].set_linewidth(w_frame)

        if ("savetofile" in kwargs):
            plt.savefig(kwargs.get("savetofile"), bbox_inches = "tight")
        
        plt.show()

        

        return None

    def plot_diffLifetimes(self, selection = None, **kwargs):

        if not(hasattr(self, 'densities')):
            print("Cannot plot the difflifetimes, please fit them first before plotting")
            return None

        if selection == None:
            selection = range(len(self.TRPLs_files))
        #Plot series
        fig, ax = plt.subplots(1,2, figsize = (kwargs.get("xplotsize", 20), kwargs.get("xplotsize", 10)))

        for i, (sPL) in enumerate(selection):
                ax[0].scatter(1e6*(self.TRPLs_ts[:,sPL]), self.TRPLs_n[:,sPL], label = self.TRPLs_files[sPL], alpha = 1)
                ax[1].scatter(self.kT*np.log((self.densities[sPL][:-1]*self.densities[sPL][:-1])/(self.ni*self.ni)), self.diff_taus[sPL], label = self.TRPLs_files[sPL], alpha = 1)

        axisTicks_fontsize = kwargs.get("axisTicks_fontsize", 20)
        axis_fontsize = kwargs.get("axis_fontsize", 25)

        ax[0].set_title('trPL: Normalised', fontsize=25)
        ax[0].set_xscale("linear")
        ax[0].set_ylabel("Count Rate [#]", fontsize=axis_fontsize) 
        ax[0].set_xlabel("Time [$\mu$s]", fontsize=axis_fontsize) 
        if ("xlim_PL" in kwargs):
            ax[0].set_xlim(kwargs.get("xlim_PL"))
        if ("ylim_PL" in kwargs):
            ax[0].set_ylim(kwargs.get("ylim_PL"))

        ax[1].set_title('QFLS vs Differential Lifetime', fontsize=25)
        ax[1].set_xscale("linear")
        ax[1].set_ylabel("Differential lifetime [s]", fontsize=axis_fontsize) 
        ax[1].set_xlabel("QFLS [eV]", fontsize=axis_fontsize)
        if ("xlim_diff" in kwargs):
            ax[1].set_xlim(kwargs.get("xlim_diff"))
        if ("ylim_diff" in kwargs):
            ax[1].set_ylim(kwargs.get("ylim_diff"))


        for a in ax:
            a.set_yscale(kwargs.get("yscale", "log"))
            a.set_xscale(kwargs.get("xscale", "linear"))
            axis_fontsize = 29
            axisTicks_fontsize = 20
            a.legend(fontsize = 15)
            a.tick_params(axis='x', labelsize=axisTicks_fontsize)
            a.tick_params(axis='y', labelsize=axisTicks_fontsize)
            a.grid(visible = True, which = 'both', linestyle = '--')

            w_frame = 2
            a.spines["bottom"].set_linewidth(w_frame)
            a.spines["top"].set_linewidth(w_frame)
            a.spines["left"].set_linewidth(w_frame)
            a.spines["right"].set_linewidth(w_frame)

        if ("savetofile" in kwargs):
            plt.savefig(kwargs.get("savetofile"), bbox_inches = "tight")
        
        plt.show()

        return None

    def fit_difflifetimes(self, selection = None, n_exp = None, l2 = None):
        """
        Calculates differential lifetime values, given a raw TRPL table, using a arbritrary amount of exponentials to fit the data.

        Parameters
        ----------
        time: time values, array like.
        
        TRPL_denoised : denoised TRPL values, array-like
        
        powers : Powers corresponding to the second dimension of TRPL_raw, array-like
        
        thickness : thickness of the samples, in cm.
        
        l2: number of data points considered for fitting.
            
        Returns
        -------
        time_fit: x_axis considered for fit
        densities2: the calculated carrier densities following n0*sqrt(data)
        diff_taus: diffenrential lifetimes, tau_diff = -2*(dt/d(log(fit))). following Thomas Kirchartzs' work
        
        """
        if selection == None:
            selection = range(len(self.TRPLs_files))

        if n_exp == None:
            n_exp = [5 for i in selection]

        ns_raw = self.TRPLs_raw[:,selection].transpose()
        time = self.TRPLs_ts[:,selection].transpose()

        diff_taus = []
        densities2 = []
        time_fit = []
        print("Number of exponentials for fit used is = "+str(n_exp)+"\n")
        
        f, ax = plt.subplots(3, len(selection), figsize=(25,12))
        for i in range(len(selection)):
            #t = time[i, (ns_raw[i, :] > 2*self.Noise[selection[0]]) & (time[i,:] > 0)]
            t = time[i, (time[i,:] > 0)]
            pl = ns_raw[i, (time[i,:] > 0)]
            #initial guess for fitting
            p = [1]*(2*n_exp[i]+1)
            if (l2 == None):
                #previous_ps, pcov = scipy.optimize.curve_fit(fitfunc, t[(pl >= 0)], np.log(pl[(pl >= 0)]), maxfev = 150000, p0 = p) 
                self.fitnoise = self.TRPLs_noise[selection[i]]
                previous_ps, pcov = scipy.optimize.curve_fit(self.fitfunc, t, (pl), maxfev = 150000, p0 = p) 
                #np.exp(np.log(start = 1, stop = 1+len(t[t < 1e-9*l2[i]]))))#method = 'lm', loss)
                fit = (self.fitfunc(t, *previous_ps))       
                tau_diff = -1*(np.diff(t)/np.diff(np.log(fit)))
                print("L2 is None")
            else:
                self.fitnoise = self.TRPLs_noise[selection[i]]
                previous_ps, pcov = scipy.optimize.curve_fit(self.fitfunc, t[(t < 1e-9*l2[i])], pl[(t < 1e-9*l2[i])], maxfev = 1500000, p0 = p)
                fit = (self.fitfunc(t[t < 1e-9*l2[i]], *previous_ps))
                fit_denoised = fit - self.TRPLs_noise[selection[i]] 
                tau_diff = -2*(np.diff(t[t < 1e-9*l2[i]])/np.diff(np.log(fit_denoised)))

            carrier_densities_fit = np.sqrt(fit_denoised/np.max(fit_denoised))*(self.n0s[selection[i]])
            print('{:.2e}'.format(self.n0s[selection[i]]))
            
            #Plotting
            ax[0, i].scatter(1e9*t, pl, marker = 'x')
            ax[0, i].plot(1e9*t[:len(fit)], (fit), color = 'orange')
            ax[0, i].set_yscale("log")
            ax[0, i].set_xlim([-0.05*l2[i] , l2[i]*2])
            ax[0, i].set_xlabel("time [ns]")
            ax[0, i].set_ylabel("PL counts [#]")

            ax[1, i].plot(1e9*t[:len(tau_diff)], tau_diff)
            ax[1, i].set_xlim([min(1e9*t), max(1e9*t[:len(tau_diff)])])
            ax[1, i].set_xlabel("time [ns]")
            ax[1, i].set_ylabel("Differential lifetime [s]")
            
            
            ax[2, i].plot(carrier_densities_fit[1:], tau_diff)
            ax[2, i].set_xlabel("Carrier Concentration [cm-3]")
            ax[2, i].set_ylabel("Differential lifetime [s]")
            ax[2, i].set_xscale("log")
            ax[2, i].set_yscale("log")
        
            densities2.append(carrier_densities_fit)
            diff_taus.append(tau_diff)
            time_fit.append(t[:len(fit)])
            
        f = plt.figure()
        for i, (b, c) in enumerate(zip(diff_taus, densities2)):
            plt.plot(c[:-1], b)
            #plt.plot(c[0:cut], savgol_filter(b[0:cut], 20, 7, mode = "nearest"))
            
        ax = f.gca()
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_xlabel("Carrier Density [cm-3]")
        ax.set_ylabel("Differential lifetime [s]")
        plt.legend(range(len(ns_raw)), loc = 'upper left')

        if (hasattr(self, 'time_fit')):
            for i, s in enumerate(selection):
                self.time_fit[s] = time_fit[i]
                self.densities[s] = densities2[i]
                self.diff_taus[s] = diff_taus[i]
        else: 
            self.time_fit = time_fit
            self.densities = densities2
            self.diff_taus = diff_taus
        
        return time_fit, densities2, diff_taus

    def diff_savedata(self, filename, selection = None):
        """
        Saves the differential lifetime values. 

        Parameters - They come from the output of the fit_difflifetimes function
        ----------
        filename: file name

        slection: id of files to be saved
            
        Returns
        -------
        data: Dataframe that was used to save the .csv file.
        
        """
        if selection == None:
            selection = range(len(self.TRPLs_files))

        if (hasattr(self, 'densities')):
            files_selection = [self.TRPLs_files[i] for i in selection]
            times =  [self.time_fit[i] for i in selection]
            densities = [self.densities[i] for i in selection]
            tau_diffs = [self.diff_taus[i] for i in selection]
        else:
            print("Diff Lifetimes not calculated yet. Please do so first.")
            return None

        data = pd.DataFrame()
        
        l = len(times[0])
        for i, (time, density, tau_d) in enumerate(zip(times, densities, tau_diffs)): 
            qfls = self.kT*np.log((density[:-1]*density[:-1])/(self.ni*self.ni))
            if (i == 0):
                how = 'right'
            elif(len(time) > l):
                how = 'right'
            else:
                how = 'left'
            
            data = data.join(pd.Series(time[:-1]).rename("time: "+files_selection[i]), how = how)
            data = data.join(pd.Series(density[:-1]).rename("density: "+files_selection[i]), how = how)
            data = data.join(pd.Series(qfls).rename("qfls: "+files_selection[i]), how = how)
            data = data.join(pd.Series(tau_d).rename("tau_diff: "+files_selection[i]), how = how)

            l = len(time)

        data.to_csv(filename)

        return data

    def fitfunc(self, x, *args):
        """
        
        Used for fitting of a multi (varying amount defined by len(args)) of exponential decay.

        Parameters
        ----------
        x: evaluation points of function.
        
        *args : arguments for multiexponential decay. len(args) has to be 2n + 1.
            
        Returns
        -------
        f = arg[0] + arg[1]*exp(-arg[2]*x) + ... + arg[i]*exp(-arg[i+1]*x)
        
        """
        #print(args)
        params = np.array([arg for arg in args])
        #params = params[:-1]
        
        if ((len(params) == 1) or (len(params) > 30)):
            print("Number of params is wrong n = "+str(len(params))+"\n")
        
        s = params[0]
        for i, a in enumerate(params):
            if (i == 0):
                continue
            if not(i % 2):
                continue
            else:
                s = s + params[i]*np.exp(-params[i+1]*x)
                
        return (s+self.fitnoise)

    def fitfunc2(self, x, *args):

        params = np.array([arg for arg in args])

        if ((len(params) < 2) or (len(params) > 20)):
                print("Number of params in fitfunc is wrong n = "+str(len(params))+"\n")

        numerator = 0
        denum = 0
        for i in range(int(len(params)/2)):
            numerator = numerator + params[i]*(x**i)
            denum = denum + params[int(len(params)/2)+i]*(x**i)
        


        return numerator/denum


