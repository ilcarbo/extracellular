import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Record:

    def __init__(self, filename, total_time, exp_t, **tags):
        self._filename = filename
        self._total_time = total_time * 1000 # Convert to ms
        self._exp_t = exp_t
        self.__dict__.update(tags)

    #def __init__(self, filename, total_time, exp_t, tag1= None, tag2= None, tag3= None, tag4= None, tag5= None):
    #    self._filename = filename
    #    self._total_time = total_time * 1000 # Convert to ms
    #    self._exp_t = exp_t # type of experiment, it can be Ach, AchStr, AchAtr, AchBtx1, AchBtx2
    #    self._tag1 = tag1
    #    self._tag2 = tag2
    #    self._tag3 = tag3
    #    self._tag4 = tag4
    #    self._tag5 = tag5

    def read_and_clean(self):
        # import data table as pandas df
        self._data = pd.read_csv(self._filename, sep='\t', engine="python", header= 0, skiprows= 2)

        # Remove empty columns
        self._data = self._data.iloc[:, 0:33]

        # Keep only columns of interest
        self.my_data = self._data.loc[:, ["Event Start Time (ms)", "Event End Time (ms)", "Baseline (pA)", "Peak Amp (pA)",
                               "Time to Peak (ms)", "Time of Peak (ms)", "Rise Tau (ms)", "Decay Tau (ms)",
                               "Inst. Freq. (Hz)", "Interevent Interval (ms)"]]

        # Convert not found and nan to np.nans
        nan_vals = ["nan", "Not found"]
        self.my_data.replace(nan_vals, np.nan, inplace=True)

        # Make all columns numeric
        self.my_data = self.my_data.apply(pd.to_numeric)

        # Rename columns
        self.my_data.columns = ["Start", "End", "Baseline", "PeakAmp",
                           "TimetoPeak", "PeakTime", "RiseTau", "DecayTau",
                           "InstFreq", "ISI"]

        # Create event duration column
        self.my_data["Duration"] = self.my_data.End - self.my_data.Start

        # Drop values with irrational (longer) event durations
        self.my_data = self.my_data[self.my_data.Duration <= (self.my_data.Duration.mean() + 2 * self.my_data.Duration.std())]

        # Drop events with peak amplitudes > mean +- 2std or < mean +- 2std
        self.my_data = self.my_data[self.my_data.PeakAmp <= (self.my_data.PeakAmp.mean() + 2 * self.my_data.PeakAmp.std())]
        self.my_data = self.my_data[self.my_data.PeakAmp >= (self.my_data.PeakAmp.mean() - 2 * self.my_data.PeakAmp.std())]

        # Create Time column with NaNs
        self.my_data["TimeTag"] = np.nan #TODO assign values 0 and end to general dataframe

    def divide(self):
        # Divide data on segments based on type of experiment
        segments = [] # TODO make it self, list of dataframes, use the list for analysis
        # TODO kill underscore in tags (?)
        # All experiments have at least 3 initial segments.
        segm_1_finish = (self._tag1 - 5) * 1000
        segm_2_start = (self._tag1 + 5) * 1000
        segm_2_finish = (self._tag2 - 5) * 1000
        segm_3_start = (self._tag2 + 5) * 1000
        # The end is the same for all experiments
        end = self._total_time - 5000

        # TODO Change segments name directly?
        # TODO create time column for each segment with Nan, change first item for start, last for finish.
        # Experiments with Ach only have 3 segments (?)
        if self._exp_t == 'Ach':
            # self.data_Rpre = self.my_data[self.my_data.PeakTime < segm_1_finish] # change it directly in my_data
            #self.data_Rpre['TimeTag'].iloc[0] = 0 # this way raises chained assignment warning
            #self.my_data.loc[self.data_Rpre.index[0], 'TimeTag'] = 0 # Correct way? NO
            self.my_data.loc[self.my_data.loc[self.my_data.PeakTime < segm_1_finish].index[0], 'TimeTag'] = 0
            #self.data_Rpre['TimeTag'].iloc[-1] = segm_1_finish # this way raises chained assignment warning
            #self.my_data.loc[self.data_Rpre.index[-1], 'TimeTag'] = segm_1_finish # correct way? NO
            self.my_data.loc[self.my_data.loc[self.my_data.PeakTime < segm_1_finish].index[-1], 'TimeTag'] = segm_1_finish
            self.data_Rpre = self.my_data[self.my_data.PeakTime < segm_1_finish] # Es una grasada pero funciona
            self.data_Rpre.name = 'Ringer'
            segments.append(self.data_Rpre)
            # self.data_ach = self.my_data[(self.my_data.PeakTime > segm_2_start) & (self.my_data.PeakTime < segm_2_finish)]
            # self.data_ach['TimeTag'].iloc[0] = segm_2_start
            # self.data_ach['TimeTag'].iloc[-1] = segm_2_finish
            # self.data_ach.name = 'Ach'
            # segments.append(self.data_ach)
            # self.data_Rpost = self.my_data[(self.my_data.PeakTime > segm_3_start) & (self.my_data.PeakTime < end)]
            # self.data_Rpost['TimeTag'].iloc[0] = segm_3_start
            # self.data_Rpost['TimeTag'].iloc[-1] = end
            # self.data_Rpost.name = 'Ringer'
            # segments.append(self.data_Rpost) #TODO check chained assignments from hell
            # dataframes can have attributes assigned, data_Rpre.tstart and data_Rpre.tend can be assigned but they'll
            # work only for that copy

        #UNCHANGED
        elif self._exp_t == 'AchStr':
            segm_3_finish = (self._tag3 - 5) * 1000
            segm_4_start = (self._tag3 + 5) * 1000
            self.data_pre1 = self.my_data[self.my_data.PeakTime < segm_1_finish]
            self.data_pre1.name = 'Ringer'
            segments.append(self.data_pre1.name)
            self.data_pre2 = self.my_data[(self.my_data.PeakTime > segm_2_start) & (self.my_data.PeakTime < segm_2_finish)]
            self.data_pre2.name = 'Ringer + Str'
            segments.append(self.data_pre2.name)
            self.data_during = self.my_data[(self.my_data.PeakTime > segm_3_start) & (self.my_data.PeakTime < segm_3_finish)]
            self.data_during.name = 'Ach + Str'
            segments.append(self.data_during.name)
            self.data_post = self.my_data[(self.my_data.PeakTime > segm_4_start) & (self.my_data.PeakTime < end)]
            self.data_post.name = 'Ringer'
            segments.append(self.data_post.name)
        elif self._exp_t == 'AchAtr':
            segm_3_finish = (self._tag3 - 5) * 1000
            segm_4_start = (self._tag3 + 5) * 1000
            segm_4_finish = (self._tag4 - 5) * 1000
            segm_5_start = (self._tag4 + 5) * 1000
            segm_5_finish = (self._tag5 - 5) * 1000
            segm_6_start = (self._tag5 + 5) * 1000
            self.data_pre1 = self.my_data[self.my_data.PeakTime < segm_1_finish]
            self.data_pre1.name = 'Ringer'
            segments.append(self.data_pre1.name)
            self.data_during1 = self.my_data[
                (self.my_data.PeakTime > segm_2_start) & (self.my_data.PeakTime < segm_2_finish)]
            self.data_during1.name = 'Ach'
            segments.append(self.data_during1.name)
            self.data_post1 = self.my_data[
                (self.my_data.PeakTime > segm_3_start) & (self.my_data.PeakTime < segm_3_finish)]
            self.data_post1.name = 'Ringer'
            segments.append(self.data_post1.name)
            self.data_pre2 = self.my_data[(self.my_data.PeakTime > segm_4_start) & (self.my_data.PeakTime < segm_4_finish)]
            self.data_pre2.name = 'Ringer + Atr'
            segments.append(self.data_pre2.name)
            self.data_during2 = self.my_data[(self.my_data.PeakTime > segm_5_start) & (self.my_data.PeakTime < segm_5_finish)]
            self.data_during2.name = "Ach + Atr"
            segments.append(self.data_during2.name)
            self.data_post2 = self.my_data[(self.my_data.PeakTime > segm_6_start) & (self.my_data.PeakTime < end)]
            self.data_post2.name = "Ringer"
            segments.append(self.data_post2.name)
        elif self._exp_t == 'AchBtx1':
            segm_3_finish = (self._tag3 - 5) * 1000
            segm_4_start = (self._tag3 + 5) * 1000
            self.data_pre1 = self.my_data[self.my_data.PeakTime < segm_1_finish]
            self.data_pre1.name = 'Ringer'
            segments.append(self.data_pre1.name)
            self.data_pre2 = self.my_data[
                (self.my_data.PeakTime > segm_2_start) & (self.my_data.PeakTime < segm_2_finish)]
            self.data_pre2.name = 'Ringer + Btx'
            segments.append(self.data_pre2.name)
            self.data_during = self.my_data[
                (self.my_data.PeakTime > segm_3_start) & (self.my_data.PeakTime < segm_3_finish)]
            self.data_during.name = 'Ach + Btx'
            segments.append(self.data_during.name)
            self.data_post = self.my_data[(self.my_data.PeakTime > segm_4_start) & (self.my_data.PeakTime < end)]
            self.data_post.name = 'Ringer'
            segments.append(self.data_post.name)
        elif self._exp_t == 'AchBtx2':
            segm_3_finish = (self._tag3 - 5) * 1000
            segm_4_start = (self._tag3 + 5) * 1000
            segm_4_finish = (self._tag4 - 5) * 1000
            segm_5_start = (self._tag4 + 5) * 1000
            segm_5_finish = (self._tag5 - 5) * 1000
            segm_6_start = (self._tag5 + 5) * 1000
            self.data_pre1 = self.my_data[self.my_data.PeakTime < segm_1_finish]
            self.data_pre1.name = 'Ringer'
            segments.append(self.data_pre1.name)
            self.data_during1 = self.my_data[
                (self.my_data.PeakTime > segm_2_start) & (self.my_data.PeakTime < segm_2_finish)]
            self.data_during1.name = 'Ach'
            segments.append(self.data_during1.name)
            self.data_post1 = self.my_data[
                (self.my_data.PeakTime > segm_3_start) & (self.my_data.PeakTime < segm_3_finish)]
            self.data_post1.name = 'Ringer'
            segments.append(self.data_post1.name)
            self.data_pre2 = self.my_data[
                (self.my_data.PeakTime > segm_4_start) & (self.my_data.PeakTime < segm_4_finish)]
            self.data_pre2.name = 'Ringer + Btx'
            segments.append(self.data_pre2.name)
            self.data_during2 = self.my_data[
                (self.my_data.PeakTime > segm_5_start) & (self.my_data.PeakTime < segm_5_finish)]
            self.data_during2.name = "Ach + Btx"
            segments.append(self.data_during2.name)
            self.data_post2 = self.my_data[(self.my_data.PeakTime > segm_6_start) & (self.my_data.PeakTime < end)]
            self.data_post2.name = "Ringer"
            segments.append(self.data_post2.name)

        # TODO define type of experiment when initializing, divide based on that tag. keep time tags with kwargs

    # TODO print how many data segments were obtained.
        print(f'Your experiment has {len(segments)} segments: {[segment.name for segment in segments]}')


    # TODO make analysis class?

    def raster_plot(self,line= 0.01):
        # Line size can be chosen, TODO tweak graph parameters
        plt.eventplot(self.my_data.PeakTime, linewidths=line)
        plt.show()

    def afr(self, data):
        # Returns the average firing rate of the required section
        # TODO check units
        return len(data) / self._total_time

    def isi(self, data):
        # Returns mean ISI for the section
        return data.ISI.mean()

    def ifr1(self, data, window= 50, plot= False): #TODO fix, doesn't work
        #
        bins = np.linspace(0, self._total_time, num=(self._total_time / window))
        self.my_data["BinnedPeaks"] = pd.cut(self.my_data["PeakTime"], bins)
        binned = self.my_data[["PeakTime", "BinnedPeaks"]].groupby("BinnedPeaks").count() / window

        # Return mean ifr?, in Hz?
        # binned_sps = [x * 1000 for x in binned]

        if plot == True:
            binned.plot()


    #TODO function for printing to file, unique file.
    #TODO statistics.


#if __name__ == "__main__":
    #TODO print explanation for the use of the module