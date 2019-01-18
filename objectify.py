import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Record:

    def __init__(self, filename, total_time, exp_t, **tags):
        self._filename = filename
        self._total_time = total_time * 1000 # Convert to ms
        self._exp_t = exp_t
        #add _ to tags
        for key in tags:
            if not key.startswith('_'):
                tags['_' + key] = tags.pop(key)
        self.__dict__.update(tags)

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
        self.my_data["TimeTag"] = np.nan
        self.my_data.iloc[0, -1] = 0
        self.my_data.iloc[-1, -1] = self._total_time
    def divide(self):
        # Divide data on segments based on type of experiment
        self._segments = [] # TODO list of dataframes, use the list for analysis
        # All experiments have at least 3 initial segments.
        segm_1_finish = (self._tag1 - 5) * 1000
        segm_2_start = (self._tag1 + 5) * 1000
        segm_2_finish = (self._tag2 - 5) * 1000
        segm_3_start = (self._tag2 + 5) * 1000
        # The end is the same for all experiments, = self._total_time
        # if last 5 seconds should be overlooked
        # end = self._total_time - 5000

        # Experiments with Ach only have 3 segments (?)
        if self._exp_t == 'Ach':

            # First segment
            self.my_data.loc[self.my_data.loc[self.my_data.PeakTime < segm_1_finish].index[0], 'TimeTag'] = 0
            self.my_data.loc[self.my_data.loc[self.my_data.PeakTime < segm_1_finish].index[-1], 'TimeTag'] = segm_1_finish
            self.data_pre = self.my_data[self.my_data.PeakTime < segm_1_finish]
            self.data_pre.name = "pre: Ringer1"
            self._segments.append(self.data_pre)
            # Second segment
            self.my_data.loc[self.my_data.loc[(self.my_data.PeakTime > segm_2_start) &
                                              (self.my_data.PeakTime < segm_2_finish)].index[0], 'TimeTag'] = segm_2_start
            self.my_data.loc[self.my_data.loc[(self.my_data.PeakTime > segm_2_start) &
                                              (self.my_data.PeakTime < segm_2_finish)].index[-1], 'TimeTag'] = segm_2_finish
            self.data_during = self.my_data[(self.my_data.PeakTime > segm_2_start) & (self.my_data.PeakTime < segm_2_finish)]
            self.data_during.name = 'during: Ach'
            self._segments.append(self.data_during)
            # Last segment
            self.my_data.loc[self.my_data.loc[(self.my_data.PeakTime > segm_3_start) &
                                              (self.my_data.PeakTime < self._total_time)].index[0], 'TimeTag'] = segm_3_start
            self.my_data.loc[self.my_data.loc[(self.my_data.PeakTime > segm_3_start) &
                                              (self.my_data.PeakTime < self._total_time)].index[-1], 'TimeTag'] = self._total_time
            self.data_post = self.my_data[(self.my_data.PeakTime > segm_3_start) & (self.my_data.PeakTime < self._total_time)]
            self.data_post.name = 'post: Ringer2'
            self._segments.append(self.data_post)

        elif self._exp_t == 'AchStr':
            segm_3_finish = (self._tag3 - 5) * 1000
            segm_4_start = (self._tag3 + 5) * 1000

            # First segment
            self.my_data.loc[self.my_data.loc[self.my_data.PeakTime < segm_1_finish].index[0], 'TimeTag'] = 0
            self.my_data.loc[self.my_data.loc[self.my_data.PeakTime < segm_1_finish].index[-1], 'TimeTag'] = segm_1_finish
            self.data_pre1 = self.my_data[self.my_data.PeakTime < segm_1_finish]
            self.data_pre1.name = "pre1: Ringer1"
            self._segments.append(self.data_pre1)
            # Second segment
            self.my_data.loc[self.my_data.loc[(self.my_data.PeakTime > segm_2_start) &
                                              (self.my_data.PeakTime < segm_2_finish)].index[0], 'TimeTag'] = segm_2_start
            self.my_data.loc[self.my_data.loc[(self.my_data.PeakTime > segm_2_start) &
                                              (self.my_data.PeakTime < segm_2_finish)].index[-1], 'TimeTag'] = segm_2_finish
            self.data_pre2 = self.my_data[(self.my_data.PeakTime > segm_2_start) & (self.my_data.PeakTime < segm_2_finish)]
            self.data_pre2.name = 'pre2: Ringer + Str'
            self._segments.append(self.data_pre2)
            # Third segment
            self.my_data.loc[self.my_data.loc[(self.my_data.PeakTime > segm_3_start) &
                                              (self.my_data.PeakTime < segm_3_finish)].index[0], 'TimeTag'] = segm_3_start
            self.my_data.loc[self.my_data.loc[(self.my_data.PeakTime > segm_3_start) &
                                              (self.my_data.PeakTime < segm_3_finish)].index[-1], 'TimeTag'] = segm_3_finish
            self.data_during = self.my_data[(self.my_data.PeakTime > segm_3_start) & (self.my_data.PeakTime < segm_3_finish)]
            self.data_during.name = 'during: Ach + Str'
            self._segments.append(self.data_during)
            # Last segment
            self.my_data.loc[self.my_data.loc[(self.my_data.PeakTime > segm_4_start) &
                                              (self.my_data.PeakTime < self._total_time)].index[0], 'TimeTag'] = segm_4_start
            self.my_data.loc[self.my_data.loc[(self.my_data.PeakTime > segm_4_start) &
                                              (self.my_data.PeakTime < self._total_time)].index[-1], 'TimeTag'] = self._total_time
            self.data_post = self.my_data[(self.my_data.PeakTime > segm_4_start) & (self.my_data.PeakTime < self._total_time)]
            self.data_post.name = 'post: Ringer2'
            self._segments.append(self.data_post)

        elif self._exp_t == 'AchAtr':
            segm_3_finish = (self._tag3 - 5) * 1000
            segm_4_start = (self._tag3 + 5) * 1000
            segm_4_finish = (self._tag4 - 5) * 1000
            segm_5_start = (self._tag4 + 5) * 1000
            segm_5_finish = (self._tag5 - 5) * 1000
            segm_6_start = (self._tag5 + 5) * 1000

            # First segment
            self.my_data.loc[self.my_data.loc[self.my_data.PeakTime < segm_1_finish].index[0], 'TimeTag'] = 0
            self.my_data.loc[self.my_data.loc[self.my_data.PeakTime < segm_1_finish].index[-1], 'TimeTag'] = segm_1_finish
            self.data_pre1 = self.my_data[self.my_data.PeakTime < segm_1_finish]
            self.data_pre1.name = 'pre1: Ringer1'
            self._segments.append(self.data_pre1)
            # Second segment
            self.my_data.loc[self.my_data.loc[(self.my_data.PeakTime > segm_2_start) &
                                              (self.my_data.PeakTime < segm_2_finish)].index[0], 'TimeTag'] = segm_2_start
            self.my_data.loc[self.my_data.loc[(self.my_data.PeakTime > segm_2_start) &
                                              (self.my_data.PeakTime < segm_2_finish)].index[-1], 'TimeTag'] = segm_2_finish
            self.data_during1 = self.my_data[(self.my_data.PeakTime > segm_2_start) & (self.my_data.PeakTime < segm_2_finish)]
            self.data_during1.name = 'during1: Ach'
            self._segments.append(self.data_during1)
            # Third segment
            self.my_data.loc[self.my_data.loc[(self.my_data.PeakTime > segm_3_start) &
                                              (self.my_data.PeakTime < segm_3_finish)].index[0], 'TimeTag'] = segm_3_start
            self.my_data.loc[self.my_data.loc[(self.my_data.PeakTime > segm_3_start) &
                                              (self.my_data.PeakTime < segm_3_finish)].index[-1], 'TimeTag'] = segm_3_finish
            self.data_post1 = self.my_data[(self.my_data.PeakTime > segm_3_start) & (self.my_data.PeakTime < segm_3_finish)]
            self.data_post1.name = 'post1: Ringer2'
            self._segments.append(self.data_post1)
            # Fourth segment
            self.my_data.loc[self.my_data.loc[(self.my_data.PeakTime > segm_4_start) &
                                              (self.my_data.PeakTime < segm_4_finish)].index[0], 'TimeTag'] = segm_4_start
            self.my_data.loc[self.my_data.loc[(self.my_data.PeakTime > segm_4_start) &
                                              (self.my_data.PeakTime < segm_4_finish)].index[-1], 'TimeTag'] = segm_4_finish
            self.data_pre2 = self.my_data[(self.my_data.PeakTime > segm_4_start) & (self.my_data.PeakTime < segm_4_finish)]
            self.data_pre2.name = 'pre2: Ringer + Atr'
            self._segments.append(self.data_pre2)
            # Fifth segment
            self.my_data.loc[self.my_data.loc[(self.my_data.PeakTime > segm_5_start) &
                                              (self.my_data.PeakTime < segm_5_finish)].index[0], 'TimeTag'] = segm_5_start
            self.my_data.loc[self.my_data.loc[(self.my_data.PeakTime > segm_5_start) &
                                              (self.my_data.PeakTime < segm_5_finish)].index[-1], 'TimeTag'] = segm_5_finish
            self.data_during2 = self.my_data[(self.my_data.PeakTime > segm_5_start) & (self.my_data.PeakTime < segm_5_finish)]
            self.data_during2.name = 'during2: Ach + Atr'
            self._segments.append(self.data_during2)
            # Last segment
            self.my_data.loc[self.my_data.loc[(self.my_data.PeakTime > segm_6_start) &
                                              (self.my_data.PeakTime < self._total_time)].index[0], 'TimeTag'] = segm_6_start
            self.my_data.loc[self.my_data.loc[(self.my_data.PeakTime > segm_6_start) &
                                              (self.my_data.PeakTime < self._total_time)].index[-1], 'TimeTag'] = self._total_time
            self.data_post2 = self.my_data[(self.my_data.PeakTime > segm_6_start) & (self.my_data.PeakTime < self._total_time)]
            self.data_post2.name = 'post2: Ringer3'
            self._segments.append(self.data_post2)

        elif self._exp_t == 'AchBtx1':
            segm_3_finish = (self._tag3 - 5) * 1000
            segm_4_start = (self._tag3 + 5) * 1000

            # First segment
            self.my_data.loc[self.my_data.loc[self.my_data.PeakTime < segm_1_finish].index[0], 'TimeTag'] = 0
            self.my_data.loc[self.my_data.loc[self.my_data.PeakTime < segm_1_finish].index[-1], 'TimeTag'] = segm_1_finish
            self.data_pre1 = self.my_data[self.my_data.PeakTime < segm_1_finish]
            self.data_pre1.name = "pre1: Ringer1"
            self._segments.append(self.data_pre1)
            # Second segment
            self.my_data.loc[self.my_data.loc[(self.my_data.PeakTime > segm_2_start) &
                                              (self.my_data.PeakTime < segm_2_finish)].index[0], 'TimeTag'] = segm_2_start
            self.my_data.loc[self.my_data.loc[(self.my_data.PeakTime > segm_2_start) &
                                              (self.my_data.PeakTime < segm_2_finish)].index[-1], 'TimeTag'] = segm_2_finish
            self.data_pre2 = self.my_data[(self.my_data.PeakTime > segm_2_start) & (self.my_data.PeakTime < segm_2_finish)]
            self.data_pre2.name = 'pre2: Ringer + Btx'
            self._segments.append(self.data_pre2)
            # Third segment
            self.my_data.loc[self.my_data.loc[(self.my_data.PeakTime > segm_3_start) &
                                              (self.my_data.PeakTime < segm_3_finish)].index[0], 'TimeTag'] = segm_3_start
            self.my_data.loc[self.my_data.loc[(self.my_data.PeakTime > segm_3_start) &
                                              (self.my_data.PeakTime < segm_3_finish)].index[-1], 'TimeTag'] = segm_3_finish
            self.data_during = self.my_data[(self.my_data.PeakTime > segm_3_start) & (self.my_data.PeakTime < segm_3_finish)]
            self.data_during.name = 'during: Ach + Btx'
            self._segments.append(self.data_during)
            # Last segment
            self.my_data.loc[self.my_data.loc[(self.my_data.PeakTime > segm_4_start) &
                                              (self.my_data.PeakTime < self._total_time)].index[0], 'TimeTag'] = segm_4_start
            self.my_data.loc[self.my_data.loc[(self.my_data.PeakTime > segm_4_start) &
                                              (self.my_data.PeakTime < self._total_time)].index[-1], 'TimeTag'] = self._total_time
            self.data_post = self.my_data[(self.my_data.PeakTime > segm_4_start) & (self.my_data.PeakTime < self._total_time)]
            self.data_post.name = 'post: Ringer2'
            self._segments.append(self.data_post)

        elif self._exp_t == 'AchBtx2':
            segm_3_finish = (self._tag3 - 5) * 1000
            segm_4_start = (self._tag3 + 5) * 1000
            segm_4_finish = (self._tag4 - 5) * 1000
            segm_5_start = (self._tag4 + 5) * 1000
            segm_5_finish = (self._tag5 - 5) * 1000
            segm_6_start = (self._tag5 + 5) * 1000

            # First segment
            self.my_data.loc[self.my_data.loc[self.my_data.PeakTime < segm_1_finish].index[0], 'TimeTag'] = 0
            self.my_data.loc[self.my_data.loc[self.my_data.PeakTime < segm_1_finish].index[-1], 'TimeTag'] = segm_1_finish
            self.data_pre1 = self.my_data[self.my_data.PeakTime < segm_1_finish]
            self.data_pre1.name = 'pre1: Ringer1'
            self._segments.append(self.data_pre1)
            # Second segment
            self.my_data.loc[self.my_data.loc[(self.my_data.PeakTime > segm_2_start) &
                                              (self.my_data.PeakTime < segm_2_finish)].index[0], 'TimeTag'] = segm_2_start
            self.my_data.loc[self.my_data.loc[(self.my_data.PeakTime > segm_2_start) &
                                              (self.my_data.PeakTime < segm_2_finish)].index[-1], 'TimeTag'] = segm_2_finish
            self.data_during1 = self.my_data[(self.my_data.PeakTime > segm_2_start) & (self.my_data.PeakTime < segm_2_finish)]
            self.data_during1.name = 'during1: Ach'
            self._segments.append(self.data_during1)
            # Third segment
            self.my_data.loc[self.my_data.loc[(self.my_data.PeakTime > segm_3_start) &
                                              (self.my_data.PeakTime < segm_3_finish)].index[0], 'TimeTag'] = segm_3_start
            self.my_data.loc[self.my_data.loc[(self.my_data.PeakTime > segm_3_start) &
                                              (self.my_data.PeakTime < segm_3_finish)].index[-1], 'TimeTag'] = segm_3_finish
            self.data_post1 = self.my_data[(self.my_data.PeakTime > segm_3_start) & (self.my_data.PeakTime < segm_3_finish)]
            self.data_post1.name = 'post1: Ringer2'
            self._segments.append(self.data_post1)
            # Fourth segment
            self.my_data.loc[self.my_data.loc[(self.my_data.PeakTime > segm_4_start) &
                                              (self.my_data.PeakTime < segm_4_finish)].index[0], 'TimeTag'] = segm_4_start
            self.my_data.loc[self.my_data.loc[(self.my_data.PeakTime > segm_4_start) &
                                              (self.my_data.PeakTime < segm_4_finish)].index[-1], 'TimeTag'] = segm_4_finish
            self.data_pre2 = self.my_data[(self.my_data.PeakTime > segm_4_start) & (self.my_data.PeakTime < segm_4_finish)]
            self.data_pre2.name = 'pre2: Ringer + Btx'
            self._segments.append(self.data_pre2)
            # Fifth segment
            self.my_data.loc[self.my_data.loc[(self.my_data.PeakTime > segm_5_start) &
                                              (self.my_data.PeakTime < segm_5_finish)].index[0], 'TimeTag'] = segm_5_start
            self.my_data.loc[self.my_data.loc[(self.my_data.PeakTime > segm_5_start) &
                                              (self.my_data.PeakTime < segm_5_finish)].index[-1], 'TimeTag'] = segm_5_finish
            self.data_during2 = self.my_data[(self.my_data.PeakTime > segm_5_start) & (self.my_data.PeakTime < segm_5_finish)]
            self.data_during2.name = 'during2: Ach + Btx'
            self._segments.append(self.data_during2)
            # Last segment
            self.my_data.loc[self.my_data.loc[(self.my_data.PeakTime > segm_6_start) &
                                              (self.my_data.PeakTime < self._total_time)].index[0], 'TimeTag'] = segm_6_start
            self.my_data.loc[self.my_data.loc[(self.my_data.PeakTime > segm_6_start) &
                                              (self.my_data.PeakTime < self._total_time)].index[-1], 'TimeTag'] = self._total_time
            self.data_post2 = self.my_data[(self.my_data.PeakTime > segm_6_start) & (self.my_data.PeakTime < self._total_time)]
            self.data_post2.name = 'post2: Ringer3'
            self._segments.append(self.data_post2)

    # TODO print the "name" + the name of the segment, workaround, set the names with the segment name
        print(f'Your experiment has {len(self._segments)} segments: {[segment.name for segment in self._segments]}')


    # TODO make analysis class?

    def raster_plot(self,line= 0.01):
        # Line size can be chosen, TODO tweak graph parameters
        plt.eventplot(self.my_data.PeakTime, linewidths=line)
        plt.show()

    def measure_time(self, data):
        return data.loc[data.index[-1], 'TimeTag'] - data.loc[data.index[0], 'TimeTag']

    def afr(self, data= 'everything'):
        ''' Returns the average firing rate of the required section in spikes / second (Hz)
        Returns a summary of all the sections by default
        '''
        for segment in self._segments:
            segment.afr = (len(segment) / self.measure_time(segment)) * 1000
        if data == 'everything':
            for segment in self._segments:
                print(segment.name + ' -> ' + str(round(segment.afr, 3)) + ' Hz')
        elif data == 'global':
            afr = (len(self.my_data) / self.measure_time(self.my_data)) * 1000
            print(f'The average firing rate for the whole experiment is: {round(afr, 3)} Hz')
        else:
            for seg in self._segments:
                if data in seg.name:
                    return seg.afr # round?


    def meanisi(self, data= 'everything'):
        ''' Returns mean ISI for the required section
        Returns a summary of all the sections by default
        '''
        for segment in self._segments:
            segment.isi = segment.ISI.mean()
        if data == 'everything':
            for segment in self._segments:
                print(segment.name + ' -> ' + str(round(segment.isi, 3)))
        elif data == 'global':
            print(f'The mean ISI for the whole experiment is: {round(self.my_data.ISI.mean(), 3)}')
        else: 
            for seg in self._segments:
                if data in seg.name:
                    return seg.ISI.mean() # round?


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