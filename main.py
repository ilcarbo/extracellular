import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# TODO transform to objects.
# class will be record, atributes: file, tags, total time...
#file_name = 'data/p1.atf'
#tags = list?
total_time = 300000 # in ms

# TODO view differences between view and copy, how to use them

# Detect spikes with clampfit and export table as .ATF
# import data table as pandas df

data = pd.read_csv('data/p1.atf', sep='\t', engine="python", header= 0, skiprows= 2)
#
#data = pd.read_table('data/testeo.txt', sep= '/s+', skiprows= 2)


# Remove empty columns
data = data.iloc[:, 0:33]

# Keep only columns of interest
my_data = data.loc[:, ["Event Start Time (ms)", "Event End Time (ms)", "Baseline (pA)", "Peak Amp (pA)",
                       "Time to Peak (ms)", "Time of Peak (ms)", "Rise Tau (ms)", "Decay Tau (ms)",
                       "Inst. Freq. (Hz)", "Interevent Interval (ms)"]]

# Convert not found and nan to np.nans
nan_vals = ["nan", "Not found"]
my_data.replace(nan_vals, np.nan, inplace=True)

# Make all columns numeric
my_data = my_data.apply(pd.to_numeric)

# Rename columns
my_data.columns = ["Start", "End", "Baseline", "PeakAmp",
                   "TimetoPeak", "PeakTime", "RiseTau", "DecayTau",
                   "InstFreq", "ISI"]

# Create event duration column
my_data["Duration"]  = my_data.End - my_data.Start

# Drop values with irrational (longer) event durations
my_data = my_data[my_data.Duration <= (my_data.Duration.mean() + my_data.Duration.std())]

# Drop events with peak amplitudes > mean +- 2std or < mean +- 2std
#my_data = my_data[my_data.PeakAmp <= (my_data.PeakAmp.mean() + 2 * my_data.PeakAmp.std())]
#my_data = my_data[my_data.PeakAmp >= (my_data.PeakAmp.mean() - 2 * my_data.PeakAmp.std())]
my_data = my_data[my_data.PeakAmp <= (my_data.PeakAmp.mean() + 2 * my_data.PeakAmp.std()) & \
my_data.PeakAmp >= (my_data.PeakAmp.mean() - 2 * my_data.PeakAmp.std())]

# Raster plot
def rasterPlot(data, line)
    plt.eventplot(data, linewidths= line)
    plt.show()
#plt.eventplot(my_data.PeakTime, linewidths= 0.01)

# divide data in pre, drug, post, etc.
# input tag times manually (TODO get them automatically)
# add and substract 5 seconds directly? / get the lenght of the recording automatically. or input it manually
#
tag1 = 58.4044 * 1000 # time in ms
tag2 = 119.5678 * 1000
#tagn = ...

#data_pre = my_data[my_data.PeakTime < tag1]
# perfusion opening and closing effect can be avoided taking 5 seconds before and after tag.
data_pre = my_data[my_data.PeakTime < (tag1 - 5000)]
data_during = my_data[(my_data.PeakTime > (tag1 + 5000)) & (my_data.PeakTime < (tag2 - 5000))]
data_post = my_data[my_data.PeakTime > (tag2 + 5000)]



# determine average firing rate in each section
def afr(data, time):
    return len(data) / time

#get mean ISI for each section (ISI HAS TO BE NUMERIC)
def isi(data):
    return data.ISI.mean()

#bin spikes in x ms windows and plot "firing rate". TODO make it function
#window = 50 # in ms
#bins = np.linspace(0, total_time, num= (total_time / window))
#my_data["BinnedPeaks"] = pd.cut(my_data["PeakTime"], bins)
#binned = my_data[["PeakTime", "BinnedPeaks"]].groupby("BinnedPeaks").count() / window
# TODO convert to spikes per second?, list comprehension can be used?
#binned_sps = [x * 1000 for x in binned]
#binned.plot()
#MAKE HIST? WHAT ARE THE PARAMETERS?

#bin spikes with varying bin size, fixed spikes per bin
bins = 50
bins

#make output file with: filename, freq for each window, ISI for each window
