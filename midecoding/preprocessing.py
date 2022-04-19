import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import signal
plt.style.use('seaborn-darkgrid')

CHANNELS = ["C3","CZ","C4"]

def preprocess(data, detrend=True, outlier_handling=True, avg_ref=True):
    '''
    Removes unnecessary columns. Renames columns properly (FP1->C3, M1->Cz,P3->C4).
    Removes trends and large scale fluctuations from data. (approximately 2Hz Highpass based on simple moving average)
    Replaces sample (if it exceeds 6 standard deviations) by the average of previous and next sample. Only adresses single sample outliers. 
    Does not effectively remove larger artifacts.
    Rereferences against average of C3,CZ and C4.
    
    Args:
        data (pandas.DataFrame): Raw Data read from unaltered EEG-Droid recording.
    
    Returns
        df (pandas.DataFrame): Preprocessed data
    '''
    
    # Adjust Column Names
    df = data[["sampling_time","FP1","M1","P3","stimulus"]]
    df = df.rename(columns={"FP1":"C3","M1":"CZ","P3":"C4"})
    
            
    #Detrend - Simple Moving Average with inferred cut-off frequency of ~2.01Hz (calculated in "cutoff frequency.ipynb")
    if(detrend):
        for ch in CHANNELS:
            # Simple moving average Filter, high-pass cut-off frequency ~ 
            rolled = df[ch].rolling(55,min_periods=1,closed="right",center=False)
            df.loc[:,ch] = df[ch] - rolled.mean()
        
    #Remove outliers
    if(outlier_handling):
        if(not detrend):
            print("WARNING: Outlier Removal without prior detrending might yield faulty results.")
        for ch in CHANNELS:
            df.loc[:,ch] = replace_outliers(df.loc[:,ch])
    
    # Average Reference   
    if (avg_ref):
        df["AVG"] = df[CHANNELS].mean(axis=1) 
        for ch in CHANNELS:
            df.loc[:,ch] -= df.loc[:,"AVG"]
                
    return df
    
def replace_outliers(data, n_stds=6):
    '''
    Finds values that exceed the data mean by n_stds. Values are replaced by the mean of the sample before and after.
    Only works with single sample artifacts. Artifacts that extend over multiple samples are not effectively handled by
    this function.
    Args: 
        data (pandas.Series): series containing EEG data
        n_stds (int): number of standard deviations values may deviate from the mean
    Returns:
        data (pandas.Series) 
    '''
    # Check if a value exceeds local variance by a factor of stds
    threshold = data.std()*n_stds
    mean = data.mean()
    #variances = data.rolling(1000,min_periods=1,closed="right",center=False).std()
    
    for i in range(1,len(data)-1):
        if abs(data.loc[i]-mean) > threshold:
            #print(f"DEBUG: Obvious Outlier detected at millisecond {i*4} with value {df.loc[i,ch]}")
            data.loc[i] = (data.loc[i-1]+data.loc[i+1])/2#
            data.loc[i]
    return data


### TRIAL EXTRACTION ###
def get_stimulus_timestamps(data):
    '''
    Custom function for MI-Data timestamps. Finds the onsets of different stimuli and returns
    a list of (timestamp,stimulus) pairs. Timestamps are integers that represent the amount of milliseconds
    passed since the beginning of the recording.
    Args:
        data (DataFrame): Pandas DataFrame with, including a column named "stimulus"
    Returns:
        [(int,str)]: List of timestamp, stimulus pairs
    '''
    
    stimuli = data.stimulus.values
    stimulus_timestamps = []
    for i,stimulus in enumerate(stimuli):
        if stimuli[i-1] != stimuli[i]: 
            stimulus_timestamps.append((i*4,stimulus))  # '4' -> Assuming a sr of 250Hz
    return stimulus_timestamps

def get_mi_trials(data, augmentation=False):
    '''
    Cuts out slices of a data frame that exclusively contain signals associated with motor imagery.
    In an academic context, these slices are called "trials".
    For the current experiment these slices contain the data obtained while an arrow was displayed on-screen.
    Useful here for calculating evoked signals by averaging motor imagery data. Excludes trials with signal values >45muV, 
    as they likely contain artifactsç
    
    Args:
        df (DataFrame): Pandas DataFrame with EEG data. Expected columns: [idx,(CHANNELS),stimulus]) 
        augmentation (Boolean): Switch that determines whether 25 new sub trials are added per trial.
    Returns:
        (lh_signals,rh_signals): Two lists containing a bunch of equally long pd.DataFrames,
                                each DataFrame representing one trial.
        info (dic): Dictionary with entries "lh_discarded" and "rh_discarded" to check how many trials were discarded.
        
    '''
    df = data.copy(deep=True)
    lh_signals = []
    rh_signals = []
    lh_discarded = 0
    rh_discarded = 0

    stimulus_timestamps = get_stimulus_timestamps(df)

    for i, (timestamp,stimulus) in enumerate(stimulus_timestamps):
        # If the current timestamp signalizes the onset of a left- or right pointing arrow
        if stimulus in ["l","r"]:
            # Calculate the indeces of the interval during which the arrow is displayed
            start_idx = int(timestamp/4)   # timestamp is in ms, but we need the sample index. (4ms / sample)
            end_idx = start_idx+1000  
            n_subtrials = 1
            
            if (augmentation):
                start_idx = int((timestamp-3000)/4) #Starts 3 Seconds prior to the actual motor imagery signal
                n_subtrials = 25                    # 25 trials with stride of 124ms 
                
                            
            for _ in range(n_subtrials):
                end_idx = start_idx + 1000      # 1000 samples correspond to 4 seconds.                
                # Obtain this particular slice of data
                trial = df[start_idx:end_idx]
                #print(f"DEBUG: end idx = {end_idx}, last idx = {stimulus_timestamps[-1][0]/4}")

                if likely_contains_artifact(trial):
                    if stimulus == "l":
                        lh_discarded += 1
                    else:
                        rh_discarded += 1
                        
                else:
                    # Reset indices to 0
                    trial.index -=  trial.index[0]
                    trial["stimulus"][0] = stimulus
                    # Add it to the list of signals evoked by left-ward or right-ward pointing arrows
                    if stimulus == "l":
                        lh_signals.append(trial)
                    else:
                        rh_signals.append(trial)
                
                if (augmentation):
                    start_idx += 31    # + 31 samples <=> +124ms  
            
    return lh_signals,rh_signals,{"lh_discarded":lh_discarded,"rh_discarded":rh_discarded}

def likely_contains_artifact(trial):
    
    return (trial[CHANNELS].max()>=45).any() or (trial[CHANNELS].min()<=-45).any()

def reshape_trials_for_cnn(trials):
    '''
    Args:
        trials ([pd.DataFrame]) List of DataFrames, each DataFrame representing a 4 second MI trial
                                  
    Returns:
        trials_ndarray (np.ndarray): array with shape (number of trials, samples per trial, number of channels)
        labels_ndarray (np.ndarray): array with shape (number of trials, 2), 2 representing the two classes: "left" and "right" (one hot encoding). e.g. : labels_ndarray[0] = np.ndarray([1,0])  -> "left" or np.ndarray([0,1]) -> "right"
    '''
    CHANNELS = ["C3","CZ", "C4"]
    n_trials = len(trials)
    n_samples = len(trials[0])
    n_channels = len(CHANNELS)
    mi_classes = 2
    
    left_class = np.array([1,0])
    right_class = np.array([0,1])
    empty_class = np.array([0,0])
    
    trials_ndarray = np.ndarray((n_trials,n_samples,n_channels), np.float)
    labels_ndarray = np.ndarray((n_trials,mi_classes), np.float)
    
    for i,trial in enumerate(trials):
        trials_ndarray[i] = trial[CHANNELS].values
        
        if trial["stimulus"][0] == 'l':
            labels_ndarray[i] = left_class
        elif trial["stimulus"][0] == 'r':
            labels_ndarray[i] = right_class
        else:
            labels_ndarray[i] = empty_class
    
    return trials_ndarray, labels_ndarray  # basically input , output
    
def evoked_from_slices(slices):
    '''
    Calculates the evoked signal (average response after stimulus) based on slices of equal length
    Args:
        slices ([pandas.DataFrame]): List of DataFrame slices representing stimulus responses
    Returns
        evoked_signal (pandas.DataFrame): DataFrame with each column representing the evoked signal
    '''
    
    normalized_slices = []
    for snippet in slices:
        normalized_snippet = (snippet[CHANNELS] - snippet[CHANNELS].mean())/snippet[CHANNELS].std()
        normalized_slices.append(normalized_snippet)
        
    # Compute the evoked signal (average normalized slices)
    evoked_signal = sum(normalized_slices)/len(normalized_slices)
    
    print(f"DEBUG: Total number of slices:{len(normalized_slices)}")
    evoked_signal.insert(0,"sampling_time",evoked_signal.index*4)
    
    return evoked_signal

def evoked_potentials(df):
    lh_trials, rh_trials, info = get_mi_trials(df)
    lh_info = {"used":len(lh_trials),"discarded":info["lh_discarded"]}
    rh_info = {"used":len(rh_trials),"discarded":info["rh_discarded"]}
    
    lh_evoked = evoked_from_slices(lh_trials)
    rh_evoked = evoked_from_slices(rh_trials)
    
    return {"lh_evoked":lh_evoked,"lh_info":lh_info,"rh_evoked":rh_evoked,"rh_info":rh_info}
    

### VISUALIZATION ###
def plot_channel(data,channel,ax=None, stimuli_visible=True):
    '''
    Quick and simple visualization of a channel. Creates an axis if none has been passed.
    Args: 
        data (pd.DataFrame): containing EEG signal values and expected columns: ["sampling_time",(..Channels..), "stimulus", (optional others)]
        channel (str): Name of the Channel to plot, e.g. "C3"
        ax (AxesSubplot): axis to plot on
        stimuli_visible (boolean): determines whether stimuli onsets are additionally drawn over the plot as vertical lines
    
    '''
    
    if ax==None:
        fig,ax = plt.subplots()
    
    ax.plot(data["sampling_time"],data[channel],label=channel)
    ax.set_title(f"Signal of {channel}")
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("microVolt")
    ax.legend()
    
    if(stimuli_visible):
        plot_stimuli(ax)    
        
def plot_stimuli(data,ax):
    '''
    Draws stimuli onsets as vertical lines on top of a plot.
    Args:
        data (DataFrame): Pandas DataFrame with, including a column named "stimulus"
        ax (AxesSubplot): axis to plot on
    '''
    stimulus_timestamps = get_stimulus_timestamps(data)
    
    colors = {"f":"red","b":"black","r":"green","l":"blue"}
    y_min, y_max = ax.get_ylim()

    for stimulus in stimulus_timestamps:
        time = stimulus[0]
        stimulus = stimulus[1]
        plt.vlines(time, y_min, y_max, colors[stimulus],label=stimulus)
    
    # add legend
    eeg_lines = [Line2D([0], [0], color=line.get_color()) for line in ax.lines]
    eeg_labels = [line.get_label() for line in ax.lines]
    
    stimuli_lines = [Line2D([0], [0], color="black", lw=2),
                Line2D([0], [0], color="red", lw=2),
                Line2D([0], [0], color="green", lw=2),
                Line2D([0], [0], color="blue", lw=2)]
    stimuli_labels = ["Blank","Fixation","Right Hand MI","Left Hand MI"]
    
    ax.legend(eeg_lines+stimuli_lines,eeg_labels+stimuli_labels)
    
def plot_evoked(df=None,evoked=None,subject=""):
        
    if evoked == None:
        evoked = evoked_potentials(df)
        
    lh_evoked = evoked["lh_evoked"]
    rh_evoked = evoked["rh_evoked"]

    
    #info for fig title
    n_lh = evoked["lh_info"]["used"]
    n_rh = evoked["rh_info"]["used"]
    discarded = evoked["rh_info"]["discarded"] + evoked["lh_info"]["discarded"]
    fig_title = f"{subject} across 3 Sessions \n LH Stimuli:{n_lh}, RH Stimuli:{n_rh}  \n Total discarded={discarded}"
    
    fig, ax = plt.subplots(len(CHANNELS),1)
    for i,ch in enumerate(CHANNELS):
        ax[i].plot(lh_evoked["sampling_time"], lh_evoked[ch],label="Left Hand MI",color="green",lw=1)
        ax[i].plot(rh_evoked["sampling_time"], rh_evoked[ch],label="Right Hand MI",color="red",lw=1)
        ax[i].set_title(f"Evoked Signals on Electrode {ch}")
        ax[i].set_ylabel("μV")
        handles,labels = ax[i].get_legend_handles_labels()
        
    fig.legend(handles,labels,loc="lower right")
    fig.suptitle(fig_title)
    plt.tight_layout()
    plt.show()
    
    return
    
def plot_spectrograms(df=None,evoked=None,subject=""):
    fig,axs = plt.subplots(2,3,sharey=True)
    
    #Load Data 
    if evoked == None:
        evoked = evoked_potentials(df)
    lh_evoked = evoked["lh_evoked"]
    rh_evoked = evoked["rh_evoked"] 
    
    for i,ch in enumerate(CHANNELS):
        # Left Hand
        f, t, Sxx = signal.spectrogram(lh_evoked[ch], 250)
        axs[0,i].pcolormesh(t, f, Sxx, shading='nearest', vmin=Sxx.min(),vmax=Sxx.max()*0.7)
        axs[0,i].set_ylim(4,30)
        
        axs[0,i].set_ylabel('Frequency [Hz]')
        axs[0,i].set_xlabel('Time [sec]')
        axs[0,i].set_title(ch + " Left Hand MI")

        # Right Hand
        f, t, Sxx = signal.spectrogram(rh_evoked[ch], 250)
        axs[1,i].pcolormesh(t, f, Sxx, shading='nearest',vmin=Sxx.min(),vmax=Sxx.max()*0.7)
        axs[1,i].set_ylim(4,30)

        
        axs[1,i].set_ylabel('Frequency [Hz]')
        axs[1,i].set_xlabel('Time [sec]')
        axs[1,i].set_title(ch + " Right Hand MI")
        
    
    plt.suptitle(f"spectrograms of evoked signals for {subject}")
    plt.tight_layout()
    plt.show()
    
def plot_preprocessing_steps(file,save_plots=True):
    data = pd.read_csv(file,skipfooter=2)
    subject = file.split("_")[1].split("-")[0]
    
    raw = preprocess(data,avg_ref=False, detrend=False,outlier_handling=False)
    detrended = preprocess(data,detrend=True,outlier_handling=False, avg_ref=False)
    no_outlier = preprocess(data,detrend=True,outlier_handling=True,avg_ref=False)
    avg_ref = preprocess(data, detrend=True,outlier_handling=True, avg_ref=True)
    fig, axs = plt.subplots(4,1,sharex=True)

    preprocessed_datasets = [raw,detrended,no_outlier,avg_ref]
    preprocessed_labels = ["Raw","Detrended with SMA, cut-off frequency: ~2.01Hz","Single Outlier Replacement","Average Referenced",]

    for i,ax in enumerate(axs):
        data = preprocessed_datasets[i]
        title = preprocessed_labels[i]
        lw=2
        alpha=1
        for ch in CHANNELS:
            lw*=0.5
            alpha*=0.9
            ax.plot(data["sampling_time"],data[ch],label=ch,lw = lw, alpha=alpha)
            ax.set_title(title)
            handles,labels = ax.get_legend_handles_labels()

    plt.tight_layout()
    fig.legend(handles,labels,loc="upper right")
    fig.suptitle(f"Preprocessing for subject {subject}")
    plt.show()
    if(save_plots):
        plt.savefig(f"Plots/{subject}-Raw.png")
        
def plot_psd(lh_evoked,rh_evoked,figtitle=""):
    fig,axs = plt.subplots(3,1,sharey=True)

    for i in range(len(axs)):
        ch = CHANNELS[i]
        axs[i].psd(rh_evoked[ch],250,100, label="Right Hand MI")
        axs[i].psd(lh_evoked[ch],250,100,label="Left Hand MI")
        axs[i].set_title(f"PSD of {ch}")
        axs[i].set_ylabel("dB/Hz")
        handles,labels = axs[i].get_legend_handles_labels()

    fig.suptitle(figtitle)
    fig.legend(handles,labels,loc="upper right")
    plt.tight_layout()
    plt.show()