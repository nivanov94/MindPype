%% Preprocess Cho et al. EEG data files for use with BCI Trainer and BCI Capture

clear all;

files = dir('s*.mat');

for f_i = 1:length(files)
    file = files(f_i);
    fname = file.name;
    fprintf("Processing file %s.\n",fname);
    
    fdata = load(fname);
    
    % extract parameters
    Fs = fdata.eeg.srate;
    Nt = fdata.eeg.n_imagery_trials;
    markers = fdata.eeg.imagery_event;
    
    % calculate the trial length in terms of samples
    Ns = ceil(fdata.eeg.frame(2)/1000 * Fs) - 2*Fs; % crop the first and last second
    
    % bandpass filter the data between 4 and 38 Hz
    [z,p,k] = butter(4,[4,38]/(Fs/2),'bandpass');
    sos = zp2sos(z,p,k);
    
    % apply the zero-phase filter
    fprintf("\tFiltering data...\n");
    filtered_left  = filtfilt(sos,1,double(fdata.eeg.imagery_left)');
    filtered_right = filtfilt(sos,1,double(fdata.eeg.imagery_right)');
    
    % resample the data to 250 Hz
    ups = 125;
    downs = 256;
    left = resample(filtered_left,ups,downs);
    right = resample(filtered_right,ups,downs);
    
    % scale Ns
    Ns = Ns / Fs * 250;
    
    % epoch the data
    fprintf("\tEpoching data...\n");
    Nc = size(filtered_left,2);
    class1_trials = zeros(Nt,Ns,Nc);
    class2_trials = zeros(Nt,Ns,Nc);
    
    trial_start_indices = find(markers) + Fs;
    for tn = 1:Nt
        ts_t = round(trial_start_indices(tn) * 250 / Fs);
        class1_trials(tn,:,:) = left(ts_t:ts_t+(Ns-1),:);
        class2_trials(tn,:,:) = right(ts_t:ts_t+(Ns-1),:);
    end
    
    save(['preprocessed-' fname],'class1_trials','class2_trials');
end