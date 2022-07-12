%% Preprocess Kaya et al. EEG data files for use with BCI Trainer and BCI Capture

clear all;

files = dir('CLA*Subject*.mat');

for f_i = 1:length(files)
    file = files(f_i);
    fname = file.name;
    fprintf("Processing file %s.\n",fname);
    
    fdata = load(fname);
    
    % extract parameters
    Fs = fdata.o.sampFreq;
    markers = fdata.o.marker;
    Nt = zeros(1,3);
    class_trial_begin_indices = cell(1,3);
    
    for i_c = 1:3
        class_markers = diff(markers==i_c);
        class_trial_begin_indices{i_c} = find(class_markers==1);
    end
    
    % calculate the trial length in terms of samples
    Ns = Fs + 3*Fs; % From paper, 1 s trials, add a 1.5s buffer on each side of trial
    
    % scale Ns for resampling
    Ns = Ns / Fs * 250;
    
    % bandpass filter the data between 8 and 30 Hz
    [z,p,k] = butter(4,[4,38]/(Fs/2),'bandpass');
    sos = zp2sos(z,p,k);
    
    % apply the zero-phase filter
    fprintf("\tFiltering data...\n");
    filtered_data = filtfilt(sos,1,fdata.o.data);
    
    % resample
    fprintf("Resampling data...\n");
    ups = 5;
    downs = 4;
    data = resample(filtered_data,ups,downs);
    
    % epoch the data
    fprintf("\tEpoching data...\n");
    Nc = size(filtered_data,2);
    class_trials = cell(1,3);
    
    for i_c = 1:3
        trial_start_indices = round((class_trial_begin_indices{i_c} - 1.5*Fs)*ups/downs);
        ctmat = zeros(length(class_trial_begin_indices{i_c}),Ns,Nc);
        for tn = 1:length(trial_start_indices)
            ts_t = trial_start_indices(tn);
            ctmat(tn,:,:) = data(ts_t:ts_t+(Ns-1),:);
        end
        class_trials{i_c} = ctmat;
    end
    
    class1_trials = class_trials{1};
    class2_trials = class_trials{2};
    class3_trials = class_trials{3};
    
    save(['preprocessed-' fname],'class1_trials','class2_trials','class3_trials');
end