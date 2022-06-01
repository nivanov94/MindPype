%% Preprocess BCI Comp IV-2a EEG data files for use with BCI Trainer and BCI Capture

clear all;

files = dir('raw_data/A0*.gdf');
label_files = dir('true_labels/A0*.mat');

for f_i = 1:length(files)
    file = files(f_i);
    fname = file.name;
    fprintf("Processing file %s.\n",fname);
    
    label_file = label_files(f_i);
    label_fname = label_file.name;
    
    [eeg, info] = sload(['raw_data\' fname]);
    eeg = eeg(:,1:22);
    eeg(isnan(eeg)) = -realmax('double');
    labels = load(['true_labels\' label_fname]);
    
    % extract parameters
    Fs = info.SampleRate;
    trial_start_indices = info.EVENT.POS(info.EVENT.TYP == 768);
    Nt = zeros(1,4);
    class_trial_begin_indices = cell(1,4);
    
    % get run start indices
    run_start_indices = info.EVENT.POS(info.EVENT.TYP == 32766);
    run_start_indices = run_start_indices(4:end); % ignore first three runs with eye movement data
    runs = length(run_start_indices);
    
    for i_c = 1:4
        class_markers = labels.classlabel == i_c;
        class_trial_begin_indices{i_c} = trial_start_indices(class_markers);
    end
    
    % calculate the trial length in terms of samples
    Ns = 6 * Fs; % From paper, 6 s trials
    
    % bandpass filter the data between 4 and 38 Hz
    [z,p,k] = butter(4,[4,38]/(Fs/2),'bandpass');
    sos = zp2sos(z,p,k);
    
    % apply the zero-phase filter
    fprintf("\tFiltering data...\n");
    filtered_data = zeros(size(eeg));
    
    for i_r = 1:runs
        if i_r < runs
            end_index = run_start_indices(i_r+1) - 1;
        else
            end_index = size(eeg,1);
        end
        filtered_data(run_start_indices(i_r)+100:end_index,:) =  filtfilt(sos,1,eeg(run_start_indices(i_r)+100:end_index,:));
    end
    
    % epoch the data
    fprintf("\tEpoching data...\n");
    Nc = size(filtered_data,2);
    class_trials = cell(1,4);
    
    for i_c = 1:4
        trial_start_indices = class_trial_begin_indices{i_c};
        ctmat = zeros(length(class_trial_begin_indices{i_c}),Ns,Nc);
        for tn = 1:length(trial_start_indices)
            ts_t = trial_start_indices(tn);
            ctmat(tn,:,:) = filtered_data(ts_t:ts_t+(Ns-1),:);
        end
        class_trials{i_c} = ctmat;
    end
    
    class1_trials = class_trials{1};
    class2_trials = class_trials{2};
    class3_trials = class_trials{3};
    class4_trials = class_trials{4};
    
    p = strrep(fname,'gdf','mat');
    save(['preprocessed-' p],'class1_trials','class2_trials','class3_trials','class4_trials');
end