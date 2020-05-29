%% Preprocess High Gamma Dataset EEG data files for use with BCI Trainer and BCI Capture

clear all;

files = dir('*/*.mat');
channels = {'F3','Fz','F4','FC3','FC4','C5','C3','C1','Cz','C2','C4','C6','CP3','CP4','Pz','P3','P4','PO7','Oz','PO8'};

for f_i = 1:length(files)
    file = files(f_i);
    fname = [file.folder '\' file.name];
    fprintf("Processing file %s.\n",fname);

    % load channel names
    raw_ch_names = h5read(fname,'/nfo/clab');
    file_channels = {};
    channel_indices = zeros(1,length(channels));
    
    for i = 1:length(raw_ch_names)
        if any(strcmp(channels,char(raw_ch_names{i})))
            file_channels{end+1} = ['/ch' num2str(i)];
            channel_indices(find(strcmp(channels,char(raw_ch_names{i})))) = i;
        end
    end
        
    % extract parameters
    Fs = h5read(fname,'/nfo/fs');
    mark_times = h5read(fname,'/mrk/time'); % marker times in ms
    mark_label = h5read(fname,'/mrk/event/desc');
    
    Nt = zeros(1,4);
    class_trial_begin_indices = cell(1,4);
    
    for i_c = 1:4
        class_markers = mark_label==i_c;
        class_trial_begin_indices{i_c} = ceil(mark_times(class_markers==1) / 1000 * Fs);
    end
    
    % calculate the trial length in terms of samples
    Ns = 4 * Fs; % - 2 * Fs; % From paper, 4 s trials, crop the first and last second
    
    % bandpass filter the data between 8 and 30 Hz
    [z,p,k] = butter(4,[4,38]/(Fs/2),'bandpass');
    sos = zp2sos(z,p,k);
    
    % scale Ns for downsampling 
    Ns = Ns / 2;
    
    Nc = length(channels);
    class1_trials = zeros(length(class_trial_begin_indices{1}),Ns,Nc);
    class2_trials = zeros(length(class_trial_begin_indices{2}),Ns,Nc);
    class3_trials = zeros(length(class_trial_begin_indices{3}),Ns,Nc);
    class4_trials = zeros(length(class_trial_begin_indices{4}),Ns,Nc);
    
    for i_ch = 1:length(channel_indices)
        fprintf("\tProcessing Channel: %s...\n",channels{i_ch});
        
        % load channel data
        raw_data = h5read(fname,['/ch' num2str(i_ch)]);
        
        % apply the zero-phase filter
        fprintf("\t\tFiltering data...\n");
        filtered_data = filtfilt(sos,1,raw_data);
        
        % downsample
        fprintf("\tDownsampling data...\n");
        data = resample(filtered_data,1,2);
    
        % epoch the data
        fprintf("\t\tEpoching data...\n");
    
        for i_c = 1:4
            trial_start_indices = class_trial_begin_indices{i_c};
            for tn = 1:length(trial_start_indices)
               ts_t = round(trial_start_indices(tn)/2); % divide by 2 for resampling factor
               t = data(ts_t:ts_t+(Ns-1));
               switch i_c
                   case 1
                       class1_trials(tn,:,i_ch) = t;
                   case 2
                       class2_trials(tn,:,i_ch) = t;
                   case 3
                       class3_trials(tn,:,i_ch) = t;
                   otherwise
                       class4_trials(tn,:,i_ch) = t;
               end        
            end
        end
    end
    args = [channels;num2cell(channel_indices)];
    channel_map = struct(args{:});
    
    save(['preprocessed-' erase(erase(fname,pwd),'\')],'class1_trials','class2_trials','class3_trials','class4_trials','channel_map');
end