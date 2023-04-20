function Z = im_evaluation(opts)
    % this file is 2nd step runtime on im project
    % This script file steps as below:
    % get the config globle parameters, struct type and named opts, 
    % and then set evaluation paramers, struct type and named eval, 
    % and then input opts and eval to mAP function, output mAP value, double type.
    % Authors: F. Lu. 2020. 

    % clear opts eval;
    opts.file.name = mfilename;    % get this script file name

    %% calculate mAP
    mAP = im_evaluation_mAP(opts);     
    disp(['mAP', '_', opts.datasets.name, '_', opts.features.cross_model, '_', num2str(opts.features.dimension), ' is: ', num2str(mAP)]);        % output and display
    % toc

    %% generate test report
    if ~exist('report_eval.mat', 'file')
        r = 1;
    else
        load('report_eval');
        r = numel(report_eval);
        r = r + 1;
    end
    report_eval(r).frame = opts.features.net;
    report_eval(r).method = opts.features.cross_model;
    report_eval(r).whitening = opts.features.pipeline_model;
    report_eval(r).datasets = opts.datasets.name;
    report_eval(r).epoch = opts.run.epoch;
    report_eval(r).dimension = opts.features.dimension;
    report_eval(r).mAP = mAP;
    report_eval(r).totaltime = toc;
    report_eval(r).datetime = datetime;
    save('report_eval', 'report_eval');

    report_eval = struct2table(report_eval);
    writetable(report_eval, 'report_eval.csv');

    % disp('successed to save the eval report'); 
end