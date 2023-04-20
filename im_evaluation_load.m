function data = im_evaluation_load(opts)
    % im_evaluation_load: run and load data for evaluating.
    % input:
    %   eval: include evaluation modular parameters, struct type
    %   opts: include im system global parameters, struct type
    % output:
    %   data: return the data sets, struct type, include:
    %       f_data: return the img features data, n * p double type
    %       f_name: return the img features name, n * p double type
    %       q_data: return the img query data, n * p double type
    %       q_name: return the img query name, n * p double type
    %       gt_data: return the ground data, n * 5 cell type   
    
   %% load features data   
    
    dn = opts.datasets.name;
    feature_file = [opts.run.data_temp, 'MHDF3_', dn, ''];
    MHDF3 = importdata([feature_file, opts.file.format_mat]);
    img_features_data = MHDF3.MHDF;
    img_features_name = MHDF3.name;
     
    %% load query data
    
    path = [opts.features.query_path, opts.file.format_common, opts.file.format_txt];
    [img_query_data, img_query_name, ~] = im_evaluation_load_query(img_features_data, img_features_name, path);  % load query images name and features
   
    %% load groundtruth data
    path = opts.datasets.gt_path;
    img_groundtruth_data = im_evaluation_load_groundtruth(img_query_name, path);
    
    %% return data
    data.f_data = img_features_data;
    data.f_name = img_features_name;
    data.q_data = img_query_data;
    data.q_name = img_query_name;
    data.gt_data = img_groundtruth_data;
end
