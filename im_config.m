function opts = im_config(opts)
    % config file of project on your assign data sets
    % hyper parameters config
    % Authors: F. Lu. 2020. 
    tic

    opts.file.root = fileparts(mfilename('fullpath'));  % get this project's root, opts, a new define struct type
    root = opts.file.root;          % define a simply variable for using below
    opts.file.name = mfilename;     % get this script file name, not include extend name
    opts.file.format_txt = '.txt';  % config txt file format, noted that "."
    opts.file.format_jpg = '.jpg';  % config jpg file format, noted that "."
    opts.file.format_mat = '.mat';  % config mat file format, noted that "."
    opts.file.format_npy = '.npy';  % config npy file format, noted that "."
    opts.file.format_dat = '.dat';  % config dat file format, noted that "."
    opts.file.format_cvs = '.cvs';  % config cvs file format, noted that "."
    opts.file.format_common = '*';  % config the images name, * is any name

    opts.run.data_temp = './data_temp/';       % generate temp mat data
    if ~exist(opts.run.data_temp, 'dir')
        mkdir(opts.run.data_temp);
    end
    
    if ~isfield(opts.run, 'epoch')
        opts.run.epoch = 1;                 % if need running of epoch, option of [1 2 3 ...].
    end

    opts.extract.batchsize = 1;     % config the batch images number input to CNN while extract feature, option of [1, 64, 128, 256]

    opts.datasets.datapath = './data/';   % config the data file path

    datapath = opts.datasets.datapath;        % define a simply variable for using below
    if ~isfield(opts.datasets, 'name')
        opts.datasets.name = 'oxford5k';        % config datasets name, one of [oxford5k, paris6k, roxford5k, rparis6k, oxford105k, paris106k, holidays, ukbench, flickr100k]
    end
    if strcmp(opts.datasets.name, 'oxford5k')
        opts.datasets.image_path = [datapath, '/datasets/Oxford5K/oxbuild_images/'];  % config the images datasets orgin path
        opts.datasets.gt_path = [datapath, '/datasets/Oxford5K/gt_files_170407/'];    % config the images datasets orgin path
        opts.features.query_path = [datapath, '/features/oxford5k/pool5_queries/'];      % config the query images feature save path
        opts.match.rank_path = [datapath, '/features/oxford5k/rank_file/'];             % config the optional path to save query image match ranked ouput
    end
    if strcmp(opts.datasets.name, 'oxford105k')
        opts.datasets.gt_path = [datapath, '/datasets/Oxford5K/gt_files_170407/'];    % config the images datasets orgin path
        opts.features.query_path = [datapath, '/features/oxford5k/pool5_queries/'];      % config the query images feature save path
        opts.match.rank_path = [datapath, '/features/oxford105k/rank_file/'];             % config the optional path to save query image match ranked ouput
    end
    if strcmp(opts.datasets.name, 'paris6k')
        opts.datasets.image_path = [datapath, '/datasets/Paris6K/paris_images/'];     % config the images datasets orgin path
        opts.datasets.gt_path = [datapath, '/datasets/Paris6K/gt_files_120310/'];     % config the images datasets orgin path
        opts.features.query_path = [datapath, '/features/paris6k/pool5_queries/'];       % config the query images feature save path
        opts.match.rank_path = [datapath, '/features/paris6k/rank_file/'];              % config the optional path to save query image match ranked ouput
    end
    if strcmp(opts.datasets.name, 'paris106k')
        opts.datasets.gt_path = [datapath, '/datasets/Paris6K/gt_files_120310/'];     % config the images datasets orgin path
        opts.features.query_path = [datapath, '/features/paris6k/pool5_queries/'];       % config the query images feature save path
        opts.match.rank_path = [datapath, '/features/paris106k/rank_file/'];              % config the optional path to save query image match ranked ouput
    end
    if strcmp(opts.datasets.name, 'holidays_upright')
        opts.datasets.image_path = [datapath, '/datasets/Holidays_upright/'];                 % config the images datasets orgin path
        opts.datasets.gt_path = [datapath, '/datasets/Holidays_upright/'];                    % config the images datasets orgin path
        opts.features.query_path = [datapath, '/features/holidays_upright/pool5_queries/'];     % config the query images feature save path
        opts.match.rank_path = [datapath, '/features/holidays_upright/rank_file/'];           % config the optional path to save query image match ranked ouput
    end
    if strcmp(opts.datasets.name, 'flickr100k')
        opts.datasets.image_path = [datapath, '/datasets/Flickr100K/oxc1_100k/'];     % config the images datasets orgin path
    end

    if ~isfield(opts.features, 'net')
        opts.features.net = 'vgg16';    % config net model frame, one of [vgg16, caffe, matconvnet, matconvnet_dag]
    end
    opts.features.net_layer = 'pool5';              % config images feature extracted from which net layter
    if ~isfield(opts.features, 'dimension')
        opts.features.dimension = 128;              % config the images feature extracted dimension
    end
    if ~isfield(opts.features, 'pipeline_model')
        opts.features.pipeline_model = 'none';      % config pipeline model such as Dimension reduction model, one of [none, norm, pca, pcaw]
    end
    if ~isfield(opts.features, 'cross_model')
        opts.features.cross_model = 'OSAH';        % config calculate cross model, one of [mhdf], mhdf: mid- and high deep feature
    end

    if ~isfield(opts.match, 'metric')
        opts.match.metric = 2;           % config the metric (measure) method. option: L1:1, L2:2, Canberra distance:3, Correlation similarity:4, Cosine similarity:5, Histogram intersection:6, Inner product distance:7 
    end

    save('opts', 'opts');       % save and use for some module loading

end