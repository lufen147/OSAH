function img_groundtruth_data = im_evaluation_load_groundtruth(img_query_name, path)
    % im_evaluation_load_groundtruth: Given the groundtruth image name.
    % input:
    %   img_query_name: corresponding images query names
    %   path: datasets groundtruth directory, string
    % output:
    %   img_groundtruth_data: the ground data, n * 4 cell type
    
    load('opts');
    img_groundtruth_data = cell(1,5);    
%     disp(['load images groundtruth data from ', path, '(total: ', num2str(size(img_query_name,2)), ')      ']);
    if strcmp(opts.datasets.name, 'holidays_upright')
        img_name_list = dir([path, opts.file.format_common, opts.file.format_jpg]); 
        img_name_list_num = numel(img_name_list);
        
        j = 0;
        k = 1;
        for i = 1:img_name_list_num
            this_img_name = split(img_name_list(i).name, '.');         
            
            if mod(str2double(this_img_name(1)), 100) == 0
                j = j + 1;
                img_groundtruth_data{j, 1} = this_img_name(1);   % query image name
                img_groundtruth_data{j, 4} = this_img_name(1);   % this_img_name(1); junk_set;  % junk image name
                img_groundtruth_data{j, 5} = opts.datasets.name; % datasets name
                k = 1;
                good_set = {};
                
%                 fprintf(1,'\b\b\b\b\b\b%6d',fix(j));
            else
                good_set(k,1) =  this_img_name(1);
                img_groundtruth_data{j, 2} = good_set;  % good images name
                k = k + 1;
            end
        end
    end
    
    if ismember(opts.datasets.name, ["oxford5k", "paris6k", "oxford105k", "paris106k"])
        for i = 1:length(img_query_name)
            this_img_query_name = img_query_name(i);

            groundtruth_prefix = [path, char(this_img_query_name)];        
            good_set = importdata([groundtruth_prefix, '_good.txt']);
            ok_set = importdata([groundtruth_prefix, '_ok.txt']);
            junk_set = importdata([groundtruth_prefix, '_junk.txt']);
            img_groundtruth_data{i, 1} = this_img_query_name;    % query image name
            img_groundtruth_data{i, 2} = good_set;               % good images name
            img_groundtruth_data{i, 3} = ok_set;                 % ok images name
            img_groundtruth_data{i, 4} = junk_set;               % junk images name
            img_groundtruth_data{i, 5} = opts.datasets.name;     % datasets name
            
%             fprintf(1,'\b\b\b\b\b\b%6d',fix(i));
        end        
    end
%     fprintf(1,'\n');
%     toc
end