function [img_query_data, img_query_name, img_query_image_name] = im_evaluation_load_query(img_features_data, img_features_name, path)
    % im_evaluation_load_query: Given the query image name and features after features processing.
    % input:
    %   img_features_data: features after features processing
    %   img_features_name: corresponding features names
    %   path: query image info directory, string
    % output:
    %   img_query_data: the list of loaded query features, list
    %   img_query_name: corresponding file names without extension, list
    %   img_query_image_name: corresponding image names without extension, list
    
    img_query_data = [];
    img_query_name = "";
    img_query_image_name = "";
    img_query_info = dir(path);
%     disp(['load images features info from ', path, '(total: ', num2str(size(img_query_info,1)), ')      ']);
    for i = 1:size(img_query_info)
        this_img_query_name = split(img_query_info(i).name, '.');
        img_query_name(i) = this_img_query_name{1};
        
        this_img_query_features_name = importdata([img_query_info(i).folder, '/', img_query_info(i).name]);
        if iscell(this_img_query_features_name)
            img_query_image_name(i) = this_img_query_features_name{1};
        else
            img_query_image_name(i) = this_img_query_features_name;
        end
        
        j = img_features_name == img_query_image_name(i);
        this_query_info_X = img_features_data(j, :);
        img_query_data(i,:) = this_query_info_X;
        
%         fprintf(1,'\b\b\b\b\b\b%6d',fix(i));
    end
%     fprintf(1,'\n');
%     toc
end

