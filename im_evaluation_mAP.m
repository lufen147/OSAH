function mAP = im_evaluation_mAP(opts)
    % im_evaluation_map: run full evaluation pipeline on specified data.
    % input:
    %   eval: include evaluation modular parameters, struct type
    %   opts: include im system global parameters, struct type
    % output:
    %   mAP: return the mAP value, double type
    
    data = im_evaluation_load(opts);
    
    img_features_data = data.f_data;
    img_features_name = data.f_name;
    img_query_data = data.q_data;
    img_query_name = data.q_name;
    img_groundtruth_data = data.gt_data;
    
%     disp('Ready computing the mAP...');
    ap = zeros();
    for i = 1:length(img_query_name)
        this_query_X = img_query_data(i,:);
        this_img_query_name = img_query_name{i};
        this_img_groundtruth_data = img_groundtruth_data(i,:);
        
        [indexs, ~] = get_nn(this_query_X, img_features_data, opts.match.metric);
        
        ap(i) = get_ap(indexs, this_img_query_name, img_features_name, this_img_groundtruth_data, opts);
    end
    mAP = mean(ap);
end

function [indexs, distances] = get_nn(this_query_X, img_features_data, metric)
	% Find the k top indexs and distances of index data vectors from query vector x.
    
    if metric == 1
        % L1 distance (Manhattan distance)
        distances = sum(abs(img_features_data - this_query_X), 2);
        [~, indexs] = sort(distances);
    end
    
    if metric == 2
        % L2 distance (Euclidean distance)
        distances = sum((img_features_data - this_query_X) .^ 2, 2) .^ (1/2);
        [~, indexs] = sort(distances);
    end
    
    if metric == 3
        % Canberra distance
        x = abs(img_features_data - this_query_X);
        y = abs(img_features_data) + abs(this_query_X);
        distances = sum(x ./ ( y + 1e-9), 2);
        [~, indexs] = sort(distances);
    end
    
    if metric == 4
        % Correlation similarity
        x = mean(this_query_X, 2);
        y = mean(img_features_data, 2);
        xy = sum((img_features_data - y) .* (this_query_X - x), 2);
        xy_ = (sum((img_features_data - y) .^ 2, 2) .^ (1/2)) .* (sum((this_query_X - x) .^ 2, 2) .^ (1/2));
        distances = xy ./ (xy_ + 1e-9);
        [~, indexs] = sort(distances, 'descend');
    end
    
    if metric == 5
        % Cosine similarity
        x = sum(img_features_data .* this_query_X, 2);
        y = (sum(img_features_data .^2, 2) .^ (1/2)) .* (sum(this_query_X .^ 2, 2) .^ (1/2));
        distances = x ./ ( y + 1e-9);
        [~, indexs] = sort(distances, 'descend');
    end
    
    if metric == 6
        % Histogram intersection
        
        distances = sum(min(abs(this_query_X), abs(img_features_data)), 2);
        [~, indexs] = sort(distances, 'descend');
    end
    
    if metric == 7
        % Inner product distance
        distances = img_features_data * this_query_X';
        [~, indexs] = sort(distances, 'descend');
    end
end

function ap = get_ap(indexs, this_img_query_name, img_features_name, this_img_groundtruth_data, opts)
    rank_file_name = [this_img_query_name, opts.file.format_mat];   % get the rank name from indexs
    indexs_name = "";
    for i = 1:length(indexs)
        indexs_name(i) = img_features_name(indexs(i));
    end
%     save([opts.match.rank_path, rank_file_name], 'indexs_name');

    rank_set = indexs_name;                     % compute the ap
    good_set = this_img_groundtruth_data{2};
    ok_set = this_img_groundtruth_data{3};
    junk_set = this_img_groundtruth_data{4};
    datasets = this_img_groundtruth_data{5};
       
    gt_num = (length(good_set) + length(ok_set));
    old_recall = 0.0;
    old_precision = 1.0;
    ap = 0.0;
    tp = 0;    
    j = 1;

    if isempty(good_set)
        good_set = "";
    end
    if isempty(ok_set)
        ok_set = "";
    end
    if isempty(junk_set)
        junk_set = "";
    end
    
    for i = 1:length(rank_set)
        if ismember(rank_set(i), junk_set)
            continue;
        end
        if tp == gt_num
            break;
        end        
        if ismember(rank_set(i), good_set) || ismember(rank_set(i), ok_set)
            tp = tp + 1;
        end
        recall = tp / gt_num;
        precision = tp / j;
        
        ap = ap +  (abs(recall - old_recall)) * ((precision + old_precision) / 2.0);

        old_recall = recall;
        old_precision = precision;
        j = j + 1;
    end
end
