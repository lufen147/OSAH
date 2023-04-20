%% 1 load general config
opts = im_config();

A_data = ["Oxford5K", "Paris6K", "Holidays_upright", "Flickr100k"];
B_net = ["vgg16"];
% B_net = ["alexnet", "resnet101", "mobilenetv2"];

%% 2 calculating raw descriptors
for a_i = 1:size(A_data,2)
    for b_i = 1:size(B_net,2)
        dataname = lower(A_data(a_i));
        opts.features.net = lower(B_net(b_i));
        opts.datasets.name = dataname{1};
        opts = im_config(opts);
        
        % % set dataset parameter

        filepatch = opts.datasets.image_path;
        filename = dir([filepatch, '/*.jpg']);
        if isempty(filename)
            filename = dir([filepatch, '/*/*.jpg']);
        end
        [file_num_all, temp] = size(filename);

        % % load pre-train network
        if strcmp(opts.features.net, 'alexnet')
            net = alexnet;
            layer1 = 'pool5';
            dim1 = 256;
            layer2 = 'fc7';
            dim2 = 4096;
            kernel = ones(6);       % kernel type: 1
        end
        if strcmp(opts.features.net, 'vgg16')
            net = vgg16;
            layer1 = 'pool5';
            dim1 = 512;
            layer2 = 'fc7';
            dim2 = 4096;
            kernel = ones(7);       % kernel type: 1
        end
        if strcmp(opts.features.net, 'resnet101')
            net = resnet101;
            layer1 = 'res5c_branch2b';
            dim1 = 512;
            layer2 = 'pool5';
            dim2 = 2048;
            kernel = ones(7);       % kernel type: 1
        end
        if strcmp(opts.features.net, 'mobilenetv2')
            net = mobilenetv2;
            layer1 = 'Conv_1';
            dim1 = 1280;
            layer2 = 'global_average_pooling2d_1';
            dim2 = 1280;
            kernel = ones(7);       % kernel type: 1
        end
        
        % % aggregate and save feature

        for file_num = file_num_all

            MidH = zeros(file_num, dim1);
            HighH = zeros(file_num, dim2);
            FO_H = MidH;
            FOC_H = MidH;
            FS_H = HighH;
            FSW_H = MidH;
            FCW_H = MidH;

            name_list = "";

            % % prepair to start aggregating
            disp(['aggregating from ', char(dataname), ' using ', char(opts.features.net), ' on (', num2str(file_num), '):       ']);

            for i=1:file_num
                
                imdata = imread([filename(i).folder, '/', filename(i).name]);
                
                % % pre-process the size of image
                if size(imdata,3)==1
                    rgb = cat(3,imdata,imdata,imdata);
                    imdata = mat2gray(rgb);
                end
                img = single(imdata);

                [h, w, ~] = size(img);
                
                if(h<384||w<384)
                    scaling = [384 384];
                    img_resize = imresize(img, scaling);
                else
                    img_resize = img;
                end
                if lower(dataname) == "holidays_upright"
                    scaling = 768/min(h,w);
                    img_resize = imresize(img, scaling);
                end

                % % get images name list
                split = strsplit(filename(i).name, {'.'});
                name = split(1);
                name_list(i) = name{1};

                % % get middle-level object features
                X = activations(net, img_resize, layer1, 'OutputAs', 'channels');
                
                FO_X = X;
                FO_H(i,:) = sum(FO_X, [1 2]);

                FOC_X = convn(FO_X, kernel, 'valid');
                FOC_H(i,:) = sum(FOC_X, [1 2]);
                [h, w, k1] = size(FOC_X);

               % % get high-level semantic features

                Y = activations(net, img_resize, layer2, 'OutputAs', 'channels');
                FS_Y = Y;

                FS_Y(FS_Y<0) = 0;
                FS_H(i,:) = sum(FS_Y, [1 2]);

                % % generate the two kernels

                [h, w, k2] = size(FS_Y);
                FS_Y_s = sum(FS_Y .^ 2, 3);
                FS_Y_s = reshape(FS_Y_s, [1, h*w]);
                FS_Y_s = normalize(FS_Y_s, 'norm');
                FS_Y_s = reshape(FS_Y_s, [h, w]);

                FS_Y_c = sum(FS_Y .^ 2, [1 2]);
                FS_Y_c = HDR(FS_Y_c, k1, 1);
                FS_Y_c = normalize(FS_Y_c, 'norm');

                % % get OSAH

                FSW_E = convn(FOC_X, FS_Y_s, 'same');
                FSW_H(i,:) = sum(FSW_E, [1 2]);
                
                FCW_Z_c_map = convn(FOC_X, FS_Y_c, 'valid');
                FCW_T = FSW_E .* FCW_Z_c_map;
                FCW_H(i,:) = sum(FCW_T, [1 2]);

                fprintf(1,'\b\b\b\b\b\b%6d', i);

            end
            %% gather various feature vector
            toc
            
            MHDF2.name = name_list;            
            
            MHDF2.MF = FO_H;
            MHDF2.HF = FS_H;
            MHDF2.CMF = FOC_H;
            MHDF2.DWCF = FSW_H;
            MHDF2.PWCF = FCW_H;
        end

        % save data to .mat

        save([opts.run.data_temp, 'MHDF1_', lower(dataname{1}), '_', opts.features.net{1}], 'MHDF2');
    end
end
disp(datetime);

%% 3 post-process and testing evaluation
opts = im_config();
opts.run.data_temp2 = 'data_temp/';
% opts.run.data_temp2 = 'data_temp_ok/';

A_data = ["Oxford5K", "Paris6K", "Holidays_upright", "oxford105k", "paris106k"];
B_net = ["vgg16"];
% B_net = ["alexnet"];
% B_net = ["resnet101"];
% B_net = ["mobilenetv2"];

C_dim = [64, 128, 512];

for a_i = 1:size(A_data, 2)
    for b_i = 1:size(B_net, 2)
        for c_i = 1:size(C_dim, 2)
            dataname = lower(A_data(a_i));
            opts.features.net = lower(B_net(b_i));
            opts.datasets.name = dataname{1};
            dim = C_dim(c_i);            
            opts = im_config(opts);

            if dataname == "oxford105k"
                temp1 = load([opts.run.data_temp2, 'MHDF1_', 'flickr100k', '_', opts.features.net{1}]);
                temp2 = load([opts.run.data_temp2, 'MHDF1_', 'oxford5k', '_', opts.features.net{1}]);
                MHDF2.name = [temp1.MHDF2.name, temp2.MHDF2.name];
                MHDF2.PWCF = [temp1.MHDF2.PWCF; temp2.MHDF2.PWCF];
                clear temp1 temp2;
            elseif dataname == "paris106k"
                temp1 = load([opts.run.data_temp2, 'MHDF1_', 'flickr100k', '_', opts.features.net{1}]);
                temp2 = load([opts.run.data_temp2, 'MHDF1_', 'paris6k', '_', opts.features.net{1}]);
                MHDF2.name = [temp1.MHDF2.name, temp2.MHDF2.name];
                MHDF2.PWCF = [temp1.MHDF2.PWCF; temp2.MHDF2.PWCF];
                clear temp1 temp2;
            else
                load([opts.run.data_temp2, 'MHDF1_', lower(dataname{1}), '_', opts.features.net{1}, '.mat']);
            end

            MHDF3.name = MHDF2.name;
%                 MHDF = MHDF2.MF;
%                 MHDF = MHDF2.HF;
%                 MHDF = MHDF2.CMF;
%                 MHDF = MHDF2.DWCF;
            MHDF = MHDF2.PWCF;

            train = MHDF;                

            MHDF = normalize(MHDF, 2, 'norm');
            MHDF = PCA_whitening(MHDF, train, dim);

            MHDF3.MHDF = MHDF;

            save([opts.run.data_temp, 'MHDF3_', lower(dataname{1})], 'MHDF3');

            % % evaluation
            opts.features.dimension = dim;            
            im_evaluation(opts);
        end
    end
end
disp(datetime);