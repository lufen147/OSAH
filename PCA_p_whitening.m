function features_pca = PCA_p_whitening(features, dim)
    
    features_data = normalize(features, 2, 'norm');
    features_data(isnan(features_data)) = 0;

    x_train = normalize(features_data, 2, 'norm');                   % PCA
    x_train(isnan(x_train)) = 0;
    x_train = x_train';
    mu = mean(x_train, 1);
    x_train = x_train - mu;
    sigma = x_train * x_train' ./ size(x_train, 2);
    [U, S, ~] = svd(sigma);
    
    x_test = features_data;     % PCA apply
    x_test = x_test';
    x_test = x_test - mean(x_test, 1);
    xRot = U' * x_test;
    
    epsilon = 1e-5;             % PCA-p-whitening apply
    p = 3;
    xPCAWhite = diag(1 ./ ((diag(S) + epsilon).^(1/p))) * xRot;     
    features_data = xPCAWhite(1:dim, :)';

    features_pca = normalize(features_data, 2, 'norm');

end