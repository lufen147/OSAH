function HDR_ = HDR(data, dim, scale)
    % method of Histogram dimensionality reduction
    % input: data, n * dimensionalilty matrix
    % output: data after HDR, n * dim matrix
    % Authod: GuangHai Liu, Fen Lu re-organize.

    if nargin == 2
        scale = 1;
    end
    ndim = ndims(data);
    if ndim == 2
        [rows, cols] = size(data);
        if cols>=dim
            HDR_ = zeros(rows, dim);
            SN = floor(cols / dim);
            k = 1;
            data = data .^ scale;
            for i = 1:SN:cols
                if k == dim
                    x = data(:,i:end);
                    HDR_(:, k) = func(x);
                    break;
                else
                    x = data(:,i:i+SN-1);
                    HDR_(:, k) = func(x);
                end
                k = k + 1;
            end
        else
            HDR_(:, 1:cols) = data(:,:);
            HDR_(:, cols+1:dim) = 1;
        end
    end
    
    if ndim == 3
        [rows, cols, channel] = size(data);
        HDR_ = zeros(rows, cols, dim);
        if channel >= dim
            SN = floor(channel / dim);
            k = 1;
            data = data .^ scale;
            for i = 1:SN:channel
                if k == dim
                    x = data(:, :, i:end);
                    HDR_(:, :, k) = sum(x, 3);
                    break;
                else
                    x = data(:, :, i:i+SN-1);
                    HDR_(:, :, k) = sum(x, 3);
                end
                k = k + 1;
            end
        else
            HDR_(:, :, 1:channel) = data(:,:,:);
            HDR_(:, :, channel+1:dim) = 1;
        end
    end
end

function z = func(x)
    z = sum(x, 2);
end
