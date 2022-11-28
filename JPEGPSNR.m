function [averageCompressionRatio, averagePSNR, perfectMatches] = JPEGPSNR(inputPath)
    averageCompressionRatio = zeros(10, 1);
    averagePSNR = zeros(10, 1);
    perfectMatches = zeros(10, 1);

    input = imageDatastore(inputPath);

    numImages = length(input.Files);

    while hasdata(input)
        [inputImage, inputInfo] = read(input);

        for ii = 1:10
            [output, cdv, ratio] = jpeg_computing(inputImage, ii * 10);

            averageCompressionRatio(ii) = averageCompressionRatio(ii) + ratio;

            itrPSNR = PSNR(inputImage, output);

            if isinf(itrPSNR)
                perfectMatches(ii) = perfectMatches(ii) + 1;
            else
                averagePSNR(ii) = averagePSNR(ii) + itrPSNR;
            end

            
        end
    end

    averagePSNR = averagePSNR ./ (ones(size(perfectMatches)) * (numImages - perfectMatches));
    averageCompressionRatio = averageCompressionRatio / numImages;

end

function out = PSNR(original, compressed)
    dataSize = size(original);

    squaredError = (original - compressed) .^ 2;
    sumOfSquarredError = sum(squaredError, 'all');
    meanSquaredError = sumOfSquarredError;

    for ii = 1:length(dataSize)
        meanSquaredError = meanSquaredError / dataSize(ii);
    end

    out = 10 * log(255 * 255 / meanSquaredError);
end