function comparePSNR(inputPath, outputPath, ratio)
    input = imageDatastore(inputPath);
    output = imageDatastore(outputPath);

    if length(input.Files) ~= length(output.Files)
        disp("Number of input files does not equal the number of output files.");
        return;
    end

    numImages = length(input.Files);

    averagePSNR = 0;
    perfectMatches = 0;

    while hasdata(input) && hasdata(output)
        [inputImage, inputInfo] = read(input);
        [outputImage, outputInfo] = read(output);

        psnr = PSNR(inputImage, outputImage);

        if isinf(psnr) 
            perfectMatches = perfectMatches + 1;
            numImages = numImages - 1;
        else
            averagePSNR = averagePSNR + psnr;
        end
        
    end

    averagePSNR = averagePSNR / numImages;
    disp("Average Compression Ratio:");
    disp(ratio);
    disp("Average PSNR (dB):");
    disp(averagePSNR);
    disp("Perfect Matches:");
    disp(perfectMatches);
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