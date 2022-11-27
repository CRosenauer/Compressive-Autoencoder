function imageSampler(folderpath, sampleSize)
ds = imageDatastore(folderpath, "IncludeSubfolders",true);

savePath = folderpath + "\samples\";
if ~exist(savePath, 'dir')
  mkdir(savePath);
end

while hasdata(ds) %change to while hasdata(ds)
    [img, info] = read(ds);
    if(mod(size(img, 1), sampleSize) ~= 0)
        disp("can't sample, size invalid: " + sampleSize);
        return;
    end

    %extract base file name
    baseFilePath = info.Filename;

    %MAY NEED TO CHANGE DIRECTION OF SLASH
    myFolders = split(baseFilePath,"\");
    baseFile = split(myFolders(size(myFolders, 1)), ".");

    numOfSamples1D = size(img, 1)/sampleSize;
    for j = 0:numOfSamples1D-1
        xStart = j*sampleSize +1; %Matlab arrays start at 1
        xEnd = (j+1)*sampleSize;

        for k = 0:numOfSamples1D-1
            yStart = k*sampleSize +1;
            yEnd = (k+1)*sampleSize;

            sample = img(xStart:xEnd, yStart:yEnd, :);
            sampleName = baseFile(1) + "_" + j + k + "." + "png";
            imwrite(sample, savePath + sampleName);
        end
    end

end
end