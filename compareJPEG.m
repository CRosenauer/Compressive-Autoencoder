function compareJPEG(index)
    close all
    
    base = "F:\School\ENSC 424\ML-Project\validation\";

    original = imread(strcat(base, "validation_input\validation_input", int2str(index), ".png"));
    qr32 = imread(strcat(base, "q_r32\validation_output", int2str(index), ".png"));
    qr8 = imread(strcat(base, "q_r8\validation_output", int2str(index), ".png"));
    qr2 = imread(strcat(base, "q_r2\validation_output", int2str(index), ".png"));
    jpeg100 = imread(strcat(base, "jpeg100\validation_input", int2str(index), ".jpg"));
    jpeg95 = imread(strcat(base, "jpeg95\validation_input", int2str(index), ".jpg"));
    jpeg80 = imread(strcat(base, "jpeg80\validation_input", int2str(index), ".jpg"));

    subplot(2, 5, [1 2 6 7])    
    imshow(original)
    title("Original Image")
    
    subplot(2, 5, 3)
    imshow(qr32)
    title("Autoencoder Compression Ratio: 46%")

    subplot(2, 5, 4)
     imshow(qr8)
    title("Autoencoder Compression Ratio: 31%")

    subplot(2, 5, 5)
    imshow(qr2)
    title("Autoencoder Compression Ratio: 16%")

    subplot(2, 5, 8)
    imshow(jpeg100)
    title("JPEG 4:2:2 Compression Ratio: 41%")

    subplot(2, 5, 9)
    imshow(jpeg95)
    title("JPEG 4:2:2 Compression Ratio: 25%")

    subplot(2, 5, 10)
    imshow(jpeg80)
    title("JPEG 4:2:2 Compression Ratio: 14%")
end