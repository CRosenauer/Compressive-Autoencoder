function showcase(index)
    close all
    
    base = "F:\School\ENSC 424\ML-Project\validation\";

    original = imread(strcat(base, "validation_input\validation_input", int2str(index), ".png"));
    qr256 = imread(strcat(base, "q_r256\validation_output", int2str(index), ".png"));
    qr32 = imread(strcat(base, "q_r32\validation_output", int2str(index), ".png"));
    qr8 = imread(strcat(base, "q_r8\validation_output", int2str(index), ".png"));

    subplot(2, 2, 1)    
    imshow(original)
    title("Original Image")

    subplot(2, 2, 2)
    imshow(qr256)
    title("Quantization Step: 1/256 Compression Ratio: 64%")
    
    subplot(2, 2, 3)
    imshow(qr32)
    title("Quantization Step: 1/32 Compression Ratio: 46%")

    subplot(2, 2, 4)
     imshow(qr8)
    title("Quantization Step: 1/8 Compression Ratio: 31%")

end