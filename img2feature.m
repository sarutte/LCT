function [feature]=img2feature(img) 
    %Viola-Jones face detection and crop only face part
    faceDetector=vision.CascadeObjectDetector('FrontalFaceCART');
    BB=step(faceDetector,img);
    face=imcrop(img,BB(1,:));
    
    %resize to 128x128
    fac=imresize(face,[128,128]);
    
    %crop necessary part of image
    fac=fac(20:size(fac,1)-10,15:size(fac,2)-15);
    fac=imresize(face,[128,128]);
    
    %histogram equalization
    fac = histeq(fac);
    
    fmean=[];
    fnorm=[];
    fsd=[];
    
    %separate into sub-regions
    for x=16:16:128
        for y=16:16:128
            image=fac(x-15:x,y-15:y);
            
            %Curvelet transform
            C = fdct_wrapping(image,0,1,4,8);
            temp1=reshape(C{1,1}{1,1},1,[]);
            
            %calculate each curvelet coefficient of sub-regions into mean, norm, and standard deviation
            fnorm=[fmean wentropy(temp1,'norm',2)];
            fmean=[fnorm norm(temp1)/size(temp1,2)];
            fsd=[fsd std(temp1)];           
        end
    end
    %statistical feature
    feature=[fmean fnorm fsd];
end