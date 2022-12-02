% This example shows how to call Python® from MATLAB® to compare PyTorch®
% image classification models, and then import the fastest PyTorch model
% into MATLAB.

%% Python Environment

% Set up the Python environment.
!python -m venv env
% For Windows
!env\Scripts\pip.exe install -r requirements.txt

% Set up the Python interpreter for MATLAB.
pe = pyenv(ExecutionMode="OutOfProcess",Version="env\Scripts\python.exe");

%% PyTorch Models
% Get three pretrained PyTorch models (VGG, MobileNet v2, and MNASNet) from
% the torchvision library.
model1 = py.torchvision.models.vgg16(pretrained=true);
model2 = py.torchvision.models.mobilenet_v2(pretrained=true);
model3 = py.torchvision.models.mnasnet1_0(pretrained=true);

%% Preprocess Image
% Read the image you want to classify. Show the image.
imgOriginal = imread("banana.png");
imshow(imgOriginal)
 
% Resize the image to the input size of the network.
InputSize = [224 224 3];
img = imresize(imgOriginal,InputSize(1:2));

% You must preprocess the image in the same way as the training data.
% Rescale the image. Then, normalize the image by subtracting the training
% images mean and dividing by the training images standard deviation.
imgProcessed = rescale(img,0,1);

meanIm = [0.485 0.456 0.406];
stdIm = [0.229 0.224 0.225];
imgProcessed = (imgProcessed - reshape(meanIm,[1 1 3]))./reshape(stdIm,[1 1 3]);

% Permute the image data from the Deep Learning Toolbox dimension ordering
% (HWCN) to the PyTorch dimension ordering (NCHW).
imgForTorch = permute(imgProcessed,[4 3 1 2]);

%% Classify Image with Co-Execution
% Check that the PyTorch models work as expected by classifying an image.
% Call Python from MATLAB to predict the label.

% Get the class names from squeezenet, which is also trained with
% ImageNet images (same as the torchvision models).

squeezeNet = squeezenet;
ClassNames = squeezeNet.Layers(end).Classes;
 
% Convert the image to a tensor in order to classify the image with a
% PyTorch model.
X = py.numpy.asarray(imgForTorch);
X_torch = py.torch.from_numpy(X).float();
 
% Classify the image with co-execution using the MNASNet model. The model
% predicts the correct label.
y_val = model1(X_torch);
predicted = py.torch.argmax(y_val);
label = ClassNames(double(predicted.tolist)+1);

%% Compare PyTorch Models
% Find the fastest PyTorch model by calling Python from MATLAB. Predict the 
% image classification label multiple times for each of the PyTorch models.
N = 30;

for i = 1:N
    tic
    model1(X_torch);
    T(i) = toc;
end
mean(T)

for i = 1:N
    tic
    model2(X_torch);
    T(i) = toc;
end
mean(T)

for i = 1:N
    tic
    model3(X_torch);
    T(i) = toc;
end
mean(T)

%% Save PyTorch Model
% Save the fastest PyTorch model, among the three models compared. Then,
% trace the model.
pyrun("import torch;X_rnd = torch.rand(1,3,224,224)")
pyrun("traced_model = torch.jit.trace(model3.forward,X_rnd)",model3=model3)
pyrun("traced_model.save('traced_mnasnet1_0.pt')")

%% Import PyTorch Model
% Import the MNASNet model by using the importNetworkFromPyTorch function.
net = importNetworkFromPyTorch("traced_mnasnet1_0.pt");

% Create an image input layer. Then, add the image input layer to the
% imported network and initialize the network.
inputLayer = imageInputLayer(InputSize,Normalization="none");
net = addInputLayer(net,inputLayer,Initialize=true);

% Analyze the imported network. Observe that there are no warnings or
% errors, which means that the network is ready to use.
analyzeNetwork(net)

%% Classify Image in MATLAB
% Convert the image to a dlarray object. Format the image with dimensions
% "SSCB" (spatial, spatial, channel, batch).
Img_dlarray = dlarray(single(imgProcessed),"SSCB");

% Classify the image and find the predicted label.
prob = predict(net,Img_dlarray);
[~,label_ind] = max(prob);

% Show the image with the classification label.
imshow(imgOriginal)
title(ClassNames(label_ind),FontSize=18)

%%
% Copyright 2022, The MathWorks, Inc. Copyright 2022, The MathWorks, Inc.
