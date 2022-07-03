%% This scrpt will create a simple image classifiation in matlab or art
% images

%% Clear environment
clearvars; close all; clc;


%% Prepare image data
% get only the paintings and store them
% first load in table data
artInfo = readtable("/artwork_dataset.csv");
% make for loop and filter what data we want
% get only the paintings
searchPattern = ["oil", "canvas", "painting"];
searchIdx = contains(artInfo.pictureData, searchPattern);
paintingArtInfo = artInfo(searchIdx, :);

% iterate through the list and then we will load in the images
% we will rescale and make it gray color value
% then save the image with the id name in a folder containing the artis
% name (label)
dataFolderName = "artwork100x100_rgb";
mkdir(dataFolderName);
imageSize = [100, 100];
for iImage = 1:1:height(paintingArtInfo)

    try
        artistName = split(paintingArtInfo.artist{iImage}, ',');
        artistName = artistName{1};

        imageData = imread("artwork/" +...
            num2str(paintingArtInfo.ID(iImage)) +...
            ".jpg");

        % check size of images
        if size(imageData, 3) == 3

            imageData = imresize(imageData, imageSize);

            if not(exist(dataFolderName + "/" + artistName))
                mkdir(dataFolderName + "/" + artistName);
            end

            cd(dataFolderName + "/" + artistName);

            imwrite(imageData, num2str(paintingArtInfo.ID(iImage)) + ".png");

            cd("../");
            cd("../");

        else

            warning('ImageSize doesnt fit!');

        end

    catch ErrorMessage

        warning(ErrorMessage.message);

    end

    disp(iImage/height(paintingArtInfo) * 100);

end


%% check image sizes
checkImageInfo = table();
checkImageInfo.rgb = cell(height(paintingArtInfo), 1);

folderList = dir("/Users/hannessuhr/Documents/MATLAB/ARTNet/" + ...
    "CompleteArtworkDataset/artwork227x227_rgb");

counter = 1;
for iImageFolder = 3:1:length(folderList)

    folderName = "/Users/hannessuhr/Documents/MATLAB/ARTNet/" + ...
        "CompleteArtworkDataset/artwork227x227_rgb/" + ...
        folderList(iImageFolder).name;

    imageList = dir(folderName);

    if length(imageList) > 2

        for iImage = 3:1:length(imageList)

            imageName = imageList(iImage).folder + "/" +...
                imageList(iImage).name;

            imageData = imread(imageName);
            checkImageInfo.rgb{counter} = size(imageData);
            counter = counter + 1;
        end
    end

end

%% reduce dataset size for test training
% take only labels with more than x images
% go through all folders check if image number is higher or equal to 5
% copy the folder and images over to new folder
minImageNumber = 60;

folderList = dir("/Users/hannessuhr/Documents/MATLAB/ARTNet/" + ...
    "CompleteArtworkDataset/artwork100x100_rgb");

mkdir("/Users/hannessuhr/Documents/MATLAB/ARTNet/" + ...
    "CompleteArtworkDataset/artwork100x100_rgb_small");

for iFolder = 3:1:length(folderList)

    folderName = folderList(iFolder).folder + "/" + ...
        folderList(iFolder).name;

    imageList = dir(folderName);
    if length(imageList) >= 3 + minImageNumber
        copyfile(folderName,...
            "/Users/hannessuhr/Documents/MATLAB/ARTNet/" + ...
            "CompleteArtworkDataset/artwork100x100_rgb_small/" +...
            folderList(iFolder).name);
    end
end

%% Load image data store
imageDatasetPath = "/Users/hannessuhr/Documents/MATLAB/ARTNet/" + ...
    "CompleteArtworkDataset/artwork100x100_rgb_small";

imds = imageDatastore(imageDatasetPath,...
    'LabelSource','foldername',...
    "FileExtensions", ".png",...
    "IncludeSubfolders", true);

%% Apply image augmentation
imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-5,5], ...
    'RandXTranslation',[-5 5], ...
    'RandYTranslation',[-5 5],...
    "RandScale", [0.8, 1.2],...
    "RandXShear", [-5, 5],...
    "RandYShear", [-5, 5]);

augImds = augmentedImageDatastore([100, 100, 3], imds,...
    "DataAugmentation", imageAugmenter);


%% Partition image data store

[imdsTrain, imdsValidation] =...
    splitEachLabel(imds, 0.8, 'randomize');
labelCount = countEachLabel(imds)

%
% % take only a subset of the image data store for testing maybe only 1000
% % images
% numberOfImages = 10000;
% imageIdx = randperm(numberOfImages);
% imds = subset(imds, imageIdx);
% countEachLabel(imds)

%% Define my own network architecture

inputSize = [100, 100, 3];

layers = [
    imageInputLayer(inputSize)
    convolution2dLayer(10, 30)
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3, 30)
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3, 30)
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(2, 30)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(height(labelCount))
    reluLayer
    fullyConnectedLayer(height(labelCount))
    softmaxLayer
    classificationLayer];


%% load in squeeze net and modify for our use case

net = squeezenet();
lgraph = layerGraph(net);

newConvLayer =  convolution2dLayer([1, 1], numClasses,...
    'WeightLearnRateFactor', 1,...
    'BiasLearnRateFactor', 1,...
    "Name",'new_conv');

lgraph = replaceLayer(lgraph, 'conv10', newConvLayer);

newClassificatonLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph, 'ClassificationLayer_predictions',...
    newClassificatonLayer);

% %% load a network with resnet architecture
% inputSize = [227, 227, 3];
% lgraph = resnetLayers(inputSize, numClasses);
%% Set training options

options = trainingOptions('sgdm', ...
    'MaxEpochs', 100, ...
    'ValidationFrequency', 30, ...
    "ValidationData",imdsValidation,...
    'Verbose', true, ...
    'Plots','training-progress',...
    "InitialLearnRate", 1e-3,...
    "MiniBatchSize", 32);

net = trainNetwork(augImds, layers, options);


%% check out some layers of the network
% close all;
%
% numberOfChannels = 30;
% dreamImage = deepDreamImage(net, 'conv_1', 1:1:numberOfChannels,...
%     'Verbose',0, 'NumIterations', 20);
%
% figure;
% montage(dreamImage);
%
% numberOfChannels = 30;
% dreamImage = deepDreamImage(net, 'conv_2', 1:1:numberOfChannels,...
%     'Verbose',0, 'NumIterations', 20);
%
% figure;
% montage(dreamImage);

numberOfChannels = height(labelCount);
dreamImage = deepDreamImage(net, 'fc', 1:55,...
    'Verbose',0, 'NumIterations', 100);

figure;
% imshow(dreamImage)
montage(dreamImage);


%% test the network with some examples

close all;

numberOfImages = 16;
testImageIdx = randperm(length(imds.Files), numberOfImages);
imdsTestCase = subset(imds, testImageIdx);
predictions = predict(net, imdsTestCase);

[confidence, predictionIdx] = max(predictions, [], 2);
predictionsLabels = predict(net, imdsTestCase,...
    "ReturnCategorical", true);

figure("Units", "normalized",...
    "Position", [0.1, 0.1, 0.6, 0.6]);
tiledlayout("flow");

for iImage = 1:1:length(testImageIdx)

    testImage = readimage(imdsTestCase, iImage);
    nexttile;
    imshow(testImage);
    title({sprintf("Real %s", imdsTestCase.Labels(iImage)),...
        sprintf("Predict %s", predictionsLabels(iImage)),...
        sprintf("Confidence %.3f", confidence(iImage))});

end

%%

artInfo = readtable("/Users/hannessuhr/Documents/MATLAB/ARTNet/" + ...
    "CompleteArtworkDataset/artwork_dataset.csv");
imageDatasetPath = "/Users/hannessuhr/Documents/MATLAB/ARTNet/" + ...
    "CompleteArtworkDataset/artwork100x100_rgb_small";

imds = imageDatastore(imageDatasetPath,...
    'LabelSource','foldername',...
    "FileExtensions", ".png",...
    "IncludeSubfolders", true);

% delete the artist info db entries which are not in the imds database
% fetch all numbers of image names in the imds db
imdsNumberList = cellfun(@(x) split(x,'/'), imds.Files, ...
    'UniformOutput',false);
imdsNumberList = cellfun(@(x) x{end}, imdsNumberList, ...
    'UniformOutput',false);
imdsNumberList = cellfun(@(x) str2double(x(1:end-4)), ...
    imdsNumberList, 'UniformOutput', false);
imdsNumberList = [imdsNumberList{:}];

artInfoFiltered = [];
for iID = imdsNumberList
    artInfoFiltered = [artInfoFiltered; artInfo(artInfo.ID == iID, :)];
end


% load CNN
load("/Users/hannessuhr/Documents/MATLAB/ARTNet/" + ...
    "CompleteArtworkDataset/CNN_96.mat");




%% build animation with neural net for github
% randomly select images from the imds database
% display the image name and artist on the left
% display the actual image int the middle
% display the guessed artist with confidence on the right

close all;

artnetGui = uifigure("Name", "ARTNet Examples", ...
    'Units','pixels',...
    'Position', [100, 100, 800, 500]);

guiFontSize = 20;

uiGrid = uigridlayout(artnetGui, [4, 3]);
uiGrid.ColumnWidth = {'1x', '2x', '1x'};
uiGrid.RowHeight = {'1x', '1x', '1x', '1x'};

imageAxis = uiaxes(uiGrid);
imageAxis.Layout.Row = [1, 4];
imageAxis.Layout.Column = 2;

imageNameLabel = uilabel(uiGrid, ...
    'Text', 'Image Title', ...
    'WordWrap','on',...
    'VerticalAlignment','center', ...
    'HorizontalAlignment','center',...
    'FontSize', guiFontSize);
imageNameLabel.Layout.Row = 2;
imageNameLabel.Layout.Column = 1;

imageArtistLabel = uilabel(uiGrid, ...
    'Text', 'Artist Name', ...
    'WordWrap','on',...
    'VerticalAlignment','center', ...
    'HorizontalAlignment','center',...
    'FontSize', guiFontSize);
imageArtistLabel.Layout.Row = 4;
imageArtistLabel.Layout.Column = 1;

predictedArtistName = uilabel(uiGrid, ...
    'Text', 'Predicted Artist Name', ...
    'WordWrap','on',...
    'VerticalAlignment','center', ...
    'HorizontalAlignment','center',...
    'FontSize', guiFontSize);

predictedArtistName.Layout.Row = 2;
predictedArtistName.Layout.Column = 3;

predictionConfidence = uilabel(uiGrid, ...
    'Text', 'Prediction Confidence', ...
    'WordWrap','on',...
    'VerticalAlignment','center', ...
    'HorizontalAlignment','center',...
    'FontSize', guiFontSize);
predictionConfidence.Layout.Row = 4;
predictionConfidence.Layout.Column = 3;

% iterate over the examples
pause(5)

while 1

    % get random art piece
    randID = randi([1, height(artInfoFiltered)], 1, 1);
    artTitle = artInfoFiltered.title(randID);
    artistName = artInfoFiltered.artist(randID);
    imageData = imds.readimage(randID);

    % show image of art piece
    imshow(imageData, 'Parent', imageAxis, 'InitialMagnification','fit');
    axis equal;

    % display additional information about art piece
    imageNameLabel.Text = "Image Name: " +...
        artInfoFiltered.title(randID);
    imageArtistLabel.Text = "Artist Name: " +...
        artInfoFiltered.artist(randID);

    % predict artist name
    imdsTestCase = subset(imds, randID);
    predictions = predict(net, imdsTestCase);
    [confidence, predictionIdx] = max(predictions, [], 2);
    predictionsLabels = predict(net, imdsTestCase,...
        "ReturnCategorical", true);

    % display prediction of artist name
    predictedArtistName.Text = "Predicted Artist Name: " +...
        char(predictionsLabels);
    predictionConfidence.Text = "Confidence: " +...
        num2str(confidence);

    % pause some second
    pause(3);
end


%% Convert the movie of the image artnet test


gifFromMovie('mov_1.mov', 'gif_1.gif', 'rate', 5);




