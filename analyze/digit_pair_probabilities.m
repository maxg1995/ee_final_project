type = 'normal'; % distribution/attribute of the training dataset - used for the titles of the graphs
cap_type = [upper(type(1)), lower(type(2:end))];
num_of_digits = 4; % number of digits in the training dataset

train_set = csvread('path/to/train_set.csv');
gan_set = csvread('path/to/ocr_results.csv');

train_set_truncated = train_set(1:10000); % truncate the training set to 10,000 samples
gan_set_truncated = gan_set(1:10000); % truncate the synthetic set to 10,000 samples

% Initialize a 10x10 matrix to store the counts of digit pairs
train_counts = zeros(10, 10);

% Count the occurrences of digit pairs
for j = 1:numel(train_set_truncated)
    num = train_set_truncated(j);
    num_str = num2str(num);
    for k = 1:numel(num_str)-1
        curr_digit = str2double(num_str(k));
        next_digit = str2double(num_str(k+1));
        train_counts(curr_digit+1, next_digit+1) = train_counts(curr_digit+1, next_digit+1) + 1;
    end
end

% Calculate the probabilities of digit pairs
train_total_pairs = sum(train_counts(:));
train_probabilities = (train_counts * 100) / train_total_pairs;

% Initialize a 10x10 matrix to store the counts of digit pairs
gan_counts = zeros(10, 10);

% Count the occurrences of digit pairs
for j = 1:numel(gan_set_truncated)
    num = gan_set_truncated(j);
    num_str = num2str(num);
    for k = 1:numel(num_str)-1
        curr_digit = str2double(num_str(k));
        next_digit = str2double(num_str(k+1));
        gan_counts(curr_digit+1, next_digit+1) = gan_counts(curr_digit+1, next_digit+1) + 1;
    end
end

% Calculate the probabilities of digit pairs
gan_total_pairs = sum(gan_counts(:));
gan_probabilities = (gan_counts * 100) / gan_total_pairs;

% Create a heatmap of the digit pair probabilities
figure;
subplot(1, 2, 1);
imagesc(train_probabilities);
xlabel('Next Digit');
ylabel('Current Digit');
title('Train Set');
colorbar;
axis square;

% Adjust the x-axis and y-axis tick labels
xticks(1:10);
xticklabels(0:9);
yticks(1:10);
yticklabels(0:9);

caxis([0.5, 1.9])
colorbarLabel = 'Probability [%]';
cb = colorbar;
ylabel(cb, colorbarLabel);

subplot(1, 2, 2);
imagesc(gan_probabilities);
xlabel('Next Digit');
ylabel('Current Digit');
title('Synthetic Set');
colorbar;
axis square;

% Adjust the x-axis and y-axis tick labels
xticks(1:10);
xticklabels(0:9);
yticks(1:10);
yticklabels(0:9);

caxis([0.5, 1.9])
colorbarLabel = 'Probability [%]';
cb = colorbar;
ylabel(cb, colorbarLabel);

figureHandle = gcf;
figureHandle.OuterPosition(2) = figureHandle.OuterPosition(2) - 100;
sgtitle(sprintf('Digit Pair Probabilities - %d-Digit, %s Distribution', num_of_digits, cap_type));
