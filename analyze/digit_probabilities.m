type = 'normal'; % distribution/attribute of the training dataset - used for the titles of the graphs
cap_type = [upper(type(1)), lower(type(2:end))];
num_of_digits = 4; % number of digits in the training dataset

train_set = csvread('path/to/train_set.csv');
gan_set = csvread('path/to/ocr_results.csv');

train_set_truncated = train_set(1:10000); % truncate the training set to 10,000 samples
gan_set_truncated = gan_set(1:10000); % truncate the synthetic set to 10,000 samples

% Initialize an array to store the counts of each digit (0-9)
train_counts = zeros(1, 10);

% Count the occurrences of each digit
for j = 1:numel(train_set_truncated)
    num = train_set_truncated(j);
    num_str = num2str(num);
    for k = 1:numel(num_str)
        digit = str2double(num_str(k));
        train_counts(digit+1) = train_counts(digit+1) + 1;
    end
end

% Calculate the digit probabilities
train_total_count = sum(train_counts);
train_probabilities = (train_counts * 100) / train_total_count;

% Initialize an array to store the counts of each digit (0-9)
gan_counts = zeros(1, 10);

% Count the occurrences of each digit
for j = 1:numel(gan_set_truncated)
    num = gan_set_truncated(j);
    num_str = num2str(num);
    for k = 1:numel(num_str)
        digit = str2double(num_str(k));
        gan_counts(digit+1) = gan_counts(digit+1) + 1;
    end
end

% Calculate the digit probabilities
gan_total_count = sum(gan_counts);
gan_probabilities = (gan_counts * 100) / gan_total_count;

% Create a histogram of the digit frequencies
figure;
subplot(2, 1, 1);
bar(0:9, train_probabilities);
xlabel('Digit');
ylabel('Probability [%]');
title('Train Set');
grid on;

subplot(2, 1, 2);
bar(0:9, gan_probabilities);
xlabel('Digit');
ylabel('Probability [%]');
title('Synthetic Set');
grid on;
sgtitle(sprintf('Digit Probabilities - %d-Digit, %s Distribution', num_of_digits, cap_type))
