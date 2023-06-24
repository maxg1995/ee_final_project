type = 'normal'; % distribution/attribute of the training dataset - used for the titles of the graphs
cap_type = [upper(type(1)), lower(type(2:end))];
num_of_digits = 4; % number of digits in the training dataset

train_set = csvread('path/to/train_set.csv');
gan_set = csvread('path/to/ocr_results.csv');

i_values = 5:5:1000; % number of samples to test
p_values = zeros(size(i_values)); % p-values for each 'i' value
h_values = zeros(size(i_values)); % h-values for each 'i' value

% Perform the KS test and store p-values and h-values for each 'i' value
for idx = 1:numel(i_values)
    [h_values(idx), p_values(idx)] = kstest2(train_set(1:10000), gan_set(1:i_values(idx)));
end

% Plot p-values as a function of i
figure;
subplot(2, 1, 2);
plot(i_values, p_values, 'b.-');
xlabel('Samples');
ylabel('P-Value');
title('P-Value as a Function of Number of Samples Tested');
grid on;

% Plot h-values as a function of i
subplot(2, 1, 1);
plot(i_values, h_values, 'r.-');
xlabel('Samples');
ylabel('Decision');
title('Decision as a Function of Number of Samples Tested');
grid on;
sgtitle(sprintf('Two-Sample Kolmogorov-Smirnov Test\n%d-Digit, %s Distribution', num_of_digits, cap_type));
