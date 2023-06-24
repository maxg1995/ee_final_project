type = 'normal'; % distribution/attribute of the training dataset - used for the titles of the graphs
cap_type = [upper(type(1)), lower(type(2:end))];
num_of_digits = 4; % number of digits in the training dataset

train_set = csvread('path/to/train_set.csv');
gan_set = csvread('path/to/ocr_results.csv');

xlimit=(10^(num_of_digits-1));

figure;
subplot(2,1,2);
hold on;
histogram(gan_set, 'Normalization','pdf', 'NumBins', 800);
[f,xi] = ksdensity(gan_set);
plot(xi, f, 'LineWidth', 1);
hold off;
if strcmp(type, 'normal')
    xlim([0 2*xlimit]);
end

xlabel('Number');
ylabel('Probability');
legend('Normalized PDF Histogram', 'Estimated PDF')
title('Synthetic Set');

subplot(2,1,1);
hold on;
histogram(train_set, 'Normalization','pdf', 'NumBins', 75);
[f,xi] = ksdensity(train_set);
plot(xi, f, 'LineWidth', 1);
hold off
if strcmp(type, 'normal')
    xlim([0 2*xlimit]);
end

xlabel('Number');
ylabel('Probability');
legend('Normalized PDF Histogram', 'Estimated PDF')
title('Train Set');
sgtitle(sprintf('Probability Density - %d-Digit, %s Distribution', num_of_digits, cap_type));
