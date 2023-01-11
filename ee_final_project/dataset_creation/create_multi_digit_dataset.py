from ee_final_project.dataset_creation.create_multi_digit_dataset import NumberDataset

num_of_digits = 3
train_imgs_to_gen = 120000  # 960000
test_img_to_gen = 30000  # 240000
random_seed = 1

dataset_path = "../../data"

torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

train_data = NumberDataset(
    num_to_generate=train_imgs_to_gen,
    num_of_digits=num_of_digits,
    dataset_path=dataset_path,
).res
train_size = len(train_data)

print(f"Done proccessing training set, got {train_size} numbers")

test_data = NumberDataset(
    num_to_generate=test_img_to_gen,
    num_of_digits=num_of_digits,
    dataset_path=dataset_path,
    train=False,
).res
test_size = len(test_data)
print(f"Done proccessing test set, got {test_size} numbers")

fig2, axes = plt.subplots(3, 3)
fig2.tight_layout()
for i in range(9):
    sub = axes[int(i / 3), i % 3]
    sub.imshow(train_data[i][0][0], cmap="gray", interpolation="none")
    sub.set_title("Ground Truth: {}".format(train_data[i][1]))
    sub.set_xticks([])
    sub.set_yticks([])

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
valloader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)
