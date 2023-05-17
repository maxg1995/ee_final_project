import torch
from torch.autograd import Variable

from ee_final_project.dataset_creation.dataset_base import MNISTNumbersDataset
from ee_final_project.env import BATCH_SIZE, DIGIT_DIR, EPOCHS, NUM_OF_DIGITS, device

from .gan_base import Discriminator, Generator


def D_train(
    x,
    G: Generator,
    D: Discriminator,
    mnist_dim: int,
    z_dim: int,
    criterion,
    D_optimizer,
):
    D.zero_grad()

    # Train discriminator on real
    x_real, y_real = x.view(-1, mnist_dim), torch.ones(BATCH_SIZE, 1)
    x_real, y_real = Variable(x_real.to(device)), Variable(y_real.to(device))

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    # D_real_score = D_output

    # Train discriminator on fake
    z = Variable(torch.randn(BATCH_SIZE, z_dim).to(device))
    x_fake, y_fake = G(z), Variable(torch.zeros(BATCH_SIZE, 1).to(device))

    D_output = D(x_fake)
    D_fake_loss = criterion(D_output, y_fake)
    # D_fake_score = D_output

    # Gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()

    return D_loss.data.item()


# Train the generator
def G_train(
    G: Generator,
    D: Discriminator,
    z_dim: int,
    criterion,
    G_optimizer,
):
    G.zero_grad()

    z = Variable(torch.randn(BATCH_SIZE, z_dim).to(device))
    y = Variable(torch.ones(BATCH_SIZE, 1).to(device))

    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # Gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()

    return G_loss.data.item()


def train_gan(
    training_dataset: MNISTNumbersDataset,
    test_dataset: MNISTNumbersDataset,
) -> tuple[Generator, Discriminator]:
    print("starting training")
    train_loader = torch.utils.data.DataLoader(
        training_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    # test_loader = torch.utils.data.DataLoader(
    #     test_data, batch_size=batch_size, shuffle=True
    # )

    # Build network
    z_dim = 100
    mnist_dim = training_dataset[0][0].shape[0] * training_dataset[0][0].shape[1]

    G = Generator(g_input_dim=z_dim, g_output_dim=mnist_dim).to(device)
    D = Discriminator(mnist_dim).to(device)

    # Training
    epochs = EPOCHS

    for epoch in range(1, epochs + 1):
        D_losses, G_losses = [], []
        for batch_idx, (x, _) in enumerate(train_loader):
            D_losses.append(
                D_train(
                    x,
                    G,
                    D,
                    mnist_dim,
                    z_dim,
                    D.criterion,
                    D.optimizer,
                )
            )
            G_losses.append(
                G_train(
                    G,
                    D,
                    z_dim,
                    G.criterion,
                    G.optimizer,
                )
            )

        print(
            "[%d/%d]: loss_d: %.3f, loss_g: %.3f"
            % (
                (epoch),
                epochs,
                torch.mean(torch.FloatTensor(D_losses)),
                torch.mean(torch.FloatTensor(G_losses)),
            )
        )

    torch.save(
        G,
        f"{DIGIT_DIR}/mnist_{NUM_OF_DIGITS}_digit_gan_generator_model",
    )
    torch.save(
        D,
        f"{DIGIT_DIR}/mnist_{NUM_OF_DIGITS}_digit_gan_discriminator_model",
    )
    return G, D
