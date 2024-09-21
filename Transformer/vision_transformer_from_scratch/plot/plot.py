import matplotlib.pyplot as plt


def plot_per_epochs(train_losses, test_losses, accuracies, epochs):
    # Create two subplots of train/test losses and accuracies
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot train and test losses
    ax1.plot(train_losses, label="Train loss")
    ax1.plot(test_losses, label="Test loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    # Plot accuracies
    ax2.plot(accuracies)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")

    # Save the plot with the number of epochs in the file name
    plt.savefig(f"./save_epochs/metrics_{epochs}_epochs.png")
