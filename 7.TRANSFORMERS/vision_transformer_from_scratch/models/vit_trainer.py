import torch
from models import model
from torch import nn, optim
import os
import json
from plot.plot import plot_per_epochs

class Vison_Transformer_Trainer:
    """
    The simple trainer.
    """

    def __init__(self, args):
        self.args = args
 

    def _build_model(self):

        self.model = model.ViTForClassfication(self.args)

        self.optimizer = optim.AdamW(self.model.parameters(), lr= self.args.lr, weight_decay=1e-2)
        self.loss_fn =  nn.CrossEntropyLoss()
        self.exp_name = 'VIT_Training_CIFAR10'
        self.device = self.args.device



    def train(self, trainloader, testloader, epochs, save_model_every_n_epochs=0):
        """
        Train the model for the specified number of epochs.
        """

        self._build_model()

        # Keep track of the losses and accuracies
        train_losses, test_losses, accuracies = [], [], []
        # Train the model
        for i in range(epochs):

            train_loss = self.train_epoch(trainloader)
            accuracy, test_loss = self.evaluate(testloader)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)

            print(f"Epoch: {i+1}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")

            if save_model_every_n_epochs > 0 and (i+1) % save_model_every_n_epochs == 0 and i+1 != epochs:

                print('\tSave checkpoint at epoch', i+1)

                self.save_checkpoint(self.exp_name, self.model, i+1)

                #plot the graph
                plot_per_epochs(train_losses , test_losses , accuracies , epochs)

        # Save the experiment
        self.save_experiment(self.exp_name, self.args, self.model, train_losses, test_losses, accuracies)

    def train_epoch(self, trainloader):
        """
        Train the model for one epoch.
        """
        self.model.train()
        total_loss = 0
        for batch in trainloader:
            # Move the batch to the device
            batch = [t.to(self.device) for t in batch]
            images, labels = batch
            # Zero the gradients
            self.optimizer.zero_grad()
            # Calculate the loss
            loss = self.loss_fn(self.model(images)[0], labels)
            # Backpropagate the loss
            loss.backward()
            # Update the model's parameters
            self.optimizer.step()
            total_loss += loss.item() * len(images)
        return total_loss / len(trainloader.dataset)

    @torch.no_grad()
    def evaluate(self, testloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for batch in testloader:
                # Move the batch to the device
                batch = [t.to(self.device) for t in batch]
                images, labels = batch

                # Get predictions
                logits, _ = self.model(images)

                # Calculate the loss
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item() * len(images)

                # Calculate the accuracy
                predictions = torch.argmax(logits, dim=1)
                correct += torch.sum(predictions == labels).item()
        accuracy = correct / len(testloader.dataset)
        avg_loss = total_loss / len(testloader.dataset)
        return accuracy, avg_loss
    
    def save_checkpoint(self , experiment_name, model, epoch, base_dir="experiments"):
        outdir = os.path.join(base_dir, experiment_name)
        os.makedirs(outdir, exist_ok=True)
        cpfile = os.path.join(outdir, f'model_{epoch}.pt')
        torch.save(model.state_dict(), cpfile)


    def save_experiment(self , experiment_name, args, model, train_losses, test_losses, accuracies, base_dir="experiments"):
        outdir = os.path.join(base_dir, experiment_name)
        os.makedirs(outdir, exist_ok=True)

        # Save the config
        # configfile = os.path.join(outdir, 'config.json')

        # with open(configfile, 'w') as f:
        #     json.dump(args, f, sort_keys=True, indent=4)

        # Save the metrics
        jsonfile = os.path.join(outdir, 'metrics.json')
        with open(jsonfile, 'w') as f:
            
            data = {
                'train_losses': train_losses,
                'test_losses': test_losses,
                'accuracies': accuracies,
            }

            json.dump(data, f, sort_keys=True, indent=4)

        # Save the model
        self.save_checkpoint(experiment_name, model, "final", base_dir=base_dir)
