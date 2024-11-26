#helper fucntions 
import preprocessing.unicode_to_ascii
import random
import torch
import string
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


def categoryFromOutput(output,args):
    category_lines , all_categories = preprocessing.unicode_to_ascii.conversion(args.data)
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample(args):
    category_lines , all_categories = preprocessing.unicode_to_ascii.conversion(args.data)
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor


def letterToIndex(letter):
    return all_letters.find(letter)

def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


# Save the trained model after making predictions
def save_final_model(model, path='output/final_output.pth'):
    torch.save(model.state_dict(), path)
    print(f"Model saved at {path}")

# plot the output 
def plot_the_losses(all_losses,path ="output/predictions_loss_plot.png" ):
    # Create the plot
    plt.figure()
    plt.plot(all_losses)

    # Add labels (optional)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Training Loss over Time')

    # Save the plot as an image (e.g., PNG, PDF, etc.)
    plt.savefig(path)  # You can specify the file format (png, pdf, etc.)


