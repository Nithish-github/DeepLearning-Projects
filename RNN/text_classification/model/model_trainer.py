import torch.nn as nn
import torch.nn.functional as F
import torch
import time
import math
from model.model import RNN
from utils.common  import *


def trainer_loop(args, category_tensor, line_tensor , model , criterion):
    learning_rate = 0.005 # If you set this too high, it might explode. If too low, it might not learn


    hidden = model.initHidden().to(args.device)

    model.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in model.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def train(args):

    n_iters = 100
    print_every = 5
    plot_every = 10

    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []
    start = time.time()

    #loss function 
    criterion = nn.NLLLoss()

    #total letters to consider
    all_letters = string.ascii_letters + " .,;'"
    n_letters = len(all_letters)

    # initialize the model
    n_hidden = 128
    category_lines , all_categories = preprocessing.unicode_to_ascii.conversion(args.data)
    n_categories = len(all_categories)

    #model = RNN(n_letters, n_hidden, n_categories)
    # Move model to GPU if available
    model = RNN(n_letters, n_hidden, n_categories).to(args.device)

    for iter in range(1, n_iters + 1):

        category, line, category_tensor, line_tensor = randomTrainingExample(args)
        output, loss = trainer_loop(args,category_tensor, line_tensor ,model,criterion)
        current_loss += loss

        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output,args)
            correct = '✓' if guess == category else '✗ (%s)' % category

            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    #plot the loss
    plot_the_losses(all_losses)
    #save the model 
    save_final_model(model)

    return model