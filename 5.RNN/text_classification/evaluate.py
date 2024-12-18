import torch
from utils.common  import *



# Just return an output given a line
def evaluate(args,line_tensor,model):

    hidden = model.initHidden().to(args.device)  # Move hidden state to the device

    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i].to(args.device), hidden)

    return output


def predict(args,input_line,all_categories,model,n_predictions=3):
    print('\n> %s' % input_line)

    with torch.no_grad():
        output = evaluate(args,lineToTensor(input_line),model)

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
