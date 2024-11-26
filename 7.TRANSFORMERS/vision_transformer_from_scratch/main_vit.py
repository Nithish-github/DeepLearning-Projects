import argparse
from models.vit_trainer import Vison_Transformer_Trainer
from data.dataloader import prepare_data
from plot.view import visualize_attention


# Initialize argument parser
parser = argparse.ArgumentParser(description='VISION TRANSFORMER')

parser.add_argument('--epochs', type=int, default=10,
                    help='Number of Epochs')

parser.add_argument('--lr', type=float, default=1e-2, 
                    help='Learning rate for the optimizer. Default is 1e-2.')

parser.add_argument('--patch_size', type=int, default=4,
                    help='Size of the patches the input image will be divided into. Default is 4.')

parser.add_argument('--hidden_size', type=int, default=48,
                    help='Dimensionality of the hidden layer in the transformer model. Default is 48.')

parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size for training')

parser.add_argument('--num_hidden_layers', type=int, default=4,
                    help='Number of hidden layers in the transformer model. Default is 4.')

parser.add_argument('--num_attention_heads', type=int, default=4,
                    help='Number of attention heads in the multi-head attention layer. Default is 4.')

parser.add_argument('--intermediate_size', type=int, default=4 * 48,
                    help='Size of the intermediate feedforward layer. Default is 4 times the hidden size (192).')

parser.add_argument('--hidden_dropout_prob', type=float, default=0.0,
                    help='Dropout probability for hidden layers. Default is 0.0 (no dropout).')

parser.add_argument('--attention_probs_dropout_prob', type=float, default=0.0,
                    help='Dropout probability for attention probabilities. Default is 0.0 (no dropout).')

parser.add_argument('--initializer_range', type=float, default=0.02,
                    help='Standard deviation of the truncated normal initializer for initializing weights. Default is 0.02.')

parser.add_argument('--image_size', type=int, default=32,
                    help='Size of the input image. Default is 32x32 pixels.')

parser.add_argument('--num_classes', type=int, default=10,
                    help='Number of output classes for classification. Default is 10.')

parser.add_argument('--num_channels', type=int, default=3,
                    help='Number of channels in the input image (e.g., 3 for RGB images). Default is 3.')

parser.add_argument('--qkv_bias', type=bool, default=True,
                    help='Whether to add bias to the Q, K, V projections in the attention mechanism. Default is True.')

parser.add_argument('--use_faster_attention', type=bool, default=True,
                    help='Whether to use faster attention implementation if available. Default is True.')

parser.add_argument('--device', type=str, default='cpu', help='device ids of multiple gpus')

parser.add_argument('--save_freq', type=int, default=5, help='save model with frequency')


# Parse the arguments
args = parser.parse_args()


#Dataloader   #CIFAR10
trainloader, testloader, _ = prepare_data(batch_size=args.batch_size)

print(f"Train Loader Size {len(trainloader)}\nTest Loader szie {len(testloader)}")


vit_object   = Vison_Transformer_Trainer(args)

#start the training
vit_object.train(trainloader, testloader , args.epochs , args.save_freq)


