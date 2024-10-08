import argparse
import preprocessing
import preprocessing.unicode_to_ascii
from model.model_trainer import *

# Initialize argument parser
parser = argparse.ArgumentParser(description='VISION TRANSFORMER')

parser.add_argument('--data', type=str, default='data/names/*.txt',
                    help='Path to the text data')


# Parse the arguments
args = parser.parse_args()

#convert the unicode to ascii 
category_lines , all_categories  = preprocessing.unicode_to_ascii.conversion(args.data)

print("Printing sample line",category_lines['Italian'][:5])
print("length of the categories",len(all_categories))

#Train the model 
train(args)