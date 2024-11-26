import argparse
import preprocessing
import preprocessing.unicode_to_ascii
from model.model_trainer import *
from evaluate import predict
# Initialize argument parser
parser = argparse.ArgumentParser(description='VISION TRANSFORMER')

parser.add_argument('--data', type=str, default='data/names/*.txt',
                    help='Path to the text data')

parser.add_argument('--device', type=str, default='cuda', help='device on which to train the model upon')

# Parse the arguments
args = parser.parse_args()

#convert the unicode to ascii 
category_lines , all_categories  = preprocessing.unicode_to_ascii.conversion(args.data)

#Train the model 
model = train(args)

#evalaution check
while True:
    user_input = input("Enter something (or 'q' to quit): ")

    if user_input == "":
        continue
    if user_input == 'q':  # Check if user entered 'Q' (case-insensitive)
        print("Quitting...")
        break
    else:
        predict(args , user_input , all_categories , model)

