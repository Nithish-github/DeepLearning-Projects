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
import argparse
import preprocessing
import preprocessing.unicode_to_ascii
from model.model_trainer import *
from evaluate import predict
# Initialize argument parser
parser = argparse.ArgumentParser(description='VISION TRANSFORMER')

parser.add_argument('--data', type=str, default='DeepLearning-Projects/RNN/text_classification/data/names/*.txt',
                    help='Path to the text data')

parser.add_argument('--device', type=str, default='cpu', help='device on which to train the model upon')



# Parse the arguments
args = parser.parse_args()

#convert the unicode to ascii 
category_lines , all_categories  = preprocessing.unicode_to_ascii.conversion(args.data)

# print("Printing sample line",category_lines['Italian'][:5])
# print("length of the categories",len(all_categories))

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
        predict(user_input , all_categories , model)

