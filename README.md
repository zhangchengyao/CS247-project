# CS247-project

## Keyword Extract in CQA networks
In CQA websites (Community-based Question Answering), such as Stack Overflow, precisely extracting the labels or tags for a new problem can help categorize it in a proper way. For example, one question on Stack Overflow that asked “how to turn on the syntax highlight in Vim” was tagged with “Vim” and “Highlight”. Its twin question for turning off the highlight only has tag “Vim”. Therefore, augmenting the second question with a tag “highlight” will better describe its content. So in this project, you will be asked to propose an algorithm that can accurately and efficiently extract all the tags for a new question.

## RNN Model Usage (assume under the RNN model directory)
Current code is based on code written by Rui Meng, here is the original repo https://github.com/memray/seq2seq-keyphrase-pytorch,
However, following files are modified in order to implement our own model:
- pykp/model.py
- train.py
- pykp/dataloader.py
- config.py
- preprocess.py
- evaluate.py

#### step 1: download stackoverflow dataset 
#### step 2: preprocess the data set 
python preprocess -dataset_name stackof -source_dataset_dir [PATH_TO_DATA_SET]
#### step 3: run prepared script to train and test model
In order to test on the entire test set, please comment out line 205 and 206 in evaluate.py. But get them back as training.
### NOTICE: 
The training and testing requires cuda avaialbe on the machine, otherwise it takes extremely long to train and test with zero guarantee to run successfully at the end. The model is built based on latest cuda toolkit 10.0, so please use pytorch that supports cuda10 as well.
