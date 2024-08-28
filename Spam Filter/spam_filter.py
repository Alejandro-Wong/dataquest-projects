import pandas as pd
import re

"""
Dataset gathered by Tiago A. Almeida and Jose Maria Gomez Hidalgo. 
Can be downloaded here: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
"""

# Read in dataset
df = pd.read_csv('SMSSpamCollection', sep='\t', header=None)

df.columns = ['Label', 'SMS']

# Randomize dataset
df_rand = df.sample(frac=1, random_state=1)

# Index calculation for 80/20 split
tr_test_index = round(len(df_rand) * 0.8)

# Training set
tr = df_rand[:tr_test_index].reset_index(drop=True)
# Test set
test = df_rand[tr_test_index:].reset_index(drop=True)

# Remove punctuation, make all letters lowercase
tr = tr.assign(SMS=tr['SMS'].apply(lambda x: re.sub('\W', ' ', x)))
tr['SMS'] = tr['SMS'].str.lower()

# Create vocabulary
tr['SMS'] = tr['SMS'].str.split()
vocabulary = []
for row in tr['SMS']:
    for word in row:
        vocabulary.append(word)
vocabulary = set(vocabulary)
vocabulary = list(vocabulary)

# Final training set
word_counts_per_sms = {unique_word: [0] * len(tr['SMS']) for unique_word in vocabulary}

for index, sms in enumerate(tr['SMS']):
    for word in sms:
        word_counts_per_sms[word][index] += 1

word_counts = pd.DataFrame(word_counts_per_sms)
def convert_int(num):
    return num.astype(int)
word_counts = word_counts.apply(convert_int)

# Cleaned training set
tr_set = pd.concat([tr, word_counts],join='inner', axis=1)
tr_set.fillna(0, inplace=True)

# Calculate constants
ham_df = tr_set[tr_set['Label'] == 'ham']
spam_df = tr_set[tr_set['Label'] == 'spam']

p_ham = len(ham_df) / len(tr_set)
p_spam = len(spam_df) / len(tr_set)

n_word_spam = spam_df['SMS'].apply(len)
n_spam = n_word_spam.sum()

n_word_ham = ham_df['SMS'].apply(len)
n_ham = n_word_ham.sum()

n_vocabulary = len(vocabulary)

alpha = 1

# Calculate parameters
wi_spam = {word:0 for word in vocabulary}
wi_ham = {word:0 for word in vocabulary}

for word in vocabulary:
    n_word_given_spam = spam_df[word].sum()
    p_word_given_spam = (n_word_given_spam + alpha) / (n_spam + alpha * n_vocabulary)
    wi_spam[word] = p_word_given_spam
    
    n_word_given_ham = ham_df[word].sum()
    p_word_given_ham = (n_word_given_ham + alpha) / (n_ham + alpha * n_vocabulary)
    wi_ham[word] = p_word_given_ham


# Classify new message
def classify(message):

    message = re.sub('\W', ' ', message)
    message = message.lower()
    message = message.split()

    p_spam_given_message = p_spam
    p_ham_given_message = p_ham
    
    for word in message:
        if word in wi_spam:
            p_spam_given_message *= wi_spam[word]
        if word in wi_ham:
            p_ham_given_message *= wi_ham[word]
    
    print('P(Spam|message):', p_spam_given_message)
    print('P(Ham|message):', p_ham_given_message)

    if p_ham_given_message > p_spam_given_message:
        print('Label: Ham')
    elif p_ham_given_message < p_spam_given_message:
        print('Label: Spam')
    else:
        print('Equal proabilities, have a human classify this!')

# Classify test set
def classify_test_set(message):
    message = re.sub('\W', ' ', message)
    message = message.lower()
    message = message.split()

    p_spam_given_message = p_spam
    p_ham_given_message = p_ham
    
    for word in message:
        if word in wi_spam:
            p_spam_given_message *= wi_spam[word]
        if word in wi_ham:
            p_ham_given_message *= wi_ham[word]
    
    if p_ham_given_message > p_spam_given_message:
        return 'ham'
    elif p_ham_given_message < p_spam_given_message:
        return 'spam'
    else:
        return 'human classification required'
    

test['predicted'] = test['SMS'].apply(classify_test_set)
correct = 0 
total = test.shape[0]
for row in test.iterrows():
    row = row[1]
    if row['Label'] == row['predicted']:
        correct += 1
        
print('Correct: ', correct)
print('Incorrect: ', total - correct)
print('Accuracy: ', correct/total)
