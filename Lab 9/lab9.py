import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
test_sentence = """START The mathematician ran . END

START The mathematician ran to the store . END

START The physicist ran to the store . END

START The philosopher thought about it . END

START The mathematician solved the open problem . END""".split()

# build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)
trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
            for i in range(len(test_sentence) - 2)]

# print the first 3, just so you can see what they look like
sanity_check = trigrams[5:13]

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}


class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs, embeds


losses = []
loss_function = nn.NLLLoss()
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.075)

for epoch in range(85):
    total_loss = torch.Tensor([0])
    for context, target in trigrams:

        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in variables)
        context_idxs = [word_to_ix[w] for w in context]
        context_var = autograd.Variable(torch.LongTensor(context_idxs))

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        model.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next
        # words
        log_probs, embeds = model(context_var)


        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a variable)
        loss = loss_function(log_probs, autograd.Variable(
            torch.LongTensor([word_to_ix[target]])))

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        total_loss += loss.data
    losses.append(total_loss)


for x in range(5):
    i = 0
    j = 0
    for context, target in sanity_check:

        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in variables)
        context_idxs = [word_to_ix[w] for w in context]
        context_var = autograd.Variable(torch.LongTensor(context_idxs))

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        model.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next
        # words
        log_probs, embeds = model(context_var)


        # Select the max of the log_probs, get the index of that return the word indexed at the location as the predicted word
        maxs, indices = torch.max(log_probs, 1)

        prediction = list(vocab)[indices]

        if prediction == target:
            print("SUCCESS --------------- SUCCESS")
            print("Context: " + str(context)+ " " + "Target: "+ target + " " + "Predicted: " + prediction)
            i += 1

        else:
            print("ERROR --------------- ERROR")
            print("Context: " + str(context)+ " " + "Target: "+ target + " " + "Predicted: " + prediction)
            j += 1

    print("Correctly labelled: " + str(i))
    print("Incorrectly labelled: " + str(j))

test = [(("START", "The"), "philosopher"), (("START", "The"), "physicist"), (("START", "The"), "mathematician")]

for context, target in test:

    # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
    # into integer indices and wrap them in variables)
    context_idxs = [word_to_ix[w] for w in context]
    context_var = autograd.Variable(torch.LongTensor(context_idxs))

    # Step 2. Recall that torch *accumulates* gradients. Before passing in a
    # new instance, you need to zero out the gradients from the old
    # instance
    model.zero_grad()

    # Step 3. Run the forward pass, getting log probabilities over next
    # words
    log_probs, embeds = model(context_var)

    # Get the word embeddings to perform cosine similarity after the loop, seeing that the gap is identified correctly.
    if target == "mathematician":
        mathematician_embed = model.embeddings( autograd.Variable( torch.LongTensor([word_to_ix["mathematician"]])))


    if target == "physicist":
        physicist_embed = model.embeddings( autograd.Variable( torch.LongTensor([word_to_ix["physicist"]])))

    if target == "philosopher":
        philosopher_embed = model.embeddings( autograd.Variable( torch.LongTensor([word_to_ix["philosopher"]])))


# Calculate the cosine similarity between the word embeddings
phys_math = F.cosine_similarity(physicist_embed, mathematician_embed)
phil_math = F.cosine_similarity(philosopher_embed, mathematician_embed)

if phys_math > phil_math:
    print("Predicted word: " + (list(vocab)[(list(vocab)).index("physicist")]))
else:
    print("Predicted word: " + (list(vocab)[(list(vocab)).index("philosopher")]))
