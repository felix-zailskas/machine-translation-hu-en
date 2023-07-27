from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import time
import math

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from preprocessing import sent2idx

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 140
SOS_token = 0
EOS_token = 1

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, input_embeddings, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding.from_pretrained(input_embeddings, freeze=False)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        #self.dropout = nn.Dropout(dropout_p)

    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        # embedded = self.dropout(input_seq)
        output, hidden = self.gru(embedded)
        return output, hidden

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, output_embeddings):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(output_embeddings, freeze=False)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        # self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(
            batch_size, 1, dtype=torch.long, device=device
        ).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        
        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden = self.forward_step(
                decoder_input, decoder_hidden
            )
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1)  # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(
                    -1
                ).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        return (
            decoder_outputs,
            decoder_hidden,
            None,
        )  # We return `None` for consistency in the training loop

    def forward_step(self, input_seq, hidden):
        output = self.embedding(input_seq)
        # output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        # output = self.out(output)
        return output, hidden
    

def train_epoch(
    dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion
):

    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)
        
        embedded_target_tensor = decoder.embedding(target_tensor) # we compare the embedding of the target
        loss = criterion(
            decoder_outputs.view(-1), embedded_target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return "%s (- %s)" % (asMinutes(s), asMinutes(rs))

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    
def train(
    train_dataloader,
    encoder,
    decoder,
    n_epochs,
    learning_rate=0.001,
    print_every=100,
    plot_every=100,
):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(
            train_dataloader,
            encoder,
            decoder,
            encoder_optimizer,
            decoder_optimizer,
            criterion,
        )
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print(
                "%s (%d %d%%) %.4f"
                % (
                    timeSince(start, epoch / n_epochs),
                    epoch,
                    epoch / n_epochs * 100,
                    print_loss_avg,
                )
            )

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)
    
    
## PROBLEM: how to get a word given an arbitrary embedding? an embedder is just a lookup table!

def embedding2word(embedding, w2v_model):
    word, _ = w2v_model.wv.most_similar(positive=[embedding], topn=1)
    print("works")
    return word
    
def evaluate(encoder, decoder, sentence, input_word2idx, output_word2idx, input_w2v_model, output_w2v_model):
    with torch.no_grad():
        input_tensor = sent2idx(input_word2idx, sentence, MAX_LENGTH)
        input_tensor = torch.LongTensor(input_tensor)
        encoder_outputs, encoder_hidden = encoder(input_tensor)
        #decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)
        decoder_outputs, decoder_hidden, _ = decoder(encoder_outputs.unsqueeze(0), encoder_hidden.unsqueeze(1), None)
        # decoder_outputs, decoder_hidden, _ = decoder(encoder_outputs, encoder_hidden, None)

        # _, topi = decoder_outputs.topk(1)
        # decoded_ids = topi.squeeze()
        # decoder_idxs = [embedding2word(embedding) for embedding in encoder_outputs.squeeze()]
        decoded_words = [embedding2word(embedding, output_w2v_model) for embedding in encoder_outputs.squeeze()]
        decoded_words.append('<EOS>')
        # for idx in decoder_idxs:
        #     if idx.item() == EOS_token:
        #         decoded_words.append('<EOS>')
        #         break
        #     decoded_words.append(output_word2idx[idx.item()])
    return decoded_words