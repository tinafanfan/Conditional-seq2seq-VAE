import random

import torch
from torch import nn


class EncoderRNN(nn.Module):
    def __init__(self, args, input_size, hidden_size, condition_size):

        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.condition_size = condition_size
        self.h_c_size = hidden_size + condition_size

        self.input_embedding = nn.Embedding(input_size, self.h_c_size)
        self.condition_embedding = nn.Embedding(4, condition_size)

        self.lstm = nn.LSTM(self.h_c_size, self.h_c_size)
        
        self.fc_m_h = nn.Linear(self.h_c_size, hidden_size)        
        self.fc_logv_h = nn.Linear(self.h_c_size, hidden_size)        
        self.fc_m_c = nn.Linear(self.h_c_size, hidden_size)        
        self.fc_logv_c = nn.Linear(self.h_c_size, hidden_size)  

        self.device = args.device

    def forward(self, input, hidden, cell, condition):

        # input embedding
        embedded = self.input_embedding(input)

        embedded = embedded.view(-1, 1, self.h_c_size)  # -1: batch

        # condition embedding to hidden and cell
        c = self.condition_embedding(condition).view(1, 1, self.condition_size)
        hidden_cond = torch.cat((hidden, c), dim=2)
        cell_cond = torch.cat((cell, c), dim=2)

        # LSTM
        output, (hidden, cell) = self.lstm(embedded, (hidden_cond, cell_cond))
        
        # Reparameterization trick
        m_hidden = self.fc_m_h(hidden)
        logvar_hidden = self.fc_logv_h(hidden)
        z_hidden = self.sample_z() * torch.exp(logvar_hidden/2) + m_hidden

        m_cell = self.fc_m_c(cell)
        logvar_cell = self.fc_logv_c(cell)
        z_cell = self.sample_z() * torch.exp(logvar_cell/2) + m_cell

        return output, z_hidden, z_cell, m_hidden, logvar_hidden, m_cell, logvar_cell

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)

    def sample_z(self):
        return torch.normal(
            torch.FloatTensor([0] * self.hidden_size),
            torch.FloatTensor([1] * self.hidden_size),
        ).to(self.device)


class DecoderRNN(nn.Module):
    def __init__(self, args, input_size, hidden_size, condition_size):
        super(DecoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.condition_size = condition_size
        self.h_c_size = hidden_size + condition_size

        self.input_embedding = nn.Embedding(input_size, self.h_c_size)
        self.condition_embedding = nn.Embedding(4, condition_size)

        self.lstm = nn.LSTM(self.h_c_size, self.h_c_size)
        self.fc_h = nn.Linear(self.h_c_size, hidden_size)
        self.fc_c = nn.Linear(self.h_c_size, hidden_size)

        self.out = nn.Linear(self.h_c_size, input_size)
        self.softmax = nn.Softmax(dim=1)  # LogSoftmax

    def forward(self, input, hidden, cell, condition):

        # input embedding
        embedded = self.input_embedding(input).view(1, 1, self.h_c_size)

        # condition embedding to hidden and cell
        c = self.condition_embedding(condition).view(1, 1, self.condition_size)
        
        hidden_cond = torch.cat((hidden, c), dim=2)
        cell_cond = torch.cat((cell, c), dim=2)
        
        
        # LSTM
        output, (hidden, cell) = self.lstm(embedded, (hidden_cond, cell_cond))
        hidden = self.fc_h(hidden)
        cell = self.fc_c(cell)

        # output
        output = self.out(output).view(-1, self.input_size)

        return output, hidden, cell


class cVAE(nn.Module):
    def __init__(self, args):
        super(cVAE, self).__init__()
        self.encoder = EncoderRNN(args, args.input_size, args.hidden_size, args.condition_size)
        self.decoder = DecoderRNN(args, args.input_size, args.hidden_size, args.condition_size)
#         self.teacher_forcing_ratio = args.teacher_forcing_ratio
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input, condition, kld_weight, teacher_forcing_ratio):
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        
        input = input[0]
        input_length = input.size(0)
        encoder_hidden = self.encoder.initHidden()
        encoder_cell = self.encoder.initHidden()

        encoder_output_all = []  # encoder 吐出來的全部

        (
            encoder_output,
            encoder_hidden,
            encoder_cell,
            m_hidden,
            logv_hidden,
            m_cell,
            logv_cell,
        ) = self.encoder(input[1:-1], encoder_hidden, encoder_cell, condition)

        kl_div_hidden = 0.5 * torch.sum(m_hidden ** 2 + logv_hidden.exp() - 1 - logv_hidden)
        kl_div_cell = 0.5 * torch.sum(m_cell ** 2 + logv_cell.exp() - 1 - logv_cell)

        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell
        decoder_input = input[0]

        reconst_loss = 0
        output_list = []
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(input_length - 1):
                decoder_output, decoder_hidden, decoder_cell = self.decoder(
                    decoder_input, decoder_hidden, decoder_cell, condition
                )
                reconst_loss += self.criterion(decoder_output, input[di + 1].view([1]))

                decoder_input = input[di + 1]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(input_length - 1):
                decoder_output, decoder_hidden, decoder_cell = self.decoder(
                    decoder_input, decoder_hidden, decoder_cell, condition
                )
                reconst_loss += self.criterion(decoder_output, input[di + 1].view([1]))

                softmax = nn.LogSoftmax(dim=1)
                output = torch.argmax(
                    softmax(decoder_output), dim=1
                )  # CrossEntropyLoss uses LogSoftmax + argmax
                decoder_input = output
                output_list.append(output)

        reconst_loss = reconst_loss / (input_length - 1)
        loss = reconst_loss + (kl_div_hidden + kl_div_cell) * kld_weight

        return loss, reconst_loss, kl_div_hidden, kl_div_cell
