import argparse

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from models import cVAE
from utils import WordDataset
from utils import KL_annealing
from utils import teacher_forcing
from utils import gaussian_score
from utils import compute_bleu
from torch.nn.utils import clip_grad_norm_


def train(args, train_dataloader, test_dataset, model, optimizer, device):
    total_iteration = args.num_epoch * len(train_dataloader)
    for epoch in range(args.num_epoch):
        epoch = epoch + 50
        for i, input_data in enumerate(train_dataloader):
            
            iteration = epoch*len(train_dataloader) + i
            kld_weight = KL_annealing(iteration, total_iteration) # no. of iteration
            teacher_forcing_ratio = teacher_forcing(iteration, total_iteration)
            
            input, condition = input_data
            input = input.to(device)
            condition = condition.to(device)
            optimizer.zero_grad()
            loss, reconst_loss, kl_div_hidden, kl_div_cell = model(input, condition, kld_weight, teacher_forcing_ratio)
            loss.backward()
            clip_grad_norm_(model.parameters(), args.clip_max_norm) # clip
            optimizer.step()
        
        output, bleu_score_avg = evaluate(args, test_dataset, model.encoder, model.decoder, device, save_output=False)
        gaussian_score = generate(args, test_dataset, model.encoder, model.decoder, device)
        print("Epoch[{}/{}], Iter [{}/{}], Loss: {:.4f}, Reconst Loss: {:.4f}, KL Div hidden: {:.4f}, KL Div cell: {:.4f}, bleu score avg.:{:.4f}, gaussian score:{:.4f}".format(
            epoch,
            args.num_epoch,
            iteration,
            total_iteration,
            loss,
            reconst_loss,
            kl_div_hidden,
            kl_div_cell,
            bleu_score_avg,
            gaussian_score
            )
        )
        
        torch.save(model.state_dict(), "model.pt")
        torch.save(optimizer.state_dict(), "optimizer.pt")
        
        if epoch == 0:
            with open("result_train.txt", "w") as output_file:

                output_file.write(str(epoch) +
                                  " " + str(loss.item()) + 
                                  " " + str(reconst_loss.item()) + 
                                  " " + str(kl_div_hidden.item()) + 
                                  " " + str(kl_div_cell.item()) +
                                  " " + str(bleu_score_avg) +
                                  " " + str(gaussian_score) +
                                  " " + str(kld_weight) + 
                                  " " + str(teacher_forcing_ratio))
                output_file.write('\n') 
        else:
            with open("result_train.txt", "a") as output_file:

                output_file.write(str(epoch) +
                                  " " + str(loss.item()) + 
                                  " " + str(reconst_loss.item()) + 
                                  " " + str(kl_div_hidden.item()) + 
                                  " " + str(kl_div_cell.item()) +
                                  " " + str(bleu_score_avg) +
                                  " " + str(gaussian_score) +
                                  " " + str(kld_weight) + 
                                  " " + str(teacher_forcing_ratio))
                output_file.write('\n') 
            
            
        


def evaluate(args, test_dataset, encoder, decoder, device, save_output=True):

    
    result = []
    target_list = []
    bleu_score = 0

    for i in range(len(test_dataset)):
        
        
        input_encoder,condition_encoder, target, condition_target = test_dataset[i]
        
        input_encoder = input_encoder.to(device)
        condition_encoder = condition_encoder.to(device)
        
        
        target = target.to(device)
        condition_target = condition_target.to(device)

        with torch.no_grad():

            input_encoder_len = input_encoder.size(0)

            encoder_hidden = encoder.initHidden()
            encoder_cell = encoder.initHidden()
            (
                encoder_output,
                encoder_hidden,
                encoder_cell,
                m_hidden,
                logv_hidden,
                m_cell,
                logv_cell,
            ) = encoder(input_encoder[1:-1], encoder_hidden, encoder_cell, condition_encoder)

            target_len = target.size(0)
            decoder_input = target[0]
            

            decoder_hidden = encoder_hidden
            decoder_cell = encoder_cell

            output_vec = []
            for di in range(target_len - 2):
                decoder_output, decoder_hidden, decoder_cell = decoder(
                    decoder_input, decoder_hidden, decoder_cell, condition_target
                )

                softmax = nn.LogSoftmax(dim=1)
                output = torch.argmax(
                    softmax(decoder_output), dim=1
                )  # CrossEntropyLoss uses LogSoftmax + argmax

                decoder_input = output
                output_vec.append(output)
            output_vec = torch.tensor(output_vec)
        
        
        result.append(test_dataset.chardict.StringFromLongtensor(output_vec, show_token = True))
        target_list.append(test_dataset.chardict.StringFromLongtensor(target))
        
        bleu_score += compute_bleu(reference = test_dataset.chardict.StringFromLongtensor(target), 
                                   output = test_dataset.chardict.StringFromLongtensor(output_vec, show_token = False))
    
    bleu_score_avg = bleu_score/len(test_dataset)
    print(bleu_score_avg)
    if save_output == True:
        with open("result_words.txt", "a") as output:

            output.write(str(target_list))
            output.write('\n')
            output.write(str(result))   
            output.write('\n')        
        
    return result, bleu_score_avg


def generate(args,test_dataset, encoder, decoder, device):
    
    with torch.no_grad():
        words_list = []
        for i_generate in range(args.num_generation): # generate

            noise_hidden = encoder.sample_z().view(1, 1, args.hidden_size)
            noise_cell   = encoder.sample_z().view(1, 1, args.hidden_size)
            
            output_vec_4tense = []
            for tense in range(4):

                decoder_input = torch.tensor(0).to(device)
                condition_target = torch.tensor([[tense]]).to(device)
                decoder_hidden = noise_hidden.to(device)
                decoder_cell = noise_cell.to(device)

                output_vec = []
                di = 0
                while di <= args.max_length and decoder_input != torch.tensor(1):
                    decoder_output, decoder_hidden, decoder_cell = decoder(
                        decoder_input, decoder_hidden, decoder_cell, condition_target
                    )

                    softmax = nn.LogSoftmax(dim=1)
                    output = torch.argmax(
                        softmax(decoder_output), dim=1
                    )  # CrossEntropyLoss uses LogSoftmax + argmax

                    decoder_input = output
                    output_vec.append(output)
                    di += 1
                output_vec = torch.tensor(output_vec)
                output_vec_4tense.append(test_dataset.chardict.StringFromLongtensor(output_vec, show_token = False))

            if i_generate == 0:
                with open("result_words_generation.txt", "w") as output:# erase old file
                    output.write(str(output_vec_4tense))
                    output.write('\n') 

            with open("result_words_generation.txt", "a") as output:# append to existing file
                output.write(str(output_vec_4tense))
                output.write('\n')
        
        
            words_list.append(output_vec_4tense)
    
    return gaussian_score(words_list)
        

    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epoch", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-1, help="learning rate")
    parser.add_argument("--teacher_forcing_ratio", "-tf", type=float, default=1)
    parser.add_argument("--kld_weight", "-klw", type=float, default=0.1)
    parser.add_argument("--input_size", type=int, default=28)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--condition_size", type=int, default=32)
    parser.add_argument("--clip_max_norm", type=float, default=1)
    parser.add_argument("--max_length", type=int, default=10)
    parser.add_argument("--num_generation", type=int, default=100)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--do_generate", action="store_true")    
    args = parser.parse_args()

    if args.no_cuda:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    train_dataset = WordDataset(train_ = True)
    test_dataset = WordDataset(train_ = False)
    train_dataloader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True)
    model = cVAE(args).to(device)
    optimizer = optim.SGD(model.parameters(), lr = args.learning_rate)

    if args.do_train:
        model.load_state_dict(torch.load("model.pt"))
        optimizer.load_state_dict(torch.load("optimizer.pt"))
        train(args, train_dataloader, test_dataset, model, optimizer, device)
       

    if args.do_eval:
        model.load_state_dict(torch.load("model.pt"))
        optimizer.load_state_dict(torch.load("optimizer.pt"))

        output, bleu_score_avg = evaluate(args, test_dataset, model.encoder, model.decoder, device, save_output = True)
    
    if args.do_generate:
        model.load_state_dict(torch.load("model.pt"))
        optimizer.load_state_dict(torch.load("optimizer.pt"))        
        
        gaussian_score = generate(args, test_dataset, model.encoder, model.decoder, device)
        print(gaussian_score)

if __name__ == "__main__":
    main()
