import os
import math
import argparse
import torch
from torch import optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.nn import functional as F
from seq2seq.seq2seq_model import Encoder, Decoder, Seq2Seq
from seq2seq.seq2seq_data_utils import load_dataset


def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-epochs', type=int, default=100,
                   help='number of epochs for train')
    p.add_argument('-batch_size', type=int, default=32,
                   help='number of epochs for train')
    p.add_argument('-lr', type=float, default=0.0001,
                   help='initial learning rate')
    p.add_argument('-grad_clip', type=float, default=10.0,
                   help='in case of gradient explosion')
    return p.parse_args()


def evaluate(model, val_iter, vocab_size, DE, EN):
    model.eval()
    pad = EN.vocab.stoi['<pad>']
    total_loss = 0
    for b, batch in enumerate(val_iter):
        src, len_src = batch.src
        trg, len_trg = batch.trg
        src = Variable(src.data.to(device), volatile=True)
        trg = Variable(trg.data.to(device), volatile=True)
        output = model(src, trg, teacher_forcing_ratio=0.0)
        loss = F.nll_loss(output[1:].view(-1, vocab_size),
                          trg[1:].contiguous().view(-1),
                          ignore_index=pad)
        total_loss += loss.data[0]
    return total_loss / len(val_iter)


def train(e, model, optimizer, train_iter, vocab_size, grad_clip, word2num, code2num):
    model.train()
    total_loss = 0
    pad = code2num['<pad>']
    print("pad number: ", pad)
    for b, (inputs, targets) in enumerate(train_iter):
        src = inputs
        trg = targets
        # src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg, device)
        # print("final outputs: ", output.size(),output.view(-1, vocab_size).size())
        # print("ground truth: ", trg[:,1:].size())
        loss = F.nll_loss(output.view(-1, vocab_size),
                          trg[:,1:].contiguous().view(-1),
                          ignore_index=pad)
        loss.backward()
        clip_grad_norm(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.data

        if b % 10 == 0 and b != 0:
            total_loss = total_loss / 10
            print("[%d][loss:%5.2f][pp:%5.2f]" %
                  (b, total_loss, math.exp(total_loss)))
            total_loss = 0


def main():
    hidden_size = 512
    embed_size = 256
    # assert torch.cuda.is_available()

    print("[!] preparing dataset...")
    train_loader, val_loader, test_loader, word2num, code2num = load_dataset(args.batch_size)
    word_size, code_size = len(word2num)+1, len(code2num)+1
    print("[word_size]:%d [code_size]:%d" % (word_size, code_size))

    print("[!] Instantiating models...")
    encoder = Encoder(word_size, embed_size, hidden_size,
                      n_layers=2, dropout=0.5)
    decoder = Decoder(embed_size, hidden_size, code_size,
                      n_layers=1, dropout=0.5)
    seq2seq = Seq2Seq(encoder, decoder).to(device)
    optimizer = optim.Adam(seq2seq.parameters(), lr=args.lr)
    print(seq2seq)

    best_val_loss = None
    for e in range(1, args.epochs + 1):
        train(e, seq2seq, optimizer, train_loader,
              code_size, args.grad_clip, word2num, code2num)
        val_loss = evaluate(seq2seq, val_loader, code_size, word2num, code2num)
        print("[Epoch:%d] val_loss:%5.3f | val_pp:%5.2fS"
              % (e, val_loss, math.exp(val_loss)))

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            print("[!] saving model...")
            if not os.path.isdir(".save"):
                os.makedirs(".save")
            torch.save(seq2seq.state_dict(), './.save/seq2seq_%d.pt' % (e))
            best_val_loss = val_loss
    test_loss = evaluate(seq2seq, test_loader, code_size, word2num, code2num)
    print("[TEST] loss:%5.2f" % test_loss)


if __name__ == "__main__":
    try:
        args = parse_arguments()
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
