import datetime
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from preprocessing.processor import Code


code_path = './language_model_corpus/train_code_lm.txt'
# code_path = './language_model_corpus/test.txt'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
code_lm_dir = './pretrain_code_lm'
if not os.path.exists(code_lm_dir):
    os.mkdir(code_lm_dir)
print("Saving models to {}".format(code_lm_dir))

run_id = str(int(time.time()))
maxEpochs = 8
batchSize = 64
trainLossPrintPeriod = 200
lrDecayPeriod = 6
lrGamma = 0.1
initLR = 0.0005
betas = (0.9, 0.999)
weightDecay = 1e-5

# rnn parameters
decoderContextSize   = 2
decoderEmbeddingSize = 128
decoderHiddenSize    = 1024


class CodeDataLoader(DataLoader):
    def __init__(self, batchSize, codes_indx, vocab):
        self.batchSize = batchSize
        self.codes_indx = codes_indx
        special_symbols = vocab.get_special_symbols()
        self.eosInd = special_symbols['eos']
        
    def __iter__(self):
        batchCount = len(self.codes_indx) // self.batchSize
        upperBound = batchCount * self.batchSize
        p = np.random.permutation(len(self.codes_indx))

        # Create iterator
        for i in range(0, upperBound, self.batchSize):
            batchCode = []
            batchCodeLength = []
            for j in range(i, min(i + self.batchSize, upperBound)):
                batchCode.append(self.codes_indx[p[j]])
                batchCodeLength.append(len(self.codes_indx[p[j]]))
            
            # sort in decreasing length for packing
            order = sorted(range(len(batchCodeLength)), key=batchCodeLength.__getitem__, reverse=True)
            batchCode = [batchCode[i] for i in order]
            batchCodeLength = [batchCodeLength[i] for i in order]
            maxLength = max(batchCodeLength)

            # pad each line of code.
            for j in range(len(batchCode)):
                while len(batchCode[j]) < maxLength:
                    batchCode[j].append(self.eosInd)
            yield (torch.LongTensor(batchCode), batchCodeLength)



class Decoder(nn.Module):
    def __init__(self, uniqueTokens, contextSize=2, embeddingSize=256, hiddenSize=1024):
        super(Decoder, self).__init__()
        self.contextSize = contextSize
        self.embeddingSize = embeddingSize
        self.hiddenSize = hiddenSize

        # embedding layer
        self.embedding = nn.Embedding(num_embeddings=uniqueTokens,
                                      embedding_dim=self.embeddingSize)

        # projection layers
        self.tokenProjection = nn.ModuleList()
        self.tokenProjection.append(nn.Linear(in_features=self.embeddingSize, out_features=self.hiddenSize))
        self.tokenProjection.append(nn.ReLU())
        self.tokenProjection.append(nn.Linear(in_features=self.hiddenSize, out_features=uniqueTokens))

    def __getContextEmbedding(self, batchCode, midInd):
        # Inputs:
        # batchCode:  (batchSize x batchMaxCodeLen)
        # midInd:     a scalar, indicating the index of the central token
        assert (midInd - self.contextSize >= 0)
        assert (midInd + self.contextSize < batchCode.size()[1])
        assert self.contextSize > 0

        contextEmbedding = self.embedding(batchCode[:, midInd - self.contextSize])
        contextEmbedding += self.embedding(batchCode[:, midInd + self.contextSize])
        for i in range(midInd - self.contextSize + 1, midInd):
            contextEmbedding += self.embedding(batchCode[:, i])
            contextEmbedding += self.embedding(batchCode[:, 2 * midInd - i])
        return contextEmbedding

    def forward(self, batchCode):
        # Inputs:
        # batchCode:          (batchSize x batchMaxCodeLen)
        # Outputs:
        # logits:             (batchSize x (batchMaxCodeLen - 2 * contextSize) x numUniqueTokens)

        batchSize = batchCode.size()[0]
        numTimeSteps = batchCode.size()[1]
        assert numTimeSteps >= (2 * self.contextSize + 1)

        allLogits = []
        for i in range(self.contextSize, numTimeSteps - self.contextSize):
            # Get context for CBoW
            contextEmbedding = self.__getContextEmbedding(batchCode, i) # (batchSize x embeddingSize)
            logits = contextEmbedding
            for l in self.tokenProjection:
                logits = l(logits)   # (batchSize x numUniqueTokens)
            allLogits.append(logits.unsqueeze(1))
        allLogits = torch.cat(allLogits, dim=1)
        assert allLogits.size()[1] == (numTimeSteps - 2 * self.contextSize)
        return allLogits

    def getLoss(self, batchCode, batchCodeLength):
        # Inputs:
        # batchCode:       (batchSize x batchMaxCodeLen)
        # batchCodeLength: python list of length batchSize
        logits = self.forward(batchCode)  # (batchSize x (batchMaxCodeLen - 2 * contextSize) x numUniqueTokens)

        batchSize = logits.size()[0]
        predictedCodeLen = logits.size()[1]
        assert (predictedCodeLen == batchCode.size()[1] - 2 * self.contextSize)

        batchCodeLength = list(map(lambda x: min(x - self.contextSize, predictedCodeLen), batchCodeLength))
        batchCodeLength = torch.LongTensor(batchCodeLength).to(DEVICE)
        batchCodeMiddle = batchCode[:, self.contextSize : -self.contextSize].contiguous()

        # calculate accuracy
        predictions = logits.view(batchSize * predictedCodeLen, -1).cpu().detach().argmax(dim=1).numpy()
        labels = batchCodeMiddle.view(batchSize * predictedCodeLen, ).cpu().detach().numpy()
        acc = ((predictions == labels).mean())

        # calcualte loss
        criterion = nn.CrossEntropyLoss(reduction='none').to(DEVICE)
        rawLoss = criterion(logits.view(batchSize * predictedCodeLen, -1), 
                            batchCodeMiddle.view(batchSize * predictedCodeLen, )).view(batchSize, -1) # (batchSize x (batchMaxCodeLen - 2 * contextSize))
        maskedLoss = torch.mul(rawLoss, Decoder.getMask(predictedCodeLen, batchCodeLength).to(DEVICE)) # (batchSize x (batchMaxCodeLen - 2 * contextSize))
        lossSum = torch.sum(maskedLoss, dim=1) # (batchSize, )
        loss = torch.mean(lossSum, dim=0)
        return loss, acc

    @staticmethod
    def getMask(maxLength, batchCodeLength):
        # Input:
        # maxLength:   a scalar that indicates the longest length (sequence or transcript)
        # batchLength: (batchSize, )
        msk = torch.arange(maxLength).unsqueeze(1) < batchCodeLength.view(1, -1) # maxLength x batchSize
        return msk.transpose(0, 1).float().contiguous() # batchSize x maxLength

def saveModel(decoder, run_id):
    embeddingPath = os.path.join(code_lm_dir, 'embedding-{}.t7'.format(run_id))
    torch.save((decoder.embedding).state_dict(), embeddingPath)
    print("Saved model to {}".format(embeddingPath))


def train(decoder, trainLoader, run_id='89757'):
    decoder = decoder.train()
    optimizer = torch.optim.Adam(list(decoder.parameters()), lr=initLR, weight_decay=weightDecay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lrDecayPeriod, gamma=lrGamma)

    bestLoss = None
    epochInd = 0
    for epochInd in range(maxEpochs):
        decoder = decoder.train()
                
        print("=============== Starting epoch {} ==================".format(epochInd))
        startTime = datetime.datetime.now()
        epochLoss = 0.0
        epochAcc = 0.0
        batchNum = 0
        for (paddedBatchCode, batchCodeLength) in trainLoader:
            # Prepare variables
            optimizer.zero_grad()
            paddedBatchCode = paddedBatchCode.to(DEVICE)

            # Calculate loss
            loss, acc = decoder.getLoss(paddedBatchCode, batchCodeLength)
            loss.backward()
            optimizer.step()
            
            # stats
            batchLoss = loss.item() / torch.mean(torch.LongTensor(batchCodeLength).float()).item()
            epochLoss += batchLoss
            epochAcc += acc

            if batchNum % trainLossPrintPeriod == 0:
                print("============== At batch {} ==============".format(batchNum))
                print("Batch Training loss per token: ", batchLoss)
                print("Batch accuracy: {}".format(acc))
                print("Epoch average loss per token: ", epochLoss / (batchNum + 1))
            batchNum += 1
        epochLoss = epochLoss / (batchNum + 1)
        epochAcc = epochAcc / (batchNum + 1)
        print('[TRAIN]  Epoch [{}/{}]   Loss: {}   Acc: {}'.format(epochInd, maxEpochs, epochLoss, epochAcc))
        scheduler.step()
        if (bestLoss is None) or (epochLoss < bestLoss):
            bestLoss = epochLoss
            saveModel(decoder, run_id)
        timeDiff = datetime.datetime.now() - startTime
        print("This epoch used {} seconds".format(timeDiff.total_seconds()))
        print("=============== Epoch {} finished ==================".format(epochInd))



# Dataset
code = Code()
code.load_dict(path='./vocab/')
codes_indx = code.load_data(code_path)
code.pad(pad_length=1)
codes_indx = [line for line in codes_indx if len(line) >= 2 * decoderContextSize + 1]
uniqueTokens = len(code.num2code)
print("Number of lines of code to train: {}".format(len(codes_indx)))
print("Number of unique tokens: {}".format(uniqueTokens))

# Translator and data loaders
trainLoader = CodeDataLoader(batchSize=batchSize, 
                             codes_indx=codes_indx,
                             vocab=code)

# Create a decoder and train
decoder = Decoder(uniqueTokens=uniqueTokens,
                  contextSize=decoderContextSize, 
                  embeddingSize=decoderEmbeddingSize,
                  hiddenSize=decoderHiddenSize).to(DEVICE)
train(decoder, trainLoader, run_id)
