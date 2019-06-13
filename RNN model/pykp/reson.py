import torch
import torch.nn.init as init
import torch.nn as nn
from torch.autograd import Variable

class Resonation(nn.Module):
    def __init__(self, src_dim, k):
        super(Resonation, self).__init__()
        self.w = nn.Parameter(torch.zeros((src_dim,k),device='cuda').uniform_(0,1))
        self.softmax = nn.Softmax( dim=1 )
        self.setmemory = False
    def normalize(self, w):
        return (w-w.min())/(w.max()-w.min())
    def forward(self, input):
        batch_size = input.size(0)
        trg_seq_len = input.size(1)
        src_dim = input.size(2)
        W = Variable(torch.zeros(batch_size, trg_seq_len, src_dim))
        if torch.cuda.is_available():
            W = W.cuda()
        val, ind = torch.max(torch.mm(input.view(-1,src_dim), self.softmax(self.w)), dim = 1)
        val = val.view(batch_size,-1)
        ind = ind.view(batch_size,-1)
        for i in range(batch_size):
            for j in range(1,trg_seq_len):
                W[i,j] = val[i][j-1] * (self.w.t()[ind[i][j-1]])
                W[i,j] = self.normalize(W[i,j])
      
        print(W)
        return input + torch.mul(input, W)
        
    def clear(self):
        self.memory = False 
        
    def reinforce(self, input):
        batch_size = input.size(0)
        src_dim = input.size(1)
        if not self.setmemory:
            self.setmemory = True
            val, ind = torch.max(torch.mm(input, self.softmax(self.w)), dim = 1)
            self.memory = val.view(batch_size, -1) * self.w.t()[ind]
        else:
            input = input + torch.mul(input, self.memory)
            val, ind = torch.max(torch.mm(input, self.softmax(self.w)), dim = 1)
            self.memory = val.view(batch_size, -1) * self.w.t()[ind]
        
        return input    

reson = Resonation(5,2)
input = Variable(torch.rand(3,5,5)).cuda()
input = reson(input)

print(input)