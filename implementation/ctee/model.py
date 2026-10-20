"""One shared learned embedding, two output heads, optional electrical evidence."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.optimize import minimize_scalar
from scipy.special import softmax,logsumexp
from .features import N_BASE,N_FEATURES

class SharedClassifier(nn.Module):
    def __init__(self,enhanced=True,physics_weight=0.):
        super().__init__()
        self.enhanced=enhanced
        n=N_FEATURES if enhanced else N_BASE
        self.register_buffer('mean',torch.zeros(n))
        self.register_buffer('scale',torch.ones(n))
        self.register_buffer('physics_weight',torch.tensor(float(physics_weight)))
        self.shared=nn.Sequential(nn.Linear(n,96),nn.ReLU(),nn.Dropout(.1),nn.Linear(96,64),nn.ReLU())
        self.behaviour_head=nn.Linear(64,3)
        self.composition_head=nn.Linear(64,7)

    def forward(self,x):
        z=(x[:,:self.mean.numel()]-self.mean)/self.scale
        hidden=self.shared(z.clamp(-15,15))
        b=self.behaviour_head(hidden);c=self.composition_head(hidden)
        if self.enhanced:
            targets=x.new_tensor([0.,-1.,1.])
            pb=-((x[:,N_BASE:N_BASE+1]-targets)/.3)**2
            pb=pb.clamp(-20,0)*x[:,N_BASE+3:N_BASE+4]
            pc=x[:,N_BASE+9:N_BASE+16]
            b=b+self.physics_weight*pb;c=c+self.physics_weight*pc
        return b,c

def save_model(path,model,seed,variant):
    torch.save({'state_dict':model.state_dict(),'enhanced':model.enhanced,'seed':int(seed),'variant':variant},path)

def load_model(path):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    model=SharedClassifier(saved['enhanced']);model.load_state_dict(saved['state_dict']);model.eval()
    return model,saved

def predict_logits(model,x,batch_size=4096):
    bs=[];cs=[]
    with torch.inference_mode():
        for start in range(0,len(x),batch_size):
            b,c=model(torch.as_tensor(x[start:start+batch_size],dtype=torch.float32))
            bs.append(b.numpy());cs.append(c.numpy())
    return np.vstack(bs),np.vstack(cs)

def temperatures(logits,targets):
    def objective(logt):
        z=logits/np.exp(logt)
        return float(np.mean(logsumexp(z,axis=1)-z[np.arange(len(z)),targets]))
    return float(np.exp(minimize_scalar(objective,bounds=(-2.3,2.3),method='bounded').x))

def probabilities(logits,temperature=1.):return softmax(logits/temperature,axis=1)

