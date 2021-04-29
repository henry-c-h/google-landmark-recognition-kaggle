from utils import *
import torch
import torch.nn as nn
import torchvision.models as models

OUT_DIM = {
    'resnet50': 2048,
    'resnet101': 2048,
    'densenet121': 1024,
    'densenet169': 1664,
}

# build neural network
class LandmarkRecognitionNet(nn.Module):
    def __init__(self, backbone,
                 embedding_size=512,
                 pooling='avg',
                 pretrained=True,
                 use_fc=False,
                 extract_conv=False):
        super(LandmarkRecognitionNet, self).__init__()
        
        assert backbone in ['resnet50', 'resnet101', 'densenet121', 'densenet169']

        net = getattr(models, backbone)(pretrained=pretrained)
        if backbone.startswith('resnet'):
            self.convs = nn.Sequential(*list(net.children())[:-2])
        elif backbone.startswith('densenet'):
            self.convs = nn.Sequential(*list(net.features.children()))
        
        self.extract_conv = extract_conv
        if extract_conv:
            self.out_dim = OUT_DIM[backbone]
        else:
            self.out_dim = embedding_size
            
        self.use_fc = use_fc
        self.pooling = pooling
        self.fc = nn.Linear(OUT_DIM[backbone], embedding_size)
    
        if self.pooling == 'gem':
            self.pool = GeM()
        elif self.pooling == 'max':
            self.pool = nn.AdaptiveMaxPool2d((1,1))
        elif self.pooling == 'avg':
            self.pool = nn.AdaptiveAvgPool2d((1,1))
            
        self.bn = nn.BatchNorm1d(embedding_size)
        
    def forward(self, x):
        out = self.convs(x)
        out = self.pool(out).squeeze(-1).squeeze(-1)
        if self.extract_conv:
            return out
        else:
            fc_out = self.bn(self.fc(out))
            return fc_out


def extract_vectors(net, dl, logger, batch_size, log_freq):
    """Performs the extraction of feature embeddings"""
    
    net.cuda()
    net.eval()
    
    out_dim = net.out_dim
    
    vecs = torch.zeros(len(dl.dataset), out_dim)  
    with torch.no_grad():
        for i, batch_t in enumerate(dl):
            input_t, label_t = batch_t
            input_g = input_t.cuda()
            
            start_idx = i * batch_size
            end_idx = i * batch_size + len(input_t)
            vecs[start_idx : end_idx, :] = net(input_g).cpu()
            
            if i % log_freq == 0:
                logger.info(f'Extracting features for batch {i+1}/{len(dl)}...')
            if i + 1 == len(dl):
                logger.info(f'Finished extracing features for {len(dl.dataset)} images.')
            
    return vecs