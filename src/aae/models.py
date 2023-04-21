from torch import nn


class Encoder(nn.Module):
    """Encoder Model"""
    def __init__(self):
        super(Encoder, self).__init__()
        
        
    def forward(self, x):
        return x
        
        
class Decoder(nn.Module):
    """Decoder Model"""
    def __init__(self):
        super(Decoder, self).__init__()
    
    def forward(self, x):
        return x


class Discriminator(nn.Module):
    """Discriminator Model"""
    def __init__(self):
        super(Discriminator, self).__init__()
        
    def forward(self, x):
        return x

class AAE():
    def __init__(self):
        pass
    
    def fit(self):
        pass
    
    def generate(self):
        pass