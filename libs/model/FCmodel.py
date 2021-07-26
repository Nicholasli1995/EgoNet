"""
Fully-connected model architecture for processing 1D data.

Author: Shichao Li
Contact: nicholas.li@connect.ust.hk
"""
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, 
                 num_neurons, 
                 p_dropout=0.5, 
                 kaiming=False, 
                 leaky=False):
        super(ResidualBlock, self).__init__()
        self.num_neurons = num_neurons
        self.leaky = leaky
        self.p_dropout = p_dropout
        if leaky:
            self.relu = nn.LeakyReLU(inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)
        self.w1 = nn.Linear(self.num_neurons, self.num_neurons)
        self.batch_norm1 = nn.BatchNorm1d(self.num_neurons)
        self.w2 = nn.Linear(self.num_neurons, self.num_neurons)
        self.batch_norm2 = nn.BatchNorm1d(self.num_neurons)
        if kaiming:
            # kaiming initialization
            self.w1.weight.data = nn.init.kaiming_normal_(self.w1.weight.data)
            self.w2.weight.data = nn.init.kaiming_normal_(self.w2.weight.data)
            
    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)
        out = x + y
        return out

class FCModel(nn.Module):
    def __init__(self,
                 stage_id=1,
                 num_neurons=1024,
                 num_blocks=2,
                 p_dropout=0.5,
                 norm_twoD=False,
                 kaiming=False,
                 refine_3d=False, 
                 leaky=False,
                 dm=False,
                 input_size=32,
                 output_size=64):
        """
        dm: use distance matrix feature computed from coordinates (DEPRECATED)
        leaky: use leaky ReLu instead of normal Relu
        """
        super(FCModel, self).__init__()
        self.num_neurons = num_neurons
        self.p_dropout = p_dropout
        self.num_blocks = num_blocks
        self.stage_id = stage_id
        self.refine_3d = refine_3d
        self.leaky = leaky
        self.dm = dm 
        self.input_size = input_size        
        self.output_size = output_size
        # map the input to a representation vector
        self.w1 = nn.Linear(self.input_size, self.num_neurons)
        self.batch_norm1 = nn.BatchNorm1d(self.num_neurons)
        self.res_blocks = []
        for l in range(num_blocks):
            self.res_blocks.append(ResidualBlock(num_neurons=self.num_neurons, 
                                                 p_dropout=self.p_dropout,
                                                 leaky=self.leaky))
        self.res_blocks = nn.ModuleList(self.res_blocks)
        # output
        self.w2 = nn.Linear(self.num_neurons, self.output_size)
        if self.leaky:
            self.relu = nn.LeakyReLU(inplace=True)
        else:
            self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)
        if kaiming:
            self.w1.weight.data = nn.init.kaiming_normal_(self.w1.weight.data)
            self.w2.weight.data = nn.init.kaiming_normal_(self.w2.weight.data)
            
    def forward(self, x):
        y = self.get_representation(x)
        y = self.w2(y)
        return y
    
    def get_representation(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)
        # residual blocks
        for i in range(self.num_blocks):
            y = self.res_blocks[i](y)        
        return y
    
def get_fc_model(stage_id, 
                 cfgs,
                 input_size,
                 output_size,
                 architecture_type = 'FCModel'):
    return FCModel(stage_id=stage_id, 
                   refine_3d=cfgs[architecture_type]['refine_3d'], 
                   norm_twoD=cfgs[architecture_type]['norm_twoD'],
                   num_blocks=cfgs[architecture_type]['num_blocks'], 
                   input_size=input_size, 
                   output_size=output_size, 
                   num_neurons=cfgs[architecture_type]['num_neurons'],
                   p_dropout=cfgs[architecture_type]['dropout'],
                   leaky=cfgs[architecture_type]['leaky']
                   )
    
def get_cascade():
    return nn.ModuleList([])