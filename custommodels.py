import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self,in_channels,num_residual_hidden,num_hidden_out):
        super(ResidualBlock,self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(in_channels=in_channels,out_channels=num_residual_hidden,kernel_size=3,stride=1,padding=1,bias=False),
            nn.ReLU(True),
            nn.Conv1d(in_channels=num_residual_hidden,out_channels=num_hidden_out,kernel_size=1,stride=1,padding=0,bias=False)
        )
    def forward(self,x):
        outpt = self.block(x)
        return x + outpt

class ResidualStack(nn.Module):
    def __init__(self,num_residual_layers,in_channels,num_residual_hidden,num_hidden_out):
        super(ResidualStack,self).__init__()
        self.num_residual_layers = num_residual_layers
        self.layers = nn.ModuleList([ResidualBlock(in_channels,num_residual_hidden,num_hidden_out) for _ in range(self.num_residual_layers)])
    def forward(self,x):
        for i in range(self.num_residual_layers):
            x = self.layers[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self,num_residual_layers,in_channels,num_residual_hidden,num_hiddens):
        super(Encoder,self).__init__()
        self.conv1 =  nn.Conv1d(in_channels=in_channels,out_channels=num_hiddens//2,kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels=num_hiddens//2,out_channels=num_hiddens,kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv1d(in_channels=num_hiddens,out_channels=num_hiddens,kernel_size=3, stride=1, padding=1)
        self.residual_stack = ResidualStack(num_residual_layers=num_residual_layers,in_channels=num_hiddens,num_residual_hidden=num_residual_hidden,num_hidden_out=num_hiddens)
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.residual_stack(x)
        return x


class VectorQuantizerEMA(nn.Module):
    def __init__(self,num_embeddings,embedding_dim,commitment_cost,decay,epsilon=1e-5):
        super(VectorQuantizerEMA,self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        self.embedding = nn.Embedding(self.num_embeddings,self.embedding_dim)
        self.embedding.weight.data.normal_()

        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        
        self.ema_w = nn.Parameter(torch.Tensor(num_embeddings, self.embedding_dim))
        self.ema_w.data.normal_()

    def forward(self,inputs):
        inputs = inputs.permute(0,2,1).contiguous() #move from (M,C,L) to (M,L,C)
        input_shape = inputs.shape
        flat_input = inputs.view(-1,self.embedding_dim) # (-1,C)

        #calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        #encoding
        encoding_indices = torch.argmin(distances,dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0],self.num_embeddings,device=inputs.device)
        encodings.scatter_(1,encoding_indices,1)

        #quantize
        quantized = torch.matmul(encodings,self.embedding.weight).view(input_shape)

        #use EMA
        if self.training:
            self.ema_cluster_size = self.ema_cluster_size * self.decay + (1 - self.decay) * torch.sum(encodings, 0)
                        
            # Laplace smoothing of the cluster size
            n = torch.sum(self.ema_cluster_size.data)
            self.ema_cluster_size = ((self.ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self.ema_w = nn.Parameter(self.ema_w * self.decay + (1 - self.decay) * dw)
            
            self.embedding.weight = nn.Parameter(self.ema_w / self.ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from MLC -> MCL
        return loss, quantized.permute(0, 2, 1).contiguous(), perplexity, encodings

class Decoder(nn.Module):
    def __init__(self,  num_residual_layers, in_channels, num_residual_hiddens,num_hiddens,output_channels):
        super(Decoder, self).__init__()
    
        self.conv_1 = nn.Conv1d(in_channels=in_channels,out_channels=num_hiddens,kernel_size=3, stride=1, padding=1)
        
        self.residual_stack = ResidualStack(num_residual_layers=num_residual_layers,in_channels=num_hiddens,num_residual_hidden=num_residual_hiddens,num_hidden_out=num_hiddens)
        
        self.conv_trans_1 = nn.ConvTranspose1d(in_channels=num_hiddens, out_channels=num_hiddens//2,kernel_size=4, stride=2, padding=1)
        
        self.conv_trans_2 = nn.ConvTranspose1d(in_channels=num_hiddens//2, out_channels=output_channels,kernel_size=4, stride=2, padding=1)

    def forward(self, inputs):
        x = self.conv_1(inputs)
        x = self.residual_stack(x)
        x = self.conv_trans_1(x)
        #x = F.relu(x)
        
        
        return self.conv_trans_2(x)



class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        in_channels = 1
        num_residual_hidden= 32
        num_hiddens = 128
        num_residual_layers = 3

        self.encoder = Encoder(num_residual_layers=num_residual_layers,in_channels=in_channels,num_residual_hidden=num_residual_hidden,num_hiddens=num_hiddens)
        
        embedding_dim = 255

        self.pre_emb_conv = nn.Conv1d(in_channels=num_hiddens,out_channels=embedding_dim,kernel_size=1,stride=1)

        num_embeddings = 512
        commitment_cost = .25
        decay = .99
        self.vq_vae = VectorQuantizerEMA(num_embeddings=num_embeddings,embedding_dim=embedding_dim,commitment_cost=commitment_cost,decay=decay)

        self.decoder = Decoder(num_residual_layers,embedding_dim,num_residual_hidden,num_hiddens,in_channels)


    def forward(self,x):
        x = self.encoder(x)
        x = self.pre_emb_conv(x)
        loss,quantized,perplexity,_  = self.vq_vae(x)
        x_recon = self.decoder(quantized)
        x_recon = torch.tanh(x_recon)

        return  x_recon,loss, perplexity


class VAE(nn.Module):
    def __init__(self,chunk):
        super().__init__()
        self.chunk = chunk
        self.fc1 = nn.Linear( self.chunk , 400)
        self.fc1a = nn.Linear(400,200)
        self.fc21 = nn.Linear(200, 10)
        self.fc22 = nn.Linear(200, 10)
        self.fc3 = nn.Linear(10, 200)
        self.fc3a = nn.Linear(200,400)
        self.fc4 = nn.Linear(400, self.chunk )

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h1 = F.relu(self.fc1a(h1))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std,device=mu.device)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h3 = F.relu(self.fc3a(h3))
        return torch.tanh(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()
    
    def forward(self, x):
        # Do your print / debug stuff here
        debug=False
        if debug:
            print(x.shape)
        return x

class Model2(nn.Module):
    def __init__(self,chunk):
        super(Model2,self).__init__()
        self.chunk = chunk
        self.kernel_size = (1,5)
        
        self.encode = nn.Sequential(
            torch.nn.Conv2d(1,64,kernel_size=self.kernel_size),
            torch.nn.ReLU(True),
            PrintLayer(),
            #torch.nn.AvgPool2d(kernel_size=(1,2)),
            PrintLayer(),
            torch.nn.Conv2d(64,32,kernel_size=self.kernel_size),
            torch.nn.ReLU(True),
            PrintLayer(),
            #torch.nn.AvgPool2d(kernel_size=(1,2)),
            PrintLayer(),
            torch.nn.Conv2d(32,4,kernel_size=self.kernel_size),
            torch.nn.ReLU(True),
            PrintLayer(),
            #torch.nn.AvgPool2d(kernel_size=(1,2)),
            torch.nn.Conv2d(4,2,kernel_size=self.kernel_size),
            torch.nn.ReLU(True),
            PrintLayer(),
            #torch.nn.AvgPool2d(kernel_size=(1,2)),
            PrintLayer(),
            )
        self.decode = nn.Sequential(
            torch.nn.ConvTranspose2d(2,4,kernel_size=self.kernel_size),
            torch.nn.ReLU(True),
            PrintLayer(),
            #torch.nn.UpsamplingNearest2d(scale_factor=(1,2)),
            PrintLayer(),
            torch.nn.ConvTranspose2d(4,32,kernel_size=self.kernel_size),
            torch.nn.ReLU(True),
            PrintLayer(),
            #torch.nn.UpsamplingNearest2d(scale_factor=(1,2)),
            PrintLayer(),
            torch.nn.ConvTranspose2d(32,64,kernel_size=self.kernel_size),
            torch.nn.ReLU(True),
            PrintLayer(),
            #torch.nn.UpsamplingNearest2d(scale_factor=(1,2)),
            PrintLayer(),
            torch.nn.ConvTranspose2d(64,1,kernel_size=self.kernel_size),
            torch.nn.Tanh(),
            PrintLayer(),
            )
    
    def forward(self, x):
        x = torch.unsqueeze(x,dim=2)
        encode = self.encode(x)
        #print(encode.shape)
        decode = self.decode(encode)
        return torch.squeeze(decode,dim=1)


from contextlib import nullcontext
import miscfuncs
import math

def wrapperkwargs(func, kwargs):
    return func(**kwargs)

def wrapperargs(func, args):
    return func(*args)

# A simple RNN class that consists of a single recurrent unit of type LSTM, GRU or Elman, followed by a fully connected
# layer

class SimpleRNN(nn.Module):

    def __init__(self, input_size=1, output_size=1, unit_type="LSTM", hidden_size=32, skip=1, bias_fl=True,
                 num_layers=1):
        super(SimpleRNN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        # Create dictionary of possible block types
        self.rec = wrapperargs(getattr(nn, unit_type), [input_size, hidden_size, num_layers])
        self.lin = nn.Linear(hidden_size, output_size, bias=bias_fl)
        self.bias_fl = bias_fl
        self.skip = skip
        self.save_state = True
        self.hidden = None

    def forward(self, x):
        if self.skip:
            # save the residual for the skip connection
            res = x[:, :, 0:self.skip]
            x, self.hidden = self.rec(x, self.hidden)
            return self.lin(x) + res
        else:
            x, self.hidden = self.rec(x, self.hidden)
            return self.lin(x)

    # detach hidden state, this resets gradient tracking on the hidden state
    def detach_hidden(self):
        if self.hidden.__class__ == tuple:
            self.hidden = tuple([h.clone().detach() for h in self.hidden])
        else:
            self.hidden = self.hidden.clone().detach()

    # changes the hidden state to None, causing pytorch to create an all-zero hidden state when the rec unit is called
    def reset_hidden(self):
        self.hidden = None

    # This functions saves the model and all its paraemters to a json file, so it can be loaded by a JUCE plugin
    def save_model(self, file_name, direc=''):
        if direc:
            miscfuncs.dir_check(direc)
        model_data = {'model_data': {'model': 'SimpleRNN', 'input_size': self.rec.input_size, 'skip': self.skip,
                                     'output_size': self.lin.out_features, 'unit_type': self.rec._get_name(),
                                     'num_layers': self.rec.num_layers, 'hidden_size': self.rec.hidden_size,
                                     'bias_fl': self.bias_fl}}

        if self.save_state:
            model_state = self.state_dict()
            for each in model_state:
                model_state[each] = model_state[each].tolist()
            model_data['state_dict'] = model_state

        miscfuncs.json_save(model_data, file_name, direc)

    # train_epoch runs one epoch of training
    def train_epoch(self, input_data, target_data, loss_fcn, optim, bs, init_len=200, up_fr=1000):
        # shuffle the segments at the start of the epoch
        shuffle = torch.randperm(input_data.shape[1])

        # Iterate over the batches
        ep_loss = 0
        for batch_i in range(math.ceil(shuffle.shape[0] / bs)):
            # Load batch of shuffled segments
            input_batch = input_data[:, shuffle[batch_i * bs:(batch_i + 1) * bs], :]
            target_batch = target_data[:, shuffle[batch_i * bs:(batch_i + 1) * bs], :]

            # Initialise network hidden state by processing some samples then zero the gradient buffers
            self(input_batch[0:init_len, :, :])
            self.zero_grad()

            # Choose the starting index for processing the rest of the batch sequence, in chunks of args.up_fr
            start_i = init_len
            batch_loss = 0
            # Iterate over the remaining samples in the mini batch
            for k in range(math.ceil((input_batch.shape[0] - init_len) / up_fr)):
                # Process input batch with neural network
                output = self(input_batch[start_i:start_i + up_fr, :, :])

                # Calculate loss and update network parameters
                loss = loss_fcn(output, target_batch[start_i:start_i + up_fr, :, :])
                loss.backward()
                optim.step()

                # Set the network hidden state, to detach it from the computation graph
                self.detach_hidden()
                self.zero_grad()

                # Update the start index for the next iteration and add the loss to the batch_loss total
                start_i += up_fr
                batch_loss += loss

            # Add the average batch loss to the epoch loss and reset the hidden states to zeros
            ep_loss += batch_loss / (k + 1)
            self.reset_hidden()
        return ep_loss / (batch_i + 1)

    # only proc processes a the input data and calculates the loss, optionally grad can be tracked or not
    def process_data(self, input_data, target_data, loss_fcn, chunk, grad=False):
        with (torch.no_grad() if not grad else nullcontext()):
            output = torch.empty_like(target_data)
            for l in range(int(output.size()[0] / chunk)):
                output[l * chunk:(l + 1) * chunk] = self(input_data[l * chunk:(l + 1) * chunk])
                self.detach_hidden()
            # If the data set doesn't divide evenly into the chunk length, process the remainder
            if not (output.size()[0] / chunk).is_integer():
                output[(l + 1) * chunk:-1] = self(input_data[(l + 1) * chunk:-1])
            self.reset_hidden()
            loss = loss_fcn(output, target_data)
        return output, loss



class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,n_skip,n_residue,kernel_size,dilation):
        super(ConvBlock,self).__init__()
        self.filter_conv = nn.Conv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,padding=int((dilation*(kernel_size-1))/2),dilation=dilation,bias=False)
        self.filter_act = nn.Tanh()
        self.gate_conv = nn.Conv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,padding=int((dilation*(kernel_size-1))/2),dilation=dilation,bias=False)
        self.gate_act = nn.Sigmoid()

        self.skip_scale = nn.Conv1d(in_channels=out_channels, out_channels=n_skip, kernel_size=1)
        self.residue_scale = nn.Conv1d(in_channels=out_channels, out_channels=n_residue, kernel_size=1)
    def forward(self,x):
        filter = self.filter_conv(x)
        filter = self.filter_act(filter)
        gate = self.gate_conv(x)
        gate = self.gate_act(gate)
        x = filter*gate

        skip = self.skip_scale(x)
        residue = self.residue_scale(x)

        return residue,skip

class WN_Encode(nn.Module):
    def __init__(self):
        super(WN_Encode,self).__init__()

        self.start_chan = 1
        self.num_layers = 4
        self.in_channel = 32
        self.out_channel = 256
        self.n_skip = 256
        self.n_residue = self.in_channel
        self.kernel_size = 2
        self.dilation = 2

        self.start_conv =  nn.Conv1d(in_channels=self.start_chan,out_channels=self.in_channel,kernel_size=1,bias=False)
        self.convblocks = nn.ModuleList()
        for i in range(self.num_layers):
            self.convblocks.append(ConvBlock(self.in_channel,self.out_channel,self.n_skip,self.n_residue,self.kernel_size,self.dilation))

        self.end_n = self.n_skip
        self.end_conv =  nn.Conv1d(in_channels=self.n_skip,out_channels=self.end_n,kernel_size=1,bias=False)
        self.end_activate = nn.Tanh()
    def forward(self,x):
        #x = self.One_hot(x).unsqueeze(0).transpose(1,2)
        x = self.start_conv(x)
        skip_tot = None
        for i in range(self.num_layers):
            residue,skip = self.convblocks[i](x)
            if type(skip_tot) != type(None):
                #skip_tot = skip_tot[:,:,-skip.size(2):]
                skip_tot +=skip
            else:
                skip_tot = skip
            #print(x.shape)
            #print(residue.shape)
            x = residue + x #[:,:,self.dilation*(self.kernel_size-1):]
        x = self.end_conv(skip_tot)
        x = self.end_activate(x)
        return x

class WN_Decode(nn.Module):
    def __init__(self,num_embeddings):
        super(WN_Decode,self).__init__()

        self.start_chan = num_embeddings
        self.num_layers = 4
        self.in_channel = 32
        self.out_channel = num_embeddings
        self.n_skip =  num_embeddings
        self.n_residue = self.in_channel
        self.kernel_size = 2
        self.dilation = 2

        self.start_conv =  nn.Conv1d(in_channels=self.start_chan,out_channels=self.in_channel,kernel_size=1,bias=False)
        self.convblocks = nn.ModuleList()
        for i in range(self.num_layers):
            self.convblocks.append(ConvBlock(self.in_channel,self.out_channel,self.n_skip,self.n_residue,self.kernel_size,self.dilation))

        self.end_n = 1
        self.end_conv =  nn.Conv1d(in_channels=self.n_skip,out_channels=self.end_n,kernel_size=1,bias=False)
        self.end_activate = nn.Tanh()
    def forward(self,x):
        #x = self.One_hot(x).unsqueeze(0).transpose(1,2)
        x = self.start_conv(x)
        skip_tot = None
        for i in range(self.num_layers):
            residue,skip = self.convblocks[i](x)
            if type(skip_tot) != type(None):
                #skip_tot = skip_tot[:,:,-skip.size(2):]
                skip_tot +=skip
            else:
                skip_tot = skip
            #print(x.shape)
            #print(residue.shape)
            x = residue + x #[:,:,self.dilation*(self.kernel_size-1):]
        x = self.end_conv(skip_tot)
        x = self.end_activate(x)
        return x

class WaveNet(nn.Module):
    def __init__(self):
        super(WaveNet,self).__init__()
        self.encode = WN_Encode()
 
        end_n = 256
        self.embedding_dim = 128
        self.num_classes=2

        self.pre_emb_conv = nn.Conv1d(in_channels=end_n,out_channels=self.embedding_dim,kernel_size=1,stride=1)

        self.num_embeddings = 256
        commitment_cost = .25
        decay = .99 
        self.vq_vae = VQEMA(num_embeddings=self.num_embeddings,embedding_dim=self.embedding_dim,commitment_cost=commitment_cost,decay=decay)

        self.decoder = WN_Decode(self.embedding_dim+1)



    def forward(self,x,labels):
        encode = self.encode(x)
        x_recon = self.pre_emb_conv(encode)


        loss,quantized,perplexity,_  = self.vq_vae(x_recon)
        #one_hot_y = labels.repeat((1,x_recon.shape[2],1)).permute(2,0,1) #torch.eye(self.num_classes, device=labels.device)[labels]
        one_hot_y = labels.repeat((1,quantized.shape[2],1)).permute(2,0,1) #torch.eye(self.num_classes, device=labels.device)[labels]
        #print(one_hot_y.shape,quantized.shape)
        quantized = torch.cat([quantized, one_hot_y], 1)

        #print(quantized[0])
        #print(quantized.shape)
        
        x_recon = self.decoder(quantized)
        x_recon = torch.tanh(x_recon)

        return  x_recon #,loss, perplexity
    
'''class LinSpace(nn.Module):
    def __init__(self,):
        super(LinSpace,self).__init__()
        self.flat = nn.Flatten(start_dim=1)
        self.l1 = nn.LazyLinear(256)
        self.l2 = nn.LazyLinear(self.l1.weight.shape[1])
    def forward(self,x):
        n_chans = x.shape[1]
        n_ls = x.shape[2]
        x = self.flat(x)
        x = self.l1(x)
        n_chans2 = x.shape[1]
        x = self.l2(x)
        print(n_chans,n_chans2)
        wad
        return x'''



class VQEMA(nn.Module):
    def __init__(self,num_embeddings,embedding_dim,commitment_cost,decay,epsilon=1e-5):
        super(VQEMA,self).__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        self.embedding = nn.Embedding(self.num_embeddings,self.embedding_dim)
        self.embedding.weight.data.normal_()

        self.register_buffer('ema_cluster_size', torch.zeros(num_embeddings))
        
        self.ema_w = nn.Parameter(torch.Tensor(num_embeddings, self.embedding_dim))
        self.ema_w.data.normal_()

    def forward(self,inputs):
        inputs = inputs.permute(0,2,1).contiguous() #move from (M,C,L) to (M,L,C)
        input_shape = inputs.shape
        flat_input = inputs.view(-1,self.embedding_dim) # (-1,C)

        #calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(flat_input, self.embedding.weight.t()))

        #encoding
        encoding_indices = torch.argmin(distances,dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0],self.num_embeddings,device=inputs.device)
        encodings.scatter_(1,encoding_indices,1)

        #quantize
        quantized = torch.matmul(encodings,self.embedding.weight).view(input_shape)

        #use EMA
        if self.training:
            self.ema_cluster_size = self.ema_cluster_size * self.decay + (1 - self.decay) * torch.sum(encodings, 0)
                        
            # Laplace smoothing of the cluster size
            n = torch.sum(self.ema_cluster_size.data)
            self.ema_cluster_size = ((self.ema_cluster_size + self.epsilon) / (n + self.num_embeddings * self.epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self.ema_w = nn.Parameter(self.ema_w * self.decay + (1 - self.decay) * dw)
            
            self.embedding.weight = nn.Parameter(self.ema_w / self.ema_cluster_size.unsqueeze(1))
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self.commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from MLC -> MCL
        return loss, quantized.permute(0, 2, 1).contiguous(), perplexity, encodings