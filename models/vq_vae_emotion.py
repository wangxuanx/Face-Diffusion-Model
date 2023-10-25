import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lib.base_models import Transformer, LinearEmbedding, PositionalEncoding, BaseModel


class VQAutoEncoder(BaseModel):
    """ VQ-GAN model """

    def __init__(self, args):
        super().__init__()
        self.encoder = TransformerEncoder(args)
        self.decoder = TransformerDecoder(args, args.in_dim)
        self.quantize = VectorQuantizer(args.n_embed,
                                        args.zquant_dim,
                                        beta=0.25)
        self.args = args

    def encode(self, x, one_hot):
        h = self.encoder(x, one_hot) ## x --> z'
        h = h.view(x.shape[0], -1, self.args.face_quan_num, self.args.zquant_dim)
        h = h.view(x.shape[0], -1, self.args.zquant_dim)
        # quant, emb_loss, info = self.quantize(h) ## finds nearest quantization
        # return quant, emb_loss, info
        return h
    
    ## finds nearest quantization
    def quant(self, x, one_hot):
        quanted, emb_loss, info = self.quantize(x, one_hot) ## finds nearest quantization
        return quanted, emb_loss, info

    def decode(self, quant):
        #BCL
        quant = quant.permute(0,2,1)
        quant = quant.reshape(quant.shape[0], -1, self.args.face_quan_num, self.args.zquant_dim).contiguous()
        quant = quant.reshape(quant.shape[0], -1,  self.args.face_quan_num*self.args.zquant_dim).contiguous()
        quant = quant.permute(0,2,1).contiguous()
        dec = self.decoder(quant) ## z' --> x

        return dec

    def forward(self, x, template, one_hot):
        x = x - template  # 减去模板获得运动

        ### x.shape: [B, L C]
        h = self.encode(x, one_hot)
        ### quant [B, C, L]
        quanted, emb_loss, info = self.quant(h, one_hot)

        dec = self.decode(quanted)
        dec = dec + template
        return dec, emb_loss, info

    def sample_step(self, x, x_a=None):
        quant_z, _, info = self.encode(x, x_a)
        x_sample_det = self.decode(quant_z)
        btc = quant_z.shape[0], quant_z.shape[2], quant_z.shape[1]
        indices = info[2]
        x_sample_check = self.decode_to_img(indices, btc)
        return x_sample_det, x_sample_check

    def get_quant(self, x, x_a=None):
        h = self.encode(x, x_a)
        quanted, emb_loss, info = self.quant(h)
        indices = info[2]
        return quanted, indices

    def get_distances(self, x):
        h = self.encoder(x) ## x --> z'
        d = self.quantize.get_distance(h)
        return d

    def get_quant_from_d(self, d, btc):
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        x = self.decode_to_img(min_encoding_indices, btc)
        return x

    @torch.no_grad()
    def entry_to_feature(self, index, zshape):
        index = index.long()
        quant_z = self.quantize.get_codebook_entry(index.reshape(-1),
                                                   shape=None)
        quant_z = torch.reshape(quant_z, zshape)
        return quant_z



    @torch.no_grad()
    def decode_to_img(self, index, zshape):
        index = index.long()
        quant_z = self.quantize.get_codebook_entry(index.reshape(-1),
                                                   shape=None)
        quant_z = torch.reshape(quant_z, zshape).permute(0,2,1) # B L 1 -> B L C -> B C L
        x = self.decode(quant_z)
        return x

    @torch.no_grad()
    def decode_logit(self, logits, zshape):
        if logits.dim() == 3:
            probs = F.softmax(logits, dim=-1)
            _, ix = torch.topk(probs, k=1, dim=-1)
        else:
            ix = logits
        ix = torch.reshape(ix, (-1,1))
        x = self.decode_to_img(ix, zshape)
        return x

    def get_logit(self, logits, sample=True, filter_value=-float('Inf'),
                  temperature=0.7, top_p=0.9, sample_idx=None):
        """ function that samples the distribution of logits. (used in test)
        if sample_idx is None, we perform nucleus sampling
        """
        logits = logits / temperature
        sample_idx = 0

        probs = F.softmax(logits, dim=-1) # B, N, embed_num
        if sample:
            ## multinomial sampling
            shape = probs.shape
            probs = probs.reshape(shape[0]*shape[1],shape[2])
            ix = torch.multinomial(probs, num_samples=sample_idx+1)
            probs = probs.reshape(shape[0],shape[1],shape[2])
            ix = ix.reshape(shape[0],shape[1])
        else:
            ## top 1; no sampling
            _, ix = torch.topk(probs, k=1, dim=-1)
        return ix, probs


class TransformerEncoder(nn.Module):
  """ Encoder class for VQ-VAE with Transformer backbone """

  def __init__(self, args):
    super().__init__()
    self.args = args
    size = self.args.in_dim
    dim = self.args.hidden_size
    self.vertice_mapping = nn.Sequential(nn.Linear(size, dim), nn.LeakyReLU(self.args.neg, True))
    self.emotion_mapping = nn.Sequential(nn.Linear(7, dim), nn.LeakyReLU(self.args.neg, True))
    if args.quant_factor == 0:
        layers = [nn.Sequential(
                    nn.Conv1d(dim,dim,5,stride=1,padding=2,
                                padding_mode='replicate'),
                    nn.LeakyReLU(self.args.neg, True),
                    nn.InstanceNorm1d(dim, affine=args.INaffine)
                    )]
    else:
        layers = [nn.Sequential(
                    nn.Conv1d(dim,dim,5,stride=2,padding=2,
                                padding_mode='replicate'),
                    nn.LeakyReLU(self.args.neg, True),
                    nn.InstanceNorm1d(dim, affine=args.INaffine)
                    )] 

        for _ in range(1, args.quant_factor):
            layers += [nn.Sequential(
                        nn.Conv1d(dim,dim,5,stride=1,padding=2,
                                    padding_mode='replicate'),
                        nn.LeakyReLU(self.args.neg, True),
                        nn.InstanceNorm1d(dim, affine=args.INaffine), 
                        nn.MaxPool1d(2)
                        )] 
    self.squasher = nn.Sequential(*layers)
    self.encoder_transformer = Transformer(
        in_size=self.args.hidden_size,
        hidden_size=self.args.hidden_size,
        num_hidden_layers=\
                self.args.num_hidden_layers,
        num_attention_heads=\
                self.args.num_attention_heads,
        intermediate_size=\
                self.args.intermediate_size)
    self.encoder_pos_embedding = PositionalEncoding(
        self.args.hidden_size)
    self.encoder_linear_embedding = LinearEmbedding(
        self.args.hidden_size,
        self.args.hidden_size)
    
    # turn the channel number back to args.face_quan_num*args.zquant_dim
    self.encoder_linear_embedding_post = LinearEmbedding(
        self.args.hidden_size,
        self.args.face_quan_num*self.args.zquant_dim)

  def forward(self, inputs, emotions):
    ## downdample into path-wise length seq before passing into transformer
    dummy_mask = {'max_mask': None, 'mask_index': -1, 'mask': None}
    inputs = self.vertice_mapping(inputs)
    emotions = self.emotion_mapping(emotions)
    inputs = inputs + emotions
    inputs = self.squasher(inputs.permute(0,2,1)).permute(0,2,1) # [N L C]

    encoder_features = self.encoder_linear_embedding(inputs)
    encoder_features = self.encoder_pos_embedding(encoder_features)
    encoder_features = self.encoder_transformer((encoder_features, dummy_mask))
    encoder_features = self.encoder_linear_embedding_post(encoder_features)
    return encoder_features
  

class VectorQuantizer(nn.Module):
    """
    see https://github.com/MishaLaskin/vqvae/blob/d761a999e2267766400dc646d82d3ac3657771d4/models/quantizer.py
    ____________________________________________
    Discretization bottleneck part of the VQ-VAE.
    Inputs:
    - n_e : number of embeddings
    - e_dim : dimension of embedding
    - beta : commitment cost used in loss term, beta * ||z_e(x)-sg[e]||^2
    _____________________________________________
    """

    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)  # 将特征向量映射到embedding空间：n_e * e_dim
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z, one_hot):
        z_flattened = z.reshape(-1, self.e_dim)  # 将z展平
        pos = torch.argmax(one_hot)
        # indices = torch.tensor(range(pos * 256, (pos + 1) * 256)).to(one_hot.device)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) \
            + torch.sum(self.embedding.weight[pos * 256: (pos + 1) * 256]**2, dim=1) \
            - 2 * torch.matmul(z_flattened, self.embedding.weight[pos * 256: (pos + 1) * 256].t())  # 计算z和embedding的距离
        # d1 = torch.sum(z_flattened ** 2, dim=1, keepdim=True)
        # d2 = torch.sum(self.embedding.weight**2, dim=1)
        # d3 = torch.matmul(z_flattened, self.embedding.weight.t())

        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)  # 获得最接近的索引K值，每行有一个 1 其余皆为0， 矩阵共有 K 列，同1列上可以有多个1。
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e // 7).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embedding.weight[pos * 256: (pos + 1) * 256]).view(z.shape)

        # compute loss for embedding
        loss = self.beta * torch.mean((z_q.detach()-z)**2) + torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # perplexity
        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        # reshape back to match original input shape
        z_q = z_q.permute(0, 2, 1).contiguous()
        return z_q, loss, (perplexity, min_encodings, min_encoding_indices)

    def get_distance(self, z):
        z = z.permute(0, 2, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.embedding.weight.t())
        d = torch.reshape(d, (z.shape[0], -1, z.shape[2])).permute(0,2,1).contiguous()
        return d

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        min_encodings = torch.zeros(indices.shape[0], self.n_e).to(indices)
        min_encodings.scatter_(1, indices[:,None], 1)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings.float(), self.embedding.weight)

        if shape is not None:
            z_q = z_q.view(shape)

        return z_q


class TransformerDecoder(nn.Module):
  """ Decoder class for VQ-VAE with Transformer backbone """

  def __init__(self, args, out_dim, is_audio=False):
    super().__init__()
    self.args = args
    size=self.args.hidden_size
    dim=self.args.hidden_size
    self.expander = nn.ModuleList()
    if args.quant_factor == 0:
        self.expander.append(nn.Sequential(
                    nn.Conv1d(size,dim,5,stride=1,padding=2,
                                padding_mode='replicate'),
                    nn.LeakyReLU(self.args.neg, True),
                    nn.InstanceNorm1d(dim, affine=args.INaffine)
                            ))
    else:
        self.expander.append(nn.Sequential(
                    nn.ConvTranspose1d(size,dim,5,stride=2,padding=2,
                                        output_padding=1,
                                        padding_mode='replicate'),
                    nn.LeakyReLU(self.args.neg, True),
                    nn.InstanceNorm1d(dim, affine=args.INaffine)
                            ))                      
        num_layers = args.quant_factor+2 \
            if is_audio else args.quant_factor

        for _ in range(1, num_layers):
            self.expander.append(nn.Sequential(
                                nn.Conv1d(dim,dim,5,stride=1,padding=2,
                                        padding_mode='replicate'),
                                nn.LeakyReLU(self.args.neg, True),
                                nn.InstanceNorm1d(dim, affine=args.INaffine),
                                ))

    self.decoder_transformer = Transformer(
        in_size=self.args.hidden_size,
        hidden_size=self.args.hidden_size,
        num_hidden_layers=\
            self.args.num_hidden_layers,
        num_attention_heads=\
            self.args.num_attention_heads,
        intermediate_size=\
            self.args.intermediate_size)
    self.decoder_pos_embedding = PositionalEncoding(
        self.args.hidden_size)
    self.decoder_linear_embedding = LinearEmbedding(
        self.args.hidden_size,
        self.args.hidden_size)
    self.decoder_linear_embedding_pre = LinearEmbedding(
        self.args.face_quan_num*self.args.zquant_dim,
        self.args.hidden_size)


    self.vertice_map_reverse = nn.Linear(args.hidden_size,out_dim, bias=False)

  def forward(self, inputs):
    dummy_mask = {'max_mask': None, 'mask_index': -1, 'mask': None}
    ## upsample into original length seq before passing into transformer
    inputs = inputs.permute(0,2,1) #BLC
    inputs = self.decoder_linear_embedding_pre(inputs)
    inputs = inputs.permute(0,2,1) #BCL
    for i, module in enumerate(self.expander):
        inputs = module(inputs)
        if i > 0:
            inputs = inputs.repeat_interleave(2, dim=2)

    inputs = inputs.permute(0,2,1) #BLC
    decoder_features = self.decoder_linear_embedding(inputs)
    decoder_features = self.decoder_pos_embedding(decoder_features)

    decoder_features = self.decoder_transformer((decoder_features, dummy_mask))
    pred_recon = self.vertice_map_reverse(decoder_features)
    return pred_recon