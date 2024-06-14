import argparse

# This is the condition vqvae
def vq_vae_args():
    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument('--vqvae_pretrained_path', type=str, default='/data/WX/video-diffusion-pytorch/checkpoints/vqvae/vqvae_100.pt', help='path of the pretrained vqvae')
    parser.add_argument('--n_embed', type=int, default=256 * 7, help='number of hidden units')
    parser.add_argument('--zquant_dim', type=int, default=64, help='number of residual hidden units')
    parser.add_argument('--in_dim', type=int, default=5023*3, help='number of input channels')
    parser.add_argument('--hidden_size', type=int, default=1024, help='number of hidden units')
    parser.add_argument('--neg', type=int, default=0.2, help='number of negative samples')
    parser.add_argument('--quant_factor', type=float, default=0, help='number of quantization factor')
    parser.add_argument('--INaffine', type=bool, default=False, help='number of INaffine')
    parser.add_argument('--num_hidden_layers', type=int, default=6, help='number of hidden layers')
    parser.add_argument('--num_attention_heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--intermediate_size', type=int, default=1536, help='number of intermediate size')
    parser.add_argument('--face_quan_num', type=int, default=8, help='number of face quantization')

    args = parser.parse_args()
    return args


# This is the origin vqvae
def origin_vq_vae_args():
    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument('--vqvae_pretrained_path', type=str, default='/data/WX/video-diffusion-pytorch/checkpoints/vqvae/vqvae_100.pt', help='path of the pretrained vqvae')
    parser.add_argument('--n_embed', type=int, default=256, help='number of hidden units')
    parser.add_argument('--zquant_dim', type=int, default=64, help='number of residual hidden units')
    parser.add_argument('--in_dim', type=int, default=5023*3, help='number of input channels')
    parser.add_argument('--hidden_size', type=int, default=1024, help='number of hidden units')
    parser.add_argument('--neg', type=int, default=0.2, help='number of negative samples')
    parser.add_argument('--quant_factor', type=float, default=0, help='number of quantization factor')
    parser.add_argument('--INaffine', type=bool, default=False, help='number of INaffine')
    parser.add_argument('--num_hidden_layers', type=int, default=6, help='number of hidden layers')
    parser.add_argument('--num_attention_heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--intermediate_size', type=int, default=1536, help='number of intermediate size')
    parser.add_argument('--face_quan_num', type=int, default=8, help='number of face quantization')

    args = parser.parse_args()
    return args


# This is the biwi vqvae
def biwi_vq_vae_args():
    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument('--vqvae_pretrained_path', type=str, default='/data/WX/video-diffusion-pytorch/checkpoints/vqvae/vqvae_100.pt', help='path of the pretrained vqvae')
    parser.add_argument('--n_embed', type=int, default=256, help='number of hidden units')
    parser.add_argument('--zquant_dim', type=int, default=128, help='number of residual hidden units')
    parser.add_argument('--in_dim', type=int, default=70110, help='number of input channels')
    parser.add_argument('--hidden_size', type=int, default=1024, help='number of hidden units')
    parser.add_argument('--neg', type=int, default=0.2, help='number of negative samples')
    parser.add_argument('--quant_factor', type=float, default=0, help='number of quantization factor')
    parser.add_argument('--INaffine', type=bool, default=False, help='number of INaffine')
    parser.add_argument('--num_hidden_layers', type=int, default=6, help='number of hidden layers')
    parser.add_argument('--num_attention_heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--intermediate_size', type=int, default=1536, help='number of intermediate size')
    parser.add_argument('--face_quan_num', type=int, default=8, help='number of face quantization')

    args = parser.parse_args()
    return args


# This is the vocaset vqvae
def vocaset_vq_vae_args():
    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument('--vqvae_pretrained_path', type=str, default='/data/WX/video-diffusion-pytorch/checkpoints/vqvae/vqvae_100.pt', help='path of the pretrained vqvae')
    parser.add_argument('--n_embed', type=int, default=256, help='number of hidden units')
    parser.add_argument('--zquant_dim', type=int, default=64, help='number of residual hidden units')
    parser.add_argument('--in_dim', type=int, default=15069, help='number of input channels')
    parser.add_argument('--hidden_size', type=int, default=1024, help='number of hidden units')
    parser.add_argument('--neg', type=int, default=0.2, help='number of negative samples')
    parser.add_argument('--quant_factor', type=float, default=0, help='number of quantization factor')
    parser.add_argument('--INaffine', type=bool, default=False, help='number of INaffine')
    parser.add_argument('--num_hidden_layers', type=int, default=6, help='number of hidden layers')
    parser.add_argument('--num_attention_heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--intermediate_size', type=int, default=1536, help='number of intermediate size')
    parser.add_argument('--face_quan_num', type=int, default=16, help='number of face quantization')

    args = parser.parse_args()
    return args