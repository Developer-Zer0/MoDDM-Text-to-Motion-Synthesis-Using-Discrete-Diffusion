import torch
import torch.nn.functional as F
import numpy as np

def get_recon_data(output, loss_opts, pred_data_type='pred_data'):
    loss_type = loss_opts.get('loss_feat_type', 'jfeats')
    pred_feats = output[pred_data_type].get_feats(loss_type)
    gt_feats = output['gt_data'].get_feats(loss_type)
    return pred_feats, gt_feats

def l1_loss(output, loss_opts, pred_type):
    pred_feats, gt_feats = get_recon_data(output, loss_opts, pred_type)
    loss_fn = torch.nn.SmoothL1Loss()
    return loss_fn(pred_feats, gt_feats)

def compute_l1_data_loss(output, loss_opts):
    return l1_loss(output, loss_opts, 'pred_data')

def compute_l1_m2m_loss(output, loss_opts):
    return l1_loss(output, loss_opts, 'pred_m2m')

def compute_temos_kl_loss(output, loss_opts):
    #kl_fn = torch.nn.KLLoss()
    kl_fn = torch.distributions.kl_divergence
    dist_text = output['dist_text']
    dist_motion = output['dist_motion']
    dist_gt = output['dist_gt']
    kl_t2m = kl_fn(dist_text, dist_motion).mean()
    kl_m2t = kl_fn(dist_motion, dist_text).mean()
    kl_text = kl_fn(dist_text, dist_gt).mean()
    kl_motion = kl_fn(dist_motion, dist_gt).mean()
    kl_out_loss = kl_t2m + kl_m2t + kl_text + kl_motion
    return kl_out_loss

def compute_temos_latent_loss(output, loss_opts):
    latent_text = output['latent_text']
    latent_motion = output['latent_motion']
    loss_fn = torch.nn.SmoothL1Loss()
    return loss_fn(latent_text, latent_motion)

# seq2seq
def compute_mse_loss(output, loss_opts):
    pred_feats, gt_feats = get_recon_data(output, loss_opts)
    return F.mse_loss(pred_feats, gt_feats)

def compute_motion_continuous_loss(output, loss_opts):
    pred_feats, gt_feats = get_recon_data(output, loss_opts)
    n_element = pred_feats.numel()
    diff = [abs(pred_feats[:, n, :] - pred_feats[:, n-1, :]) for n in range(1, pred_feats.shape[1])]
    cont_loss = torch.sum(torch.stack(diff)) / n_element
    return cont_loss

# Guo
def compute_l_smooth_loss(output, loss_opts):
    pred_snippets = output['snippets_motion']
    loss_fn = torch.nn.L1Loss()
    return loss_fn(pred_snippets[:, 1:], pred_snippets[:, :-1])

def compute_l_sparsity_loss(output, loss_opts):
    pred_snippets = output['snippets_motion']
    return torch.mean(torch.abs(pred_snippets))

def compute_guo_kl_loss(output, loss_opts):
    sigma1 = output["logvars_post"].mul(0.5).exp()
    sigma2 = output["logvars_pri"].mul(0.5).exp()
    kld = torch.log(sigma2 / sigma1) + (torch.exp(output["logvars_post"]) + (output["mus_post"] - output["mus_pri"]) ** 2) / (
            2 * torch.exp(output["logvars_pri"])) - 1 / 2
    return kld.sum() / output["mus_post"].shape[0]

def compute_l1_snip2snip_loss(output, loss_opts):
    pred_snippets = output['pred_snippets']
    gt_snippets = output['gt_snippets']
    loss_fn = torch.nn.SmoothL1Loss()
    return loss_fn(pred_snippets, gt_snippets)

def compute_len_crossentropy_loss(output, loss_opts):
    pred_dis = output['pred_dis']
    gt_dis = output['gt_dis']
    loss_fn = torch.nn.CrossEntropyLoss()
    return loss_fn(pred_dis, gt_dis)

def compute_matching_loss(output, loss_opts):
    output1 = output['pred_data']
    output2 = output['gt_data']
    batch_size = output2.shape[0]
    '''Positive pairs'''
    pos_labels = torch.zeros(batch_size).to(output2.device)
    loss_pos = compute_contrastive_loss(output2, output1, pos_labels)
    '''Negative Pairs, shifting index'''
    neg_labels = torch.ones(batch_size).to(output2.device)
    shift = np.random.randint(0, batch_size - 1)
    new_idx = np.arange(shift, batch_size + shift) % batch_size
    output1 = output1.clone()[new_idx]
    loss_neg = compute_contrastive_loss(output2, output1, neg_labels)
    return loss_pos + loss_neg

def compute_contrastive_loss(output1, output2, label):
    margin = 10
    euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
    loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                  (label) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2))
    return loss_contrastive

# GuyTevet Diffusion
def compute_dummy(output, loss_opts):
    try:
        return torch.mean(output['losses']['loss'])  # Total loss taken and mean over batch dimension
    except IndexError:
        return torch.mean(output['losses'])

# VQ_VAE losses
def compute_codebook_loss(output, loss_opts):
    codebook_loss = torch.sum(output['codebook_loss']) # codebook loss is already computed during forward process
    return codebook_loss

def compute_entropy_loss(output, loss_opts):
    compute_entropy_loss = torch.sum(output['entropy']) # codebook loss is already computed during forward process
    return compute_entropy_loss

def compute_perplexity_loss(output, loss_opts):
    perplexity_loss = torch.sum(output['perplexity']) # codebook loss is already computed during forward process
    return perplexity_loss

# VQ_RNN losses
def compute_rnn_loss(output, loss_opts):
    # HARDCODED
    trg_pad_idx = 1023  # codebook_dim-1
    pred = output['pred_tokens']
    gold = output['gt_tokens']
    eps = 0.1
    n_class = pred.size(1)

    # one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
    # one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
    # log_prb = F.log_softmax(pred, dim=1)
    #
    # non_pad_mask = gold.ne(trg_pad_idx)
    # loss = -(one_hot * log_prb).sum(dim=1)
    # loss = loss.masked_select(non_pad_mask).sum()

    loss = F.cross_entropy(pred, gold)

    return loss

# MLD Loss
class KLLoss:
    def __init__(self):
        pass
    def __call__(self, q, p):
        div = torch.distributions.kl_divergence(q, p)
        return div.mean()
    def __repr__(self):
        return "KLLoss()"
def compute_mld_l1_feats_loss(output, loss_opts):
    rs_set = output['rs_set']
    recon_loss = torch.nn.SmoothL1Loss(reduction='mean')
    return recon_loss(rs_set['m_rst'], rs_set['m_ref'])

def compute_mld_l1_joints_loss(output, loss_opts):
    rs_set = output['rs_set']
    recon_loss = torch.nn.SmoothL1Loss(reduction='mean')
    return recon_loss(rs_set['joints_rst'], rs_set['joints_ref'])

def compute_mld_kl_loss(output, loss_opts):
    rs_set = output['rs_set']
    kl_loss = KLLoss()
    return kl_loss(rs_set['dist_m'], rs_set['dist_ref'])

_matching_ = {"l1_data": compute_l1_data_loss, "l1_m2m": compute_l1_m2m_loss, "temos_kl": compute_temos_kl_loss, "temos_latent": compute_temos_latent_loss,
              "mse_data": compute_mse_loss, "cont_data": compute_motion_continuous_loss,  "l_smooth": compute_l_smooth_loss, "l_sparsity": compute_l_sparsity_loss,
              "guo_kl": compute_guo_kl_loss, "l1_snip2snip": compute_l1_snip2snip_loss, "len_crossentropy": compute_len_crossentropy_loss, "l_matching": compute_matching_loss,
              "l_dummy": compute_dummy, "l_codebook": compute_codebook_loss, "l_entropy": compute_entropy_loss, "l_perplexity": compute_perplexity_loss, "l_rnn": compute_rnn_loss,
              "l_mldrecfeats": compute_mld_l1_feats_loss, "l_mldrecjoints": compute_mld_l1_joints_loss, "l_mldkl": compute_mld_kl_loss,}

def get_loss_function(ltype):
    return _matching_[ltype]

def get_loss_names():
    return list(_matching_.keys())
