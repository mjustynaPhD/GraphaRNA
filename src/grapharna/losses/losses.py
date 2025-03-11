import torch
import torch.nn.functional as F
from grapharna.utils import Sampler

def p_losses(denoise_model,
             x_data,
             seqs,
             t,
             sampler: Sampler,
             loss_type="huber",
             noise=None,
             mask:list=None,
             ):

    x_start = x_data.x.contiguous()  # Get the position of the atoms. First 3 features are the coordinates
    if noise is None:
        noise = torch.randn_like(x_start)
    x_noisy = sampler.q_sample(x_start=x_start,
                               t=t,
                               noise=noise,
                               )
    
    # mask = torch.ones(x_start.shape[0]).bool()
    if mask is None:
        mask = torch.ones(x_start.shape[0]).bool()
    else:
        mask = mask.bool()

    
    x_noisy = torch.cat((x_noisy[:,:3], x_data.x[:,3:]), dim=1)
    x_data.x = x_noisy
    predicted_noise = denoise_model(x_data, seqs, t)

    noise[:, 3:] = x_start[:, 3:]  # masked coords

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise[mask], predicted_noise[mask])
        # loss_copy = F.smooth_l1_loss(noise[mask, 3:], predicted_noise[mask, 3:])
        # loss_denoise = F.smooth_l1_loss(noise[mask, :3], predicted_noise[mask, :3])
        # loss = 0.3 * loss_copy + 0.7 * loss_denoise
        # loss = loss_copy + loss_denoise
    else:
        raise NotImplementedError()

    return loss, loss