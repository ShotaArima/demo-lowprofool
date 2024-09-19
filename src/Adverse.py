# Misc
import numpy as np

# Pytorch
import torch
import torch.nn as nn
from torch.autograd import Variable


# Clipping function
def clip(current, low_bound, up_bound):
    # low_bound = torch.FloatTensor(low_bound)
    # up_bound = torch.FloatTensor(up_bound)

    low_bound = low_bound.to(current.dtype)
    up_bound = up_bound.to(current.dtype)

    # サイズが一致していることを確認
    # assert current.size() == low_bound.size() == up_bound.size(), "Sizes must match"
    assert current.size() == low_bound.size() == up_bound.size(), f"Sizes must match: current {current.size()}, low_bound {low_bound.size()}, up_bound {up_bound.size()}"

    clipped = torch.max(torch.min(current, up_bound), low_bound)
    return clipped

# def expand_bounds(bounds, target_size):
#     expanded = []
#     for value in bounds:
#         if len(expanded) < target_size:
#             expanded.append(value)
#         else:
#             expanded.extend([0, 1])  # カテゴリカル変数の境界
#     return torch.tensor(expanded)

def expand_bounds(bounds, target_size):
    expanded = bounds.copy()
    while len(expanded) < target_size:
        expanded.append(0)  # カテゴリカル変数の下限
    return expanded

def expand_max_bounds(bounds, target_size):
    expanded = bounds.copy()
    while len(expanded) < target_size:
        expanded.append(1)  # カテゴリカル変数の上限
    return expanded


def lowProFool(x, model, weights, bounds, maxiters, alpha, lambda_):
    """
    Generates an adversarial examples x' from an original sample x

    :param x: tabular sample
    :param model: neural network
    :param weights: feature importance vector associated with the dataset at hand
    :param bounds: bounds of the datasets with respect to each feature
    :param maxiters: maximum number of iterations ran to generate the adversarial examples
    :param alpha: scaling factor used to control the growth of the perturbation
    :param lambda_: trade off factor between fooling the classifier and generating imperceptible adversarial example
    :return: original label prediction, final label prediction, adversarial examples x', iteration at which the class changed
    """

    min_bounds = []
    max_bounds = []
    for feature, info in bounds.items():
        if info['type'] == 'numeric':
            min_bounds.append(info['min'])
            max_bounds.append(info['max'])
        elif info['type'] == 'categorical':
            min_bounds.extend([0]*len(info['values']))
            max_bounds.extend([1]*len(info['values']))
    print("Original min_bounds", min_bounds)
    print("Original max_bounds", max_bounds)
    
    # onr-hot encoding
    min_bounds = expand_bounds(min_bounds, x.size(0))
    max_bounds = expand_max_bounds(max_bounds, x.size(0))
    print("min_bounds", min_bounds)
    print("max_bounds", max_bounds)

    min_bounds = torch.as_tensor(min_bounds, dtype=torch.float32)
    max_bounds = torch.as_tensor(max_bounds, dtype=torch.float32)

    # r = Variable(torch.FloatTensor(1e-4 * np.ones(x.numpy().shape)), requires_grad=True) 
    r = torch.zeros_like(x, requires_grad=True)
    v = torch.tensor(weights, dtype=torch.float32)
    v = v.expand(x.size()) 

    print("r after initialization:", r)
    print("min_bounds dtype:", min_bounds.dtype)
    print("max_bounds dtype:", max_bounds.dtype)
    print("x dtype:", x.dtype)
    print("r dtype:", r.dtype)
    
    output = model(x + r)
    if output.dim() == 1:
        output = output.unsqueeze(0)
    
    probs = torch.sigmoid(output)
    orig_pred = (probs > 0.5).long().cpu().numpy().squeeze(0)
    target_pred = 1 - orig_pred

    target = torch.zeros_like(output)
    target[:, target_pred] = 1
    
    lambda_ = torch.tensor([lambda_])
    
    bce = nn.BCEWithLogitsLoss()
    l2 = lambda v, r: torch.sqrt(torch.sum(torch.mul(v * r,v * r))) #L2 norm

    best_norm_weighted = float('inf')
    best_pert_x = x
    
    loop_i, loop_change_class = 0, 0
    while loop_i < maxiters:
            
        # Zero the gradient
        # r.grad = None
        if r.grad is not None:
            r.grad.zero_()
        else:
            r.requires_grad_(True)

        # Computing loss 
        loss_1 = bce(output, target)
        loss_2 = l2(v, r)
        loss = loss_1 + lambda_ * loss_2

        # Get the gradient
        loss.backward(retain_graph=True)
        grad_r = r.grad.data.cpu().numpy().copy()
        
        # Guide perturbation to the negative of the gradient
        ri = - grad_r
    
        # limit huge step
        ri *= alpha

        # Adds new perturbation to total perturbation
        r = r.clone().detach().cpu().numpy() + ri
        
        # For later computation
        r_norm_weighted = np.sum(np.abs(r * weights))
        
        # Ready to feed the model
        r = Variable(torch.FloatTensor(r), requires_grad=True) 
        
        print("x shape:", x.shape)
        print("x:", x)
        print("r shape:", r.shape)
        print("r:", r)
        # Compute adversarial example
        xprime = x + r

        # デバッグ
        print("min_bound", min_bounds)
        print("min_bound.shape", min_bounds.shape)
        print("max_bound", max_bounds)
        print("max_bound.shape", max_bounds.shape)
        print("xprime", xprime)
        print("xprime.shape", xprime.shape)
        xprime = clip(xprime, min_bounds, max_bounds)
        
        # Classify adversarial example
        output = model(xprime)
        if output.dim() == 1:
            output = output.unsqueeze(0)

        probs = torch.sigmoid(output)
        output_pred = (probs > 0.5).long().cpu().numpy().squeeze()
        
        # Keep the best adverse at each iterations
        if not np.array_equal(output_pred, orig_pred) and r_norm_weighted < best_norm_weighted:
            best_norm_weighted = r_norm_weighted
            best_pert_x = xprime

        if np.array_equal(output_pred, orig_pred):
            loop_change_class += 1
            
        loop_i += 1
        
    # Clip at the end no matter what
    best_pert_x = clip(best_pert_x, min_bounds, max_bounds)
    output = model.forward(best_pert_x)
    if output.dim() == 1:
        output = output.unsqueeze(1)
    probs = torch.sigmoid(output)
    output_pred = output.max(0, keepdim=True)[1].cpu().numpy().squeeze()

    return orig_pred.item(), output_pred.item(), best_pert_x.detach().cpu().numpy().squeeze(), loop_change_class 

# Forked from https://github.com/LTS4/DeepFool
def deepfool(x_old, net, maxiters, alpha, bounds, weights=[], overshoot=0.002):
    """
    :param image: tabular sample
    :param net: network 
    :param maxiters: maximum number of iterations ran to generate the adversarial examples
    :param alpha: scaling factor used to control the growth of the perturbation
    :param bounds: bounds of the datasets with respect to each feature
    :param weights: feature importance vector associated with the dataset at hand
    :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
    :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    
    input_shape = x_old.numpy().shape
    x = x_old.clone()
    x = Variable(x, requires_grad=True)
    
    min_bounds = []
    max_bounds = []
    for feature, info in bounds.items():
        if info['type'] == 'numeric':
            min_bounds.append(info['min'])
            max_bounds.append(info['max'])
        elif info['type'] == 'categorical':
            min_bounds.extend([0]*len(info['values']))
            max_bounds.extend([1]*len(info['values']))
    
    min_bounds = torch.FloatTensor(min_bounds)
    max_bounds = torch.FloatTensor(max_bounds)

    output = net(x)
    if output.dim() == 1:
        output = output.unsqueeze(0)
    probs = torch.sigmoid(output)
    orig_pred = (probs > 0.5).long().item()

    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0
    while loop_i < maxiters:
        # Origin class
        output = net(x)
        if output.dim() == 1:
            output = output.unsqueeze(0)
        probs = torch.sigmoid(output)

        probs.backward(retain_graph=True)
        grad_orig = x.grad.data.numpy().copy()
        
        x.grad.zero_()

        # Target class
        (1-probs).backward(retain_graph=True)
        
        cur_grad = x.grad.data.numpy().copy()
        
        # set new w and new f
        w = cur_grad - grad_orig
        f = ((1-probs) - probs).data.numpy()

        pert = abs(f)/np.linalg.norm(w.flatten())
    
        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / np.linalg.norm(w)   
        
        if len(weights) > 0:
            r_i *= np.array(weights)

        # limit huge step
        r_i = alpha * r_i / np.linalg.norm(r_i) 
            
        r_tot = np.float32(r_tot + r_i)
        
        pert_x = x_old + (1 + overshoot) * torch.from_numpy(r_tot)
        pert_x = clip(pert_x, min_bounds, max_bounds)

        x = Variable(pert_x, requires_grad=True)

        output = net(x)
        if output.dim() == 1:
            output = output.unsqueeze(0)
        probs = torch.sigmoid(output)
        k_i = (probs > 0.5).long().item()

        if k_i != orig_pred:
            break
                    
        loop_i += 1

    r_tot = (1+overshoot)*r_tot    
    pert_x = clip(pert_x, min_bounds, max_bounds)

    return orig_pred, k_i, pert_x.cpu(), loop_i