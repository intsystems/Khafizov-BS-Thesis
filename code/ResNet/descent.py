import torch
from torch.func import functional_call

def mirror_descent(model, X_train, y_train, param_name, impact: torch.Tensor, lr, eta, lambda_value, num_steps, criterion):
    """
    Perform mirror descent optimization on a specified parameter of a model.

    Args:
        model (torch.nn.Module): The neural network model.
        X_train (torch.Tensor): The input training data.
        y_train (torch.Tensor): The target training data.
        param_name (str): The name of the parameter to be optimized.
        impact (torch.Tensor): The initial impact tensor.
        lr (float): Learning rate for the parameter update.
        eta (float): Learning rate for the impact update.
        lambda_value (float): Regularization parameter.
        num_steps (int): Number of optimization steps.
        criterion (callable): Loss function.

    Returns:
        torch.Tensor: The optimized impact tensor.
    """

    original_param = dict(model.named_parameters())[param_name]

    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    param_grad = torch.autograd.grad(loss, original_param, create_graph=True)[0]
    
    if impact is None:
        impact = param_grad.abs().clone().detach()
        impact /= impact.sum()
    
    impact = impact.clone().detach().requires_grad_(True)

    new_params = {name: param.clone() for name, param in model.named_parameters()}

    for _ in range(num_steps):
        # Update parameter using impact
        param_new = original_param - lr * impact * param_grad
        # Create new parameter dictionary
        new_params[param_name] = param_new
        # Compute outputs with new parameters
        outputs_new = functional_call(model, new_params, (X_train,))
        # Compute new loss
        loss_new = criterion(outputs_new, y_train)

        # Compute gradient of new loss w.r.t. impact
        grad_impact = torch.autograd.grad(loss_new, impact)[0]

        with torch.no_grad():
            impact_update = torch.pow(impact, 1/(1+eta*lambda_value)) * torch.exp(-(eta/(1+eta*lambda_value)) * (grad_impact))
            impact = impact_update / impact_update.sum()

        # Ensure impact requires grad for the next iteration
        impact.requires_grad_(True)

    return impact.detach()


def gradient_descent(model, X_train, y_train, param_name, impact: torch.Tensor, lr, eta, num_steps, criterion):
    """
    Performs gradient descent to optimize the impact tensor for a given model parameter.
    Args:
        model (torch.nn.Module): The neural network model.
        X_train (torch.Tensor): The input training data.
        y_train (torch.Tensor): The target training data.
        param_name (str): The name of the parameter to be optimized.
        impact (torch.Tensor): The initial impact tensor.
        lr (float): The learning rate for parameter update.
        eta (float): The learning rate for impact update.
        num_steps (int): The number of gradient descent steps.
        criterion (torch.nn.Module): The loss function.
    Returns:
        torch.Tensor: The optimized impact tensor.
    """
    original_param = dict(model.named_parameters())[param_name]

    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    param_grad = torch.autograd.grad(loss, original_param, create_graph=True)[0]

    if impact is None:
        impact = torch.zeros_like(param_grad)
        topk_indices = torch.topk(param_grad.abs().view(-1), int(0.001 * param_grad.numel())).indices
        impact.view(-1)[topk_indices] = 1.0
    
    impact = impact.clone().detach().requires_grad_(True)
    
    new_params = {name: param.clone() for name, param in model.named_parameters()}

    for _ in range(num_steps):
        # Update parameter using impact
        param_new = original_param - lr * impact * param_grad
        # Create new parameter dictionary
        new_params[param_name] = param_new
        # Compute outputs with new parameters
        outputs_new = functional_call(model, new_params, (X_train,))
        # Compute new loss
        loss_new = criterion(outputs_new, y_train)

        # Compute gradient of new loss w.r.t. impact
        grad_impact = torch.autograd.grad(loss_new, impact)[0]

        with torch.no_grad():
            impact -= eta * lr * grad_impact
            impact = torch.clip(impact, 0, 1)
        
        # Ensure impact requires grad for the next iteration
        impact.requires_grad_(True)

    return impact.detach()