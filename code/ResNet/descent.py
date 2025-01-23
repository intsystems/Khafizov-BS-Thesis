import torch
from torch.func import functional_call

def mirror_descent(model, X_train, y_train, param_name, impact: torch.Tensor, lr, eta, lambda_value, num_steps, criterion):
    impact = impact.clone().detach().requires_grad_(True)
    original_param = dict(model.named_parameters())[param_name]

    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    param_grad = torch.autograd.grad(loss, original_param, create_graph=True)[0]
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
    impact = impact.clone().detach().requires_grad_(True)
    original_param = dict(model.named_parameters())[param_name]

    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    param_grad = torch.autograd.grad(loss, original_param, create_graph=True)[0]
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