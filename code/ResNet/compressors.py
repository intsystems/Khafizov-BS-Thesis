import torch
from descent import gradient_descent, mirror_descent

class TopK:
    """
    A class used to compress gradients by selecting the top-k values.
    Attributes
    ----------
    k : int
        The number of top values to select.
    Methods
    -------
    compress(name, param)
        Compresses the gradient of the given parameter by selecting the top-k values.
    """
    def __init__(self, k):
        """
        Initializes the compressor with a given parameter.
        Args:
            k (int): The parameter to initialize the compressor with.
        """
        self.k = k
    
    def update(self, *args, **kwargs):
        """
        Placeholder for the update method.
        """
        pass

    def compress(self, name, param):
        """
        Compresses the gradient tensor by retaining only the top-k absolute values.
        Args:
            name (str): The name of the parameter (not used in the current implementation).
            param (torch.nn.Parameter): The parameter whose gradient tensor is to be compressed.
        Returns:
            torch.Tensor: The compressed gradient tensor with only the top-k absolute values retained.
        """
        k = int(self.k * param.numel())
        tensor = param.grad.view(-1)  # Flatten the tensor to a vector
        topk_values, topk_indices = tensor.abs().topk(k)
        mask = torch.zeros_like(tensor, dtype=torch.bool)
        mask.scatter_(0, topk_indices, True)
        compressed_tensor = tensor * mask
        compressed_tensor = compressed_tensor.view(param.grad.size())  # Reshape back to original size
        return compressed_tensor


class RandK:
    """
    A class used to represent a Random K Compressor.

    Attributes
    ----------
    k : float
        The fraction of elements to keep in the tensor during compression.

    Methods
    -------
    compress(name, param)
        Compresses the gradient of the given parameter by randomly keeping a fraction of elements.
    """
    def __init__(self, k):
        """
        Initializes the compressor with a given parameter.

        Args:
            k (int): The parameter to initialize the compressor.
        """
        self.k = k
    
    def update(self, *args, **kwargs):
        """
        Placeholder for the update method.
        """
        pass

    def compress(self, name, param):
        """
        Compresses the gradient tensor by randomly masking elements.

        Args:
            name (str): The name of the parameter (not used in the current implementation).
            param (torch.nn.Parameter): The parameter whose gradient tensor will be compressed.

        Returns:
            torch.Tensor: The compressed gradient tensor with a fraction of elements randomly masked.
        """
        k = int(self.k * param.numel())
        tensor = param.grad
        mask = torch.randperm(tensor.numel()) < k
        mask = mask.view(tensor.size())
        compressed_tensor = tensor * mask
        return compressed_tensor


class ImpK_b:
    """
    A class used to perform importance-based compression on model parameters.
    Attributes
    ----------
    model : torch.nn.Module
        The neural network model whose parameters are to be compressed.
    k : float
        The fraction of parameters to retain after compression.
    w : dict
        A dictionary containing the importance weights for each parameter in the model.
    mode : int, optional
        The mode of compression (default is 0).
    Methods
    -------
    update(X_train, y_train, criterion, lr, eta, num_steps)
        Updates the importance weights using mirror descent.
    compress(name, param)
        Compresses the given parameter tensor based on the importance weights.
    """
    def __init__(self, model, k, weighted=True):
        """
        Initializes the compressor with the given model, compression factor, and mode.

        Args:
            model (torch.nn.Module): The neural network model to be compressed.
            k (int): The compression factor.
            mode (int, optional): The mode of compression. Defaults to 0.
        """
        self.model = model
        self.k = k
        self.w = {name: (imp := torch.ones_like(param)) / imp.sum()
            for name, param in model.named_parameters()
        }
        self.weighted = weighted

    def update(self, X_train, y_train, criterion, lr, eta, num_steps):
        """
        Update the model parameters using mirror descent optimization on a simplex.

        Parameters:
        -----------
        X_train : torch.Tensor
            The input training data.
        y_train : torch.Tensor
            The target training labels.
        criterion : torch.nn.Module
            The loss function used to evaluate the model.
        lr : float
            The learning rate for the optimization.
        eta : float
            The step size parameter for mirror descent.
        num_steps : int
            The number of steps to perform in the mirror descent optimization.

        Returns:
        --------
        None
        """
        for name, param in self.model.named_parameters():
            self.w[name] = mirror_descent(
                model=self.model,
                param_name=name,
                impact=None,
                lr=lr,
                eta=eta,
                lambda_value=0.1,
                num_steps=num_steps,
                X_train=X_train,
                y_train=y_train,
                criterion=criterion
            )

    def compress(self, name, param):
        """
        Compresses the gradient tensor of a parameter based on the specified mode and weight.
        Args:
            name (str): The name of the parameter.
            param (torch.nn.Parameter): The parameter whose gradient tensor is to be compressed.
        Returns:
            torch.Tensor: The compressed gradient tensor.
        Notes:
            - If weighted is 0, the compression is based on the top-k elements of the weight tensor.
            - If weighted is 1, the compression is based on the top-k elements of the element-wise product of the gradient tensor and the weight tensor.
        """
        k = int(self.k * param.numel())
        if self.weighted:
            tensor = param.grad * self.w[name] * (param.numel() / self.w[name].sum())
            impk_indices = torch.argsort(tensor.abs().flatten(), descending=True)[:k]
        else:
            tensor = param.grad
            impk_indices = torch.argsort(self.w[name].flatten(), descending=True)[:k]

        
        mask = torch.zeros_like(tensor.flatten(), dtype=torch.bool)
        mask[impk_indices] = True
        mask = mask.view(tensor.size())
        
        # Apply mask to tensor
        compressed_tensor = tensor * mask
        return compressed_tensor

class ImpK_c:
    """
    A class used to perform importance-based compression on model parameters.
    Attributes
    ----------
    model : torch.nn.Module
        The neural network model whose parameters are to be compressed.
    k : float
        The fraction of parameters to retain after compression.
    w : dict
        A dictionary containing the importance weights for each parameter in the model.
    mode : int, optional
        The mode of compression (default is 0).
    Methods
    -------
    update(X_train, y_train, criterion, lr, eta, num_steps)
        Updates the importance weights using gradient descent.
    compress(name, param)
        Compresses the given parameter tensor based on the importance weights.
    """
    def __init__(self, model, k, weighted=True):
        """
        Initializes the compressor with the given model, compression factor, and mode.

        Args:
            model (torch.nn.Module): The neural network model to be compressed.
            k (int): The compression factor.
            mode (int, optional): The mode of compression. Defaults to 0.
        """
        self.model = model
        self.k = k
        self.w = {name: (imp := torch.ones_like(param))
            for name, param in model.named_parameters()
        }
        self.weighted = weighted

    def update(self, X_train, y_train, criterion, lr, eta, num_steps):
        """
        Update the model parameters using gradient descent optimization on a cube.

        Parameters:
        -----------
        X_train : torch.Tensor
            The input training data.
        y_train : torch.Tensor
            The target training labels.
        criterion : torch.nn.Module
            The loss function used to evaluate the model.
        lr : float
            The learning rate for the optimization.
        eta : float
            The step size parameter for gradient descent.
        num_steps : int
            The number of steps to perform in the gradient descent optimization.

        Returns:
        --------
        None
        """
        for name, param in self.model.named_parameters():
            self.w[name] = gradient_descent(
                model=self.model,
                param_name=name,
                impact=None,
                lr=lr,
                eta=eta,
                num_steps=num_steps,
                X_train=X_train,
                y_train=y_train,
                criterion=criterion
            )

    def compress(self, name, param):
        """
        Compresses the gradient tensor of a parameter based on the specified mode and weight.
        Args:
            name (str): The name of the parameter.
            param (torch.nn.Parameter): The parameter whose gradient tensor is to be compressed.
        Returns:
            torch.Tensor: The compressed gradient tensor.
        Notes:
            - If weighted is 0, the compression is based on the top-k elements of the weight tensor.
            - If weighted is 1, the compression is based on the top-k elements of the element-wise product of the gradient tensor and the weight tensor.
        """
        k = int(self.k * param.numel())
        if self.weighted:
            tensor = param.grad * self.w[name]
            impk_indices = torch.argsort(tensor.abs().flatten(), descending=True)[:k]
        else:
            tensor = param.grad
            impk_indices = torch.argsort(self.w[name].flatten(), descending=True)[:k]
        
        mask = torch.zeros_like(tensor.flatten(), dtype=torch.bool)
        mask[impk_indices] = True
        mask = mask.view(tensor.size())
        
        # Apply mask to tensor
        compressed_tensor = tensor * mask
        return compressed_tensor