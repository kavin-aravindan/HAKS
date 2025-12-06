import torch
import math

class Ackley:
    """
    Ackley function, a common benchmark for minimization.
    The global minimum is f(x) = 0 at x = (0, ..., 0).
    """

    def __init__(self, dim=2, maximize=False):
        """
        Initializes the Ackley function.

        Args:
            dim (int): The number of dimensions.
            maximize (bool): If True, the function is negated for maximization.
        """
        self.name = "Ackley"
        self.dim = dim
        self.bounds = torch.tensor([[-32.768] * dim, [32.768] * dim])
        self.maximize = maximize

        # The known global minimum of the standard Ackley function.
        self.fmin = 0.0
        self.xmin = torch.zeros(dim)

        # The optimum for the configured problem (min or max).
        self.optimum_value = -self.fmin if self.maximize else self.fmin
        self.optimum_location = self.xmin

    def true_function(self, X):
        """
        Computes the Ackley function.

        Args:
            X (torch.Tensor): A (n x d) tensor of inputs.

        Returns:
            torch.Tensor: A (n x 1) tensor of function values.
                          Returns -f(X) if maximize=True.
        """
        a = 20.0
        b = 0.2
        c = 2 * torch.pi

        part1 = -a * torch.exp(-b * torch.sqrt(torch.mean(X**2, dim=1)))
        part2 = -torch.exp(torch.mean(torch.cos(c * X), dim=1))
        value = part1 + part2 + a + torch.exp(torch.tensor(1.0))

        return -value if self.maximize else value
    
    

class Branin:
    """
    Branin function, a 2D benchmark with three global minima.
    The global minimum value is approximately f(x) = 0.397887.
    """

    def __init__(self, maximize=False):
        """
        Initializes the Branin function.

        Args:
            maximize (bool): If True, the function is negated for maximization.
        """
        self.name = "Branin"
        self.dim = 2
        self.bounds = torch.tensor([[-5.0, 0.0], [10.0, 15.0]])
        self.maximize = maximize

        # The known global minimum of the standard Branin function.
        self.fmin = 0.397887
        self.xmin = torch.tensor([
            [-math.pi, 12.275],
            [math.pi, 2.275],
            [9.42478, 2.475]
        ])

        # The optimum for the configured problem (min or max).
        self.optimum_value = -self.fmin if self.maximize else self.fmin
        self.optimum_location = self.xmin


    def true_function(self, X):
        """
        Computes the Branin function.

        Args:
            X (torch.Tensor): A (n x 2) tensor of inputs.

        Returns:
            torch.Tensor: A (n x 1) tensor of function values.
                          Returns -f(X) if maximize=True.
        """
        a = 1.0
        b = 5.1 / (4 * math.pi**2)
        c = 5 / math.pi
        r = 6.0
        s = 10.0
        t = 1 / (8 * math.pi)

        x1 = X[:, 0]
        x2 = X[:, 1]

        term1 = a * (x2 - b * x1**2 + c * x1 - r)**2
        term2 = s * (1 - t) * torch.cos(x1)
        value = term1 + term2 + s

        return -value if self.maximize else value
    
    
import torch

class SineCosine:
    """
    A simple sine-cosine function.
    f(x) = sin(x * 2.5) * cos(x * 0.5)
    """

    def __init__(self, dim=1, maximize=False):
        """
        Initializes the SineCosine function.

        Args:
            dim (int): The number of dimensions. Must be 1 for this function.
            maximize (bool): If True, the function is negated for maximization.
        """
        self.name = "SineCosine"
        if dim != 1:
            raise ValueError("This example function is designed for 1 dimension.")
        self.dim = dim
        self.bounds = torch.tensor([[-5.0], [5.0]])
        self.maximize = maximize

        # The global minimum within the domain [-5, 5] is approx -0.99925
        self.fmin = -0.99925
        self.xmin = torch.tensor([[4.364]])

        # The optimum for the configured problem (min or max).
        self.optimum_value = -self.fmin if self.maximize else self.fmin
        self.optimum_location = self.xmin


    def true_function(self, X):
        """
        Computes the sine-cosine function.

        Args:
            X (torch.Tensor): A (n x d) tensor of inputs.

        Returns:
            torch.Tensor: A (n x 1) tensor of function values.
                          Returns -f(X) if maximize=True.
        """
        value = torch.sin(X * 2.5) * torch.cos(X * 0.5)
        
        return -value if self.maximize else value
    
class Eggholder:
    """
    Eggholder function, a challenging benchmark for global optimization.
    It is defined on a 2D square, usually [-512, 512] x [-512, 512].
    The global minimum is f(x) approx -959.6407 at (512, 404.2319). [1, 2, 7]
    """

    def __init__(self, dim=2, maximize=False):
        """
        Initializes the Eggholder function.

        Args:
            dim (int): The number of dimensions. Must be 2 for this function.
            maximize (bool): If True, the function is negated for maximization.
        """
        if dim != 2:
            raise ValueError("The Eggholder function is defined for 2 dimensions.")
        
        self.name = "Eggholder"
        self.dim = dim
        self.bounds = torch.tensor([[-512.0, -512.0], [512.0, 512.0]])
        self.maximize = maximize

        # The known global minimum of the standard Eggholder function.
        self.fmin = -959.6406627208505
        self.xmin = torch.tensor([[512.0, 404.2319]])

        # The optimum for the configured problem (min or max).
        self.optimum_value = -self.fmin if self.maximize else self.fmin
        self.optimum_location = self.xmin

    def true_function(self, X):
        """
        Computes the Eggholder function.

        Args:
            X (torch.Tensor): A (n x 2) tensor of inputs.

        Returns:
            torch.Tensor: A (n x 1) tensor of function values.
                          Returns -f(X) if maximize=True.
        """
        x1 = X[:, 0]
        x2 = X[:, 1]

        term1 = -(x2 + 47) * torch.sin(torch.sqrt(torch.abs(x2 + x1 / 2 + 47)))
        term2 = -x1 * torch.sin(torch.sqrt(torch.abs(x1 - (x2 + 47))))
        
        value = term1 + term2
        
        return -value if self.maximize else value
    
class Sphere:
    """
    Sphere function, a simple benchmark for minimization.
    The global minimum is f(x) = 0 at x = (0, ..., 0).
    """

    def __init__(self, dim=2, maximize=False):
        """
        Initializes the Sphere function.

        Args:
            dim (int): The number of dimensions.
            maximize (bool): If True, the function is negated for maximization.
        """
        self.name = "Sphere"
        self.dim = dim
        self.bounds = torch.tensor([[-5.12] * dim, [5.12] * dim])
        self.maximize = maximize

        # The known global minimum of the standard Sphere function.
        self.fmin = 0.0
        self.xmin = torch.zeros(dim)

        # The optimum for the configured problem (min or max).
        self.optimum_value = -self.fmin if self.maximize else self.fmin
        self.optimum_location = self.xmin

    def true_function(self, X):
        """
        Computes the Sphere function.

        Args:
            X (torch.Tensor): A (n x d) tensor of inputs.

        Returns:
            torch.Tensor: A (n x 1) tensor of function values.
                          Returns -f(X) if maximize=True.
        """
        value = torch.sum(X**2, dim=1)
        return -value if self.maximize else value

class Zakharov:
    """
    Zakharov function.
    The global minimum is f(x) = 0 at x = (0, ..., 0).
    """

    def __init__(self, dim=2, maximize=False):
        """
        Initializes the Zakharov function.

        Args:
            dim (int): The number of dimensions.
            maximize (bool): If True, the function is negated for maximization.
        """
        self.name = "Zakharov"
        self.dim = dim
        self.bounds = torch.tensor([[-5.0] * dim, [10.0] * dim])
        self.maximize = maximize

        # The known global minimum of the standard Zakharov function.
        self.fmin = 0.0
        self.xmin = torch.zeros(dim)

        # The optimum for the configured problem (min or max).
        self.optimum_value = -self.fmin if self.maximize else self.fmin
        self.optimum_location = self.xmin

    def true_function(self, X):
        """
        Computes the Zakharov function.

        Args:
            X (torch.Tensor): A (n x d) tensor of inputs.

        Returns:
            torch.Tensor: A (n x 1) tensor of function values.
                          Returns -f(X) if maximize=True.
        """
        term1 = torch.sum(X**2, dim=1)
        
        indices = torch.arange(1, self.dim + 1, device=X.device)
        sum_ix = torch.sum(0.5 * indices * X, dim=1)
        
        term2 = sum_ix**2
        term3 = sum_ix**4
        
        value = term1 + term2 + term3
        return -value if self.maximize else value

class DixonPrice:
    """
    Dixon-Price function.
    The global minimum is f(x) = 0.
    """

    def __init__(self, dim=2, maximize=False):
        """
        Initializes the Dixon-Price function.

        Args:
            dim (int): The number of dimensions.
            maximize (bool): If True, the function is negated for maximization.
        """
        self.name = "DixonPrice"
        self.dim = dim
        self.bounds = torch.tensor([[-10.0] * dim, [10.0] * dim])
        self.maximize = maximize

        # The known global minimum of the standard Dixon-Price function.
        self.fmin = 0.0
        
        # Calculate the location of the minimum
        xmin_vals = [2**(-( (2**i - 2) / (2**i) )) for i in range(1, dim + 1)]
        self.xmin = torch.tensor(xmin_vals)

        # The optimum for the configured problem (min or max).
        self.optimum_value = -self.fmin if self.maximize else self.fmin
        self.optimum_location = self.xmin

    def true_function(self, X):
        """
        Computes the Dixon-Price function.

        Args:
            X (torch.Tensor): A (n x d) tensor of inputs.

        Returns:
            torch.Tensor: A (n x 1) tensor of function values.
                          Returns -f(X) if maximize=True.
        """
        term1 = (X[:, 0] - 1)**2
        
        indices = torch.arange(2, self.dim + 1, device=X.device)
        sum_terms = indices * (2 * X[:, 1:]**2 - X[:, :-1])**2
        term2 = torch.sum(sum_terms, dim=1)
        
        value = term1 + term2
        return -value if self.maximize else value

class Levy:
    """
    Levy function.
    The global minimum is f(x) = 0 at x = (1, ..., 1).
    """

    def __init__(self, dim=2, maximize=False):
        """
        Initializes the Levy function.

        Args:
            dim (int): The number of dimensions.
            maximize (bool): If True, the function is negated for maximization.
        """
        self.name = "Levy"
        self.dim = dim
        self.bounds = torch.tensor([[-10.0] * dim, [10.0] * dim])
        self.maximize = maximize

        # The known global minimum of the standard Levy function.
        self.fmin = 0.0
        self.xmin = torch.ones(dim)

        # The optimum for the configured problem (min or max).
        self.optimum_value = -self.fmin if self.maximize else self.fmin
        self.optimum_location = self.xmin

    def true_function(self, X):
        """
        Computes the Levy function.

        Args:
            X (torch.Tensor): A (n x d) tensor of inputs.

        Returns:
            torch.Tensor: A (n x 1) tensor of function values.
                          Returns -f(X) if maximize=True.
        """
        w = 1 + (X - 1) / 4
        
        term1 = torch.sin(math.pi * w[:, 0])**2
        
        w_i = w[:, :-1]
        term2 = torch.sum((w_i - 1)**2 * (1 + 10 * torch.sin(math.pi * w_i + 1)**2), dim=1)
        
        w_d = w[:, -1]
        term3 = (w_d - 1)**2 * (1 + torch.sin(2 * math.pi * w_d)**2)
        
        value = term1 + term2 + term3
        return -value if self.maximize else value

class Rosenbrock:
    """
    Rosenbrock function.
    The global minimum is f(x) = 0 at x = (1, ..., 1).
    """

    def __init__(self, dim=2, maximize=False):
        """
        Initializes the Rosenbrock function.

        Args:
            dim (int): The number of dimensions.
            maximize (bool): If True, the function is negated for maximization.
        """
        self.name = "Rosenbrock"
        self.dim = dim
        self.bounds = torch.tensor([[-5.0] * dim, [10.0] * dim])
        self.maximize = maximize

        # The known global minimum of the standard Rosenbrock function.
        self.fmin = 0.0
        self.xmin = torch.ones(dim)

        # The optimum for the configured problem (min or max).
        self.optimum_value = -self.fmin if self.maximize else self.fmin
        self.optimum_location = self.xmin

    def true_function(self, X):
        """
        Computes the Rosenbrock function.

        Args:
            X (torch.Tensor): A (n x d) tensor of inputs.

        Returns:
            torch.Tensor: A (n x 1) tensor of function values.
                          Returns -f(X) if maximize=True.
        """
        sum_terms = 100 * (X[:, 1:] - X[:, :-1]**2)**2 + (X[:, :-1] - 1)**2
        value = torch.sum(sum_terms, dim=1)
        return -value if self.maximize else value

class Michalewicz:
    """
    Michalewicz function.
    The global minimum is not at a single point and its value depends on the dimension.
    """

    def __init__(self, dim=2, m=10, maximize=False):
        """
        Initializes the Michalewicz function.

        Args:
            dim (int): The number of dimensions.
            m (int): The steepness parameter.
            maximize (bool): If True, the function is negated for maximization.
        """
        self.name = "Michalewicz"
        self.dim = dim
        self.m = m
        self.bounds = torch.tensor([[0.0] * dim, [math.pi] * dim])
        self.maximize = maximize

        # Approximate global minimum values for common dimensions
        if dim == 2:
            self.fmin = -1.8013
            self.xmin = torch.tensor([2.20, 1.57])
        elif dim == 5:
            self.fmin = -4.687658
            self.xmin = None # Not easily available
        elif dim == 10:
            self.fmin = -9.66015
            self.xmin = None # Not easily available
        else:
            self.fmin = None # Not easily available for other dimensions
            self.xmin = None

        # The optimum for the configured problem (min or max).
        if self.fmin is not None:
            self.optimum_value = -self.fmin if self.maximize else self.fmin
        else:
            self.optimum_value = None
        self.optimum_location = self.xmin

    def true_function(self, X):
        """
        Computes the Michalewicz function.

        Args:
            X (torch.Tensor): A (n x d) tensor of inputs.

        Returns:
            torch.Tensor: A (n x 1) tensor of function values.
                          Returns -f(X) if maximize=True.
        """
        indices = torch.arange(1, self.dim + 1, device=X.device)
        term = torch.sin(X) * (torch.sin(indices * X**2 / math.pi))**(2 * self.m)
        value = -torch.sum(term, dim=1)
        
        return -value if self.maximize else value

class Linear:
    """
    Linear function: f(x) = sum(weights * x) + bias
    Simple linear function for testing kernel performance.
    """

    def __init__(self, dim=2, maximize=False):
        """
        Initializes the Linear function.

        Args:
            dim (int): The number of dimensions.
            maximize (bool): If True, the function is negated for maximization.
        """
        self.name = "Linear"
        self.dim = dim
        self.bounds = torch.tensor([[-10.0] * dim, [10.0] * dim])
        self.maximize = maximize

        # Set random weights and bias for reproducibility
        torch.manual_seed(42)
        self.weights = torch.randn(dim)
        self.bias = torch.randn(1)

        # The minimum/maximum depends on the bounds and weights
        if maximize:
            # For maximization, choose the corner that maximizes the linear function
            corner_values = []
            for corner in [[self.bounds[0]], [self.bounds[1]]]:
                corner_tensor = torch.tensor(corner).expand(1, dim)
                corner_values.append(self.true_function(corner_tensor).item())
            self.optimum_value = max(corner_values)
        else:
            # For minimization, choose the corner that minimizes the linear function
            corner_values = []
            for corner in [[self.bounds[0]], [self.bounds[1]]]:
                corner_tensor = torch.tensor(corner).expand(1, dim)
                corner_values.append(self.true_function(corner_tensor).item())
            self.optimum_value = min(corner_values)

        self.optimum_location = None  # Linear function optimum is at boundary

    def true_function(self, X):
        """
        Computes the Linear function.

        Args:
            X (torch.Tensor): A (n x d) tensor of inputs.

        Returns:
            torch.Tensor: A (n x 1) tensor of function values.
                          Returns -f(X) if maximize=True.
        """
        value = torch.sum(self.weights * X, dim=1) + self.bias
        return -value if self.maximize else value

class Periodic:
    """
    Periodic function: f(x) = sum(sin(2*pi*x_i))
    Simple periodic function for testing kernel performance.
    """

    def __init__(self, dim=2, maximize=False):
        """
        Initializes the Periodic function.

        Args:
            dim (int): The number of dimensions.
            maximize (bool): If True, the function is negated for maximization.
        """
        self.name = "Periodic"
        self.dim = dim
        self.bounds = torch.tensor([[0.0] * dim, [2.0] * dim])
        self.maximize = maximize

        # The minimum is -dim at x = (0.25, 0.25, ..., 0.25) (where sin(2*pi*0.25) = -1)
        self.fmin = -dim
        self.xmin = torch.full((dim,), 0.75)  # sin(2*pi*0.75) = -1

        # The optimum for the configured problem (min or max).
        self.optimum_value = -self.fmin if self.maximize else self.fmin
        self.optimum_location = self.xmin

    def true_function(self, X):
        """
        Computes the Periodic function.

        Args:
            X (torch.Tensor): A (n x d) tensor of inputs.

        Returns:
            torch.Tensor: A (n x 1) tensor of function values.
                          Returns -f(X) if maximize=True.
        """
        value = torch.sum(torch.sin(2 * math.pi * X), dim=1)
        return -value if self.maximize else value

class Hartman3D:
    """
    3D Hartman function.
    The global minimum is f(x) = -3.86278 at x = (0.114614, 0.555649, 0.852547).
    """

    def __init__(self, maximize=False):
        """
        Initializes the 3D Hartman function.

        Args:
            maximize (bool): If True, the function is negated for maximization.
        """
        self.name = "Hartman3D"
        self.dim = 3
        self.bounds = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        self.maximize = maximize

        # Hartman 3D parameters
        self.A = torch.tensor([
            [3.0, 10.0, 30.0],
            [0.1, 10.0, 35.0],
            [3.0, 10.0, 30.0],
            [0.1, 10.0, 35.0]
        ])
        
        self.P = torch.tensor([
            [0.3689, 0.1170, 0.2673],
            [0.4699, 0.4387, 0.7470],
            [0.1091, 0.8732, 0.5547],
            [0.03815, 0.5743, 0.8828]
        ])
        
        self.alpha = torch.tensor([1.0, 1.2, 3.0, 3.2])

        # The known global minimum of the 3D Hartman function.
        self.fmin = -3.86278
        self.xmin = torch.tensor([0.114614, 0.555649, 0.852547])

        # The optimum for the configured problem (min or max).
        self.optimum_value = -self.fmin if self.maximize else self.fmin
        self.optimum_location = self.xmin

    def true_function(self, X):
        """
        Computes the 3D Hartman function.

        Args:
            X (torch.Tensor): A (n x 3) tensor of inputs.

        Returns:
            torch.Tensor: A (n x 1) tensor of function values.
                          Returns -f(X) if maximize=True.
        """
        outer_sum = torch.zeros(X.shape[0])
        
        for i in range(4):
            inner_sum = torch.sum(self.A[i] * (X - self.P[i])**2, dim=1)
            outer_sum += self.alpha[i] * torch.exp(-inner_sum)
        
        value = -outer_sum
        return -value if self.maximize else value

class Hartman6D:
    """
    6D Hartman function.
    The global minimum is f(x) = -3.32237 at x = (0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573).
    """

    def __init__(self, maximize=False):
        """
        Initializes the 6D Hartman function.

        Args:
            maximize (bool): If True, the function is negated for maximization.
        """
        self.name = "Hartman6D"
        self.dim = 6
        self.bounds = torch.tensor([[0.0] * 6, [1.0] * 6])
        self.maximize = maximize

        # Hartman 6D parameters
        self.A = torch.tensor([
            [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
            [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
            [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
            [17.0, 8.0, 0.05, 10.0, 0.1, 14.0]
        ])
        
        self.P = torch.tensor([
            [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
            [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
            [0.2348, 0.1451, 0.3522, 0.2883, 0.3047, 0.6650],
            [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381]
        ])
        
        self.alpha = torch.tensor([1.0, 1.2, 3.0, 3.2])

        # The known global minimum of the 6D Hartman function.
        self.fmin = -3.32237
        self.xmin = torch.tensor([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])

        # The optimum for the configured problem (min or max).
        self.optimum_value = -self.fmin if self.maximize else self.fmin
        self.optimum_location = self.xmin

    def true_function(self, X):
        """
        Computes the 6D Hartman function.

        Args:
            X (torch.Tensor): A (n x 6) tensor of inputs.

        Returns:
            torch.Tensor: A (n x 1) tensor of function values.
                          Returns -f(X) if maximize=True.
        """
        outer_sum = torch.zeros(X.shape[0])
        
        for i in range(4):
            inner_sum = torch.sum(self.A[i] * (X - self.P[i])**2, dim=1)
            outer_sum += self.alpha[i] * torch.exp(-inner_sum)
        
        value = -outer_sum
        return -value if self.maximize else value

class SumSquares:
    """
    Sum of squares function.
    The global minimum is f(x) = 0 at x = (0, 0, ..., 0).
    """

    def __init__(self, dim=2, maximize=False):
        """
        Initializes the sum of squares function.

        Args:
            dim (int): The dimensionality of the input space.
            maximize (bool): If True, the function is negated for maximization.
        """
        self.name = "SumSquares"
        self.dim = dim
        self.bounds = torch.tensor([[-5.0] * dim, [5.0] * dim])
        self.maximize = maximize

        # The known global minimum of the sum of squares function.
        self.fmin = 0.0
        self.xmin = torch.zeros(dim)

        # The optimum for the configured problem (min or max).
        self.optimum_value = -self.fmin if self.maximize else self.fmin
        self.optimum_location = self.xmin

    def true_function(self, X):
        """
        Computes the sum of squares function.

        Args:
            X (torch.Tensor): A (n x dim) tensor of inputs.

        Returns:
            torch.Tensor: A (n x 1) tensor of function values.
                          Returns -f(X) if maximize=True.
        """
        value = torch.sum(X**2, dim=1)
        return -value if self.maximize else value

class Conditional2D:
    """
    A 2D conditional function defined as:
    f(x, y) = -exp(-(x+y)/3) if x + y <= 0
    f(x, y) = -cos(10(x+y))/(1+x+y) otherwise
    """
    
    def __init__(self, maximize=False):
        """
        Initializes the Conditional2D function.

        Args:
            maximize (bool): If True, the function is negated for maximization.
        """
        self.name = "Conditional2D"
        self.dim = 2

        self.bounds = torch.tensor([[-5.0, -5.0], [5.0, 5.0]])
        self.maximize = maximize

        self.fmin = -torch.exp(-(-5.0 - 5.0) / 3)
        self.xmin = [-5.0, -5.0]

        # The optimum for the configured problem (min or max).
        self.optimum_value = self.fmin
        self.optimum_location = self.xmin

    def true_function(self, X):
        """
        Computes the Conditional2D function.

        Args:
            X (torch.Tensor): A (n x 2) tensor of inputs.

        Returns:
            torch.Tensor: A (n x 1) tensor of function values.
                          Returns -f(X) if maximize=True.
        """
        # Sum the two input dimensions
        sum_xy = X[:, 0] + X[:, 1]

        # Define the two parts of the conditional function
        value_if_leq_zero = -torch.exp(-sum_xy / 3)
        value_if_gt_zero = -torch.cos(10 * sum_xy) / (1 + sum_xy)

        # Use torch.where for a vectorized conditional operation
        value = torch.where(sum_xy <= 0, value_if_leq_zero, value_if_gt_zero)

        return -value if self.maximize else value

import torch

class Griewank:
    """
    Griewank function, a common benchmark for minimization.
    The global minimum is f(x) = 0 at x = (0, ..., 0).
    """

    def __init__(self, dim=2, maximize=False):
        """
        Initializes the Griewank function.

        Args:
            dim (int): The number of dimensions.
            maximize (bool): If True, the function is negated for maximization.
        """
        self.name = "Griewank"
        self.dim = dim
        self.bounds = torch.tensor([[-600.0] * dim, [600.0] * dim])
        self.maximize = maximize

        # The known global minimum of the standard Griewank function.
        self.fmin = 0.0
        self.xmin = torch.zeros(dim)

        # The optimum for the configured problem (min or max).
        self.optimum_value = -self.fmin if self.maximize else self.fmin
        self.optimum_location = self.xmin

    def true_function(self, X):
        """
        Computes the Griewank function.

        Args:
            X (torch.Tensor): A (n x d) tensor of inputs.

        Returns:
            torch.Tensor: A (n x 1) tensor of function values.
                          Returns -f(X) if maximize=True.
        """
        # Sum component: sum(x^2 / 4000)
        sum_part = torch.sum(X**2 / 4000.0, dim=1)

        # Product component: product(cos(x / sqrt(i))) for i=1..d
        # Create indices [1, 2, ..., d]
        indices = torch.arange(1, self.dim + 1, dtype=X.dtype, device=X.device)
        prod_part = torch.prod(torch.cos(X / torch.sqrt(indices)), dim=1)

        value = sum_part - prod_part + 1.0

        return -value if self.maximize else value


class Rastrigin:
    """
    Rastrigin function, a common benchmark for minimization.
    The global minimum is f(x) = 0 at x = (0, ..., 0).
    """

    def __init__(self, dim=2, maximize=False):
        """
        Initializes the Rastrigin function.

        Args:
            dim (int): The number of dimensions.
            maximize (bool): If True, the function is negated for maximization.
        """
        self.name = "Rastrigin"
        self.dim = dim
        self.bounds = torch.tensor([[-5.12] * dim, [5.12] * dim])
        self.maximize = maximize

        # The known global minimum of the standard Rastrigin function.
        self.fmin = 0.0
        self.xmin = torch.zeros(dim)

        # The optimum for the configured problem (min or max).
        self.optimum_value = -self.fmin if self.maximize else self.fmin
        self.optimum_location = self.xmin

    def true_function(self, X):
        """
        Computes the Rastrigin function.

        Args:
            X (torch.Tensor): A (n x d) tensor of inputs.

        Returns:
            torch.Tensor: A (n x 1) tensor of function values.
                          Returns -f(X) if maximize=True.
        """
        # Formula: 10d + sum(x^2 - 10cos(2*pi*x))
        term1 = 10.0 * self.dim
        term2 = torch.sum(X**2 - 10.0 * torch.cos(2.0 * torch.pi * X), dim=1)

        value = term1 + term2

        return -value if self.maximize else value
