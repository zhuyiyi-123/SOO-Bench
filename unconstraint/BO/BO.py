import numpy as np
import torch
import tensorflow as tf
from botorch.models import FixedNoiseGP, ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.optim import optimize_acqf
torch.manual_seed(2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
dtype = torch.float32

def objective(args, input_x, input_shape, ensemble):
    original_x = input_x
    # convert the tensor into numpy before using a TF model
    # if torch.cuda.is_available():
    #     input_x = input_x.detach().cpu().numpy()
    # else:
    #     input_x = input_x.detach().numpy()
    input_x = input_x.detach().cpu().numpy()
    batch_shape = input_x.shape[:-1]
    # pass the input into a TF model
    input_x = tf.reshape(input_x, [-1, *input_shape])

    # optimize teh ground truth or the learned model
    value = ensemble.get_distribution(input_x).mean()
    ys = np.array(value)

    ys.reshape(list(batch_shape) + [1])
    # convert the scores back to pytorch tensors
    return torch.tensor(ys).type_as(
        original_x).to(device, dtype=dtype)

def initialize_model(train_x, train_obj, train_yvar, state_dict=None):
    # define models for objective
    model_obj = FixedNoiseGP(train_x, train_obj, train_yvar.expand_as(train_obj)).to(train_x)
    # combine into a multi-output GP model
    model = ModelListGP(model_obj)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    # load state dict if it is passed
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model

def obj_callable(Z):
        return Z[..., 0]
    
def optimize_acqf_and_get_observation(args, acq_func, bounds, BATCH_SIZE, NOISE_SE, input_shape, ensemble):
        """Optimizes the acquisition function, and returns
        a new candidate and a noisy observation."""
        # optimize
        try:
            candidates, _ = optimize_acqf(
                acq_function=acq_func,
                bounds=bounds,
                q=BATCH_SIZE,
                num_restarts=args.bo_num_restarts,
                raw_samples=args.bo_raw_samples,  # used for intialization heuristic
                options={"batch_limit": args.bo_batch_limit,
                         "maxiter": args.bo_maxiter})
        except RuntimeError:
            return
        # observe new values
        new_x = candidates.detach()
        exact_obj = objective(args, candidates, input_shape, ensemble)
        new_obj = exact_obj + NOISE_SE * torch.randn_like(exact_obj)
        return new_x, new_obj