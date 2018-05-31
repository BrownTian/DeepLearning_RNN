from autograd import value_and_grad
from autograd import grad as compute_grad  
import autograd.numpy as np
from autograd.misc.flatten import flatten
from autograd.misc.flatten import flatten_func

# gradient descent function
def gradient_descent(g,w_unflat,alpha_choice,max_its,version,**kwargs):
    verbose = False
    if 'verbose' in kwargs:
        verbose = kwargs['verbose']

    # flatten the input function, create gradient based on flat function
    g_flat, unflatten, w = flatten_func(g, w_unflat)
    grad = compute_grad(g)

    # record history
    w_hist = []
    w_hist.append(w_unflat)

    # over the line
    for k in range(max_its):   
        if verbose == True:
            if np.mod(k,5) == 0:
                print ('started iteration ' + str(k) + ' of ' + str(max_its))
                
                
         # check if diminishing steplength rule used
        if alpha_choice == 'diminishing':
            alpha = 1/float(k)
        else:
            alpha = alpha_choice
            
                
        # plug in value into func and derivative
        grad_eval = grad(w_unflat)
        grad_eval, _ = flatten(grad_eval)

        ### normalized or unnormalized descent step? ###
        if version == 'normalized':
            grad_norm = np.linalg.norm(grad_eval)
            if grad_norm == 0:
                grad_norm += 10**-6*np.sign(2*np.random.rand(1) - 1)
            grad_eval /= grad_norm

        # take descent step 
        w = w - alpha*grad_eval

        # record weight update
        w_unflat = unflatten(w)
        w_hist.append(w_unflat)
        
    if verbose == True:
        print ('finished all ' + str(max_its) + ' iterations')

    return w_hist