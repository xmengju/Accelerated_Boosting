#from loss_grad import *
from __init__ import * 

_MACHINE_EPSILON = np.finfo(np.float64).eps

def exp_loss (pred, y, alpha, params, indices=None, pred_weak=None, compute_fval = True, compute_grad = True):

    edge = -pred*y
    if compute_fval:
        value = np.max(edge) + np.log(np.sum(np.exp(edge - np.max(edge) )))
        #value = np.exp(value)
    else: value = None

    if compute_grad:
        n_iter = pred_weak.shape[1]
        gradient = np.zeros((n_iter,1))
        expedge = np.exp(edge - np.log(len(edge)))


        for k in indices:
            if(params['l2']  == 0):

                gradient[k] = np.sum(expedge * (-y*pred_weak[:,k]))
                #tmp = np.sign(-y*pred_weak[:,k])* np.log( np.abs(-y*pred_weak[:,k])) + edge
                #print np.max(np.exp(tmp - np.max(tmp))), np.max(np.log(np.sum(np.exp(tmp - np.max(tmp)))))
                #print "value", np.max(tmp) + np.log(np.sum(np.exp(tmp - np.max(tmp))))
                #gradient[k] = np.max(tmp) + np.log(np.sum(np.exp(tmp - np.max(tmp))))
            else:
                gradient[k] = np.sum(expedge * (-y*pred_weak[:,k]))+ 2*params['l2'] *alpha[k]
                #gradient[k] = np.exp(edge + np.log(np.sum((-y*pred_weak[:,k])))) + 2*params['l2'] *alpha[k]

    else: gradient = None
    return value, gradient

def logit_loss (pred, y, alpha, params, indices=None, pred_weak=None, compute_fval = True, compute_grad = True):

    Y_train01 = (y+1)/2.

    if compute_fval:
        value = -np.dot(Y_train01,np.log(np.maximum(expit(pred),_MACHINE_EPSILON*2))) - np.dot((1-Y_train01),np.log(np.maximum(1-expit(pred),2*_MACHINE_EPSILON)))
        value = value/len(y)
    else: value = None

    if compute_grad:

        gradient = np.zeros((params['n_iter'],1))

        for k in indices:
            if(params['l2']  == 0):
                gradient[k] = np.mean((expit(pred) - Y_train01) * pred_weak[:,k])
            else:
                gradient[k] = np.mean((expit(pred) - Y_train01) * pred_weak[:,k]) + 2*params['l2'] *alpha[k]

    else: gradient = None
    return value, gradient



def LAD_loss(pred, y, alpha, params, indices=None, pred_weak=None, compute_fval = True, compute_grad = True):
    if compute_fval:
        value = np.mean(np.abs(pred - y))
    else: value = None

    if compute_grad:
        n_iter = pred_weak.shape[1]
        gradient = np.zeros((params['n_iter'] ,1))
        for k in indices:
            if(params['l2']  == 0):
                gradient[k] = -np.mean(np.sign(y - pred)* (pred_weak[:,k]))
            else:
                gradient[k] = -np.mean(np.sign(y - pred)* (pred_weak[:,k])) + len(y)*params['l2'] *2*alpha[k]

    else: gradient = None
    return value, gradient

def L2_loss(pred, y, alpha, params, indices=None, pred_weak=None, compute_fval = True, compute_grad = True):
    if compute_fval:
        value = np.mean(np.square(pred - y)/2)
    else: value = None

    if compute_grad:
        n_iter = pred_weak.shape[1]
        gradient = np.zeros((params['n_iter'] ,1))
        for k in indices:
            if(params['l2']  == 0):
                gradient[k] = -np.mean((y - pred)* pred_weak[:,k])
            else:
                gradient[k] = -np.mean((y - pred)* pred_weak[:,k]) + len(y)*params['l2'] *2*alpha[k]

    else: gradient = None
    return value, gradient


# for logitboost
def weights_and_response(y, prob):
    """Update the working weights and response for a boosting iteration."""
    # Samples with very certain probabilities (close to 0 or 1) are weighted
    # less than samples with probabilities closer to 1/2. This is to
    # encourage the higher-uncertainty samples to contribute more during
    # model training
    sample_weight = prob * (1. - prob)

    # Don't allow sample weights to be too close to zero for numerical
    # stability (cf. p. 353 in Friedman, Hastie, & Tibshirani (2000)).
    sample_weight = np.maximum(sample_weight, 2. * _MACHINE_EPSILON)

    # Compute the regression response z = (y - p) / (p * (1 - p))
    #z = ((y - prob) / (prob* (1. - prob)))
    z = ((y - prob) / (sample_weight)) 
    
    z = np.maximum(np.minimum(z,MAXZ),-MAXZ)

    # Very negative and very positive values of z are clipped for numerical
    # stability (cf. p. 352 in Friedman, Hastie, & Tibshirani (2000)).
    #z = np.clip(z, a_min=-self.max_response, a_max=self.max_response)
    return sample_weight, z

def get_best_lastalpha_exp(d,pred_train, pred_train_weak_i, Y_train, type = "vanilla"):
    # indicator of misclassified examples
    miss = [int(x) for x in pred_train_weak_i != Y_train]


    # weighted training error
    err_m = np.dot(d,miss) / sum(d)
        
    if float(err_m) < 1.e-10:
        return 0.
    elif type == "vanilla":
        return  0.5 * np.log( (1 - err_m) / float(err_m))
    else:
        return np.log ( (-1. + np.sqrt(1.+ 4.*(1. - err_m)*err_m))/(2.*err_m))



""" HELPER FUNCTION: GET ERROR RATE ========================================="""
def get_error_rate(pred, Y, params):
    if(params['l'] == 'logit' or params['l'] == 'exp' ):
        return sum(np.sign(pred) != np.sign(Y)) / float(len(Y))
    if(params['l'] == 'lad' or params['l'] == 'l2'):
        return(np.mean(np.abs(pred - Y)))

""" HELPER FUNCTION: PRINT ERROR RATE ======================================="""
def print_error_rate(err):
    print ('Error rate: Training: %.4f - Test: %.4f' % err)

""" HELPER FUNCTION: GENERIC CLASSIFIER ====================================="""
def generic_weak_learner(Y_train, X_train, Y_test, X_test, weak_l):
    weak_l.fit(X_train,Y_train)
    pred_train = weak_l.predict(X_train)
    pred_test = weak_l.predict(X_test)
    return get_error_rate(pred_train, Y_train), \
           get_error_rate(pred_test, Y_test)

""" PLOT FUNCTION ==========================================================="""
def plot_error_rate(config, params, err, fname, quantity, xgboost_error, learning_rate_XG):
    T = err.shape[1]
    num_variants = err.shape[0]     
    print (T, num_variants)
    cols = ['k', 'g', 'r', 'cyan', 'b', 'magenta', 'gray', 'orange'] 

    plt.figure()
    for i in range( num_variants ):

        params = parse_params(config[i], params)

        cont = 0
        simple = 1
        if (params["type_updated_coordinates"]['total'] > 0):
            label_suffix = '(T'
            simple = 0
        else:
            if (params["type_updated_coordinates"]['random'] > 0):
                label_suffix = '(' + str(params["type_updated_coordinates"]['random']) + 'R'
                cont = 1
                simple = 0
            if (params["type_updated_coordinates"]['greedy'] > 0):
                simple = 0
                if cont == 1:
                    label_suffix = label_suffix + ',' + str(params["type_updated_coordinates"]['greedy']) + 'G'
                else:
                    label_suffix = '(' + str(params["type_updated_coordinates"]['greedy']) + 'G'
        
        if (simple == 1):
            label_prefix = 'Simple'
            label_suffix = ''
        else:
            label_suffix = label_suffix + ')'
            if (params["is_acceleration"] == 0):
                label_prefix = 'GD-'
            elif (params["is_acceleration"] == 1):
                label_prefix = 'Nesterov-'
            else:
                label_prefix = 'Polyak-'

            if (params["is_line_search"] == 0):
                label_prefix = label_prefix + 'C'
            else:
                label_prefix = label_prefix + 'LS'

        plt.plot( range(1,T+1), err[i,:], color = cols[i], label = label_prefix + label_suffix)
    
    
    col_xg = ['pink','orange','yellow','brown']
    for k in range(len(xgboost_error)):
        plt.plot( range(1,T+1), xgboost_error[k], color = col_xg[k], label = 'XGBoost' + str(learning_rate_XG[k]))

    plt.xlabel('Number of iterations', fontsize = 12)
    plt.ylabel(quantity, fontsize = 12)
    plt.title(quantity + ' vs Number of iterations', fontsize = 16)

    plt.grid(True)
    plt.legend(loc = "upper right")
    plt.show()
    plt.savefig(fname)

""" Refresh pred matrices after each alpha update ============================"""
def refresh_pred(pred, alpha, pred_weak, indices = None, old_alpha = None, compact=False):
    if indices is None:
        pred = np.matmul(pred_weak,alpha).squeeze()
    else:
        if not compact: 
            alpha = alpha[indices]
            old_alpha = old_alpha[indices]
        pred = pred + np.matmul(pred_weak[:,indices],alpha-old_alpha).squeeze()
    return pred

""" Get which indices to update =========================================="""
def get_indices(params, i, pred_train, pred_train_weak, Y_train):

    sel = params["type_updated_coordinates"]
   

    indices = [i] # added the most recent coordinate to the list of coordinates to be updated 

    if (sel['total'] > 0) and (i > 0):
        indices.extend( range(i) )

    if (sel['random'] >0) and (i > 0):
        # random selection of the other coordinate 
        for coordinate in range(sel['random']): 
            indices.append( np.random.randint( low=0, high=i, size=1 )[0] )

    if (sel['gmin'] > 0 or sel['gmax'] > 0 or sel['ghmin'] > 0 ) and (i > 0):
        #pick the coordinates according to their gradient norms - Gauss Southwell
        # compute the gradient wrt to all coordinates until i
        grad = np.zeros((i))
        grad_norm = np.zeros((i))
        grad_hmin = np.zeros((i))


        for k in range(i):
            if(params['l'] == 'exp'):
                grad[k] = np.mean(np.exp(-Y_train * pred_train) * (-Y_train  * pred_train_weak[:,k])) 
                grad_hmin[k] = grad[k]  + (1/2.)*np.mean(np.exp(-Y_train * pred_train) * np.square(-Y_train  * pred_train_weak[:,k])) 
                grad_norm[k] = np.abs(grad[k]) 

            if(params['l'] == 'logit'):
                Y_train01 = (Y_train - 1)/2.
                grad[k] =  np.mean((expit(pred_train) - Y_train01) * pred_train_weak[:,k]) 
                grad_hmin[k]  = grad[k] + (1/2.)*np.mean( expit(pred_train) * (1.-expit(pred_train))* pred_train_weak[:,k]*pred_train_weak[:,k])
                grad_norm[k] = np.abs(grad[k]) 

            if(params['l'] == 'lad'):
                grad[k] = np.mean(np.sign((Y_train - pred_train) * pred_train_weak[:,k])) 
                grad_norm[k] = np.abs(grad[k]) 

            if(params['l'] == 'l2'):
                grad[k] = np.mean((Y_train - pred_train) * pred_train_weak[:,k]) 
                grad_norm[k] = np.abs(grad[k]) 

        indices.extend(grad.argsort()[:sel['gmin']])
        indices.extend(grad_hmin.argsort()[:sel['ghmin']])
        if(sel['gmax']> 0):
            indices.extend(grad_norm.argsort()[-sel['gmax']:])

        #rint(np.vstack([grad_hmin - grad, grad]).T)

        #track_indices[i,:] = [i, idx, grad_norms[idx]]

    indices = list(set(indices))

    return indices


""" Update alpha using GD ==================================================="""

