from __init__ import * 
from utils import *


                

def boost(Y_train, X_train, Y_test, X_test, params, weak_l):
    
    def get_best_last_alpha():

        if(params['l'] == 'exp'):
            a = get_best_lastalpha_exp(d,pred_train, pred_train_weak[:,i], Y_train)
        if(params['l'] == 'logit'):
            a = 1.
        if(params['l'] == 'lad'):
            def f(x):  return (np.mean(np.abs(Y_train - pred_train- x*pred_train_weak[:,i])))
            a = optimize.minimize_scalar(f).x

        if(params['l'] == 'l2'):
            def f(x): return (np.mean(np.square(Y_train - pred_train- x*pred_train_weak[:,i])))
            a = optimize.minimize_scalar(f).x
        return a 
        
    def compute_loss_grad(pred_train,alpha):
    
        if(params['l'] == 'exp'):
            loss, grad = exp_loss(pred_train, Y_train, alpha, params, indices, pred_train_weak,compute_fval = True, compute_grad = True)
        if(params['l'] == 'logit'):
            loss, grad = logit_loss(pred_train, Y_train, alpha, params, indices, pred_train_weak, compute_fval = True, compute_grad = True)
        if(params['l'] == 'lad'):
            loss, grad = LAD_loss(pred_train, Y_train, alpha, params, indices, pred_train_weak, compute_fval = True, compute_grad = True)
        if(params['l'] == 'l2'):
            loss, grad = L2_loss(pred_train, Y_train, alpha, params, indices, pred_train_weak, compute_fval = True, compute_grad = True)

        return loss, grad
    
    def compute_loss(pred_train,alpha):
    
        if(params['l'] == 'exp'):
            loss, grad = exp_loss(pred_train, Y_train, alpha, params,compute_fval = True, compute_grad = False)
        if(params['l'] == 'logit'):
            loss, grad = logit_loss(pred_train, Y_train, alpha, params, compute_fval = True, compute_grad = False)
        if(params['l'] == 'lad'):
            loss, grad = LAD_loss(pred_train, Y_train, alpha, params, compute_fval = True, compute_grad = False)
        if(params['l'] == 'l2'):
            loss, grad = L2_loss(pred_train, Y_train, alpha, params, compute_fval = True, compute_grad = False)

        return loss  
    
    def linesearch(step_size_orig, linesearch_params,grad_temp, loss_old):

        sigma,  beta = linesearch_params['sigma'],linesearch_params['beta']
        # reset step-size 
        step_size = step_size_orig

        
        
        alpha_temp = alpha -  grad_temp 
        # refresh pred_train
        pred_train_temp = refresh_pred(pred_train, alpha_temp, pred_train_weak, indices, alpha)
                    
                    
        # to be set to 1 if we find a permissible step-size
        found = False

        for i in range(linesearch_params['n_iter'] ):
            loss_temp = compute_loss(pred_train_temp, alpha_temp)
            

            # checking Armijo-condition
            if (params['l'] == 'exp'):
                diff = 1 - np.exp(loss_temp-loss_old)
                scale = 1./np.exp(loss_temp)

            else: 
                diff = loss_old - loss_temp
                scale = 1.


            if ( diff > scale*sigma * step_size * (np.linalg.norm(grad)**2) ):

                # if satisfied, use this as the next iterate
                found = True
                break

            else:
                # if not, decrease the step-size
                step_size = step_size * beta
                #print(step_size, loss_old, loss_temp, np.linalg.norm(grad))
                # calculate new possible iterate
                alpha_temp = alpha - step_size * grad_temp

                # refresth pred_train
                pred_train_temp = refresh_pred(pred_train, alpha_temp, pred_train_weak, indices, alpha)

                # compute loss at this new prospective iterate

        if found: return step_size
        else: return 1.e-10
                 
        
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    
    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]

    err_train = []; err_test = []; loss_train = []; loss_test = []

    alpha = np.zeros((params["n_iter"],1))

    pred_train_weak = np.zeros((n_train, params["n_iter"]))
    pred_test_weak = np.zeros((n_test, params["n_iter"]))

    # Adaboost 
    if(params['l'] == 'exp'):
        d = np.ones(n_train)/(n_train+0.)

    # logitboost
    if(params['l'] == 'logit'):
        p = np.full(shape=n_train, fill_value=0.5, dtype=np.float64)
        Y_train01 = (Y_train+1)/2.

    # Nesterov acceleration
    if (params['is_acceleration'] == 'polyak'):
        alpha_old = alpha + 0.
    elif (params['is_acceleration'] == 'nesterov'):
        alpha_z = np.copy(alpha)
        alpha_yold = np.copy(alpha)
        theta_old = 1
        step_size_org = params['step_size']
        if params['is_line_search']:
            t_seq = np.zeros((params["n_iter"], 1)) 
            L_seq = np.zeros((params["n_iter"], 1)) 

    for i in range(params["n_iter"]):

        if(params['l'] == 'exp'):
            # fit to the weighted samples 
            weak_l.fit(X_train, Y_train, sample_weight = d)

        if(params['l'] == 'logit'):
            # fit to the weighted samples 
            d, z = weights_and_response(Y_train01, p)
            weak_l.fit(X_train, z,  sample_weight = d)

        if(params['l'] == 'lad'):
            # fit to the negative gradient 
            weak_l.fit(X_train, np.sign(Y_train - pred_train))
        
        if(params['l'] == 'l2'):
            # fit to the negative gradient 
            weak_l.fit(X_train, Y_train - pred_train)


        # predict - storing the predictions of all the weak learners thus far
        pred_train_weak[:,i] = weak_l.predict(X_train)
        pred_test_weak[:,i] = weak_l.predict(X_test)

        
        
        
        alpha[i] = get_best_last_alpha()
        pred_train = pred_train + alpha[i]*pred_train_weak[:,i]
        pred_test = pred_test + alpha[i]*pred_test_weak[:,i]
        indices = get_indices(params, i, pred_train, pred_train_weak, Y_train)

            
        if(params['is_acceleration'] is None  or params['is_acceleration'] == 'polyak'): 

            if len(indices) > 1:
                for j in range(params["T_gd"]):
                    #print (':')
                    # compute gradient and loss at current iterate
                    loss_old, grad = compute_loss_grad(pred_train, alpha)
                    grad = grad + params['l2']  * alpha
                 
                    
                    if not params['is_line_search']:
                        # do one-step of gradient descent with original step-size (should be set to 1 if using line-search)
                        step_size = params["step_size"]
                    else:
                        step_size =  linesearch(1., params['linesearch_params'], grad, loss_old)
                        
                    # refresh pred_train
                    alpha_new = alpha - step_size * grad
                    if params["is_acceleration"] == 2:
                        alpha_new = alpha_new + params['momentum'] * (alpha - alpha_old)
                    alpha_old = alpha
                    alpha = alpha_new
                    
                    pred_train = refresh_pred(pred_train, alpha, pred_train_weak, indices, alpha)

                    
                    
        
            
        elif (params['is_acceleration'] == 'nesterov'):


            
            if i == 0: 
                
                alpha_z[indices] = alpha[indices]

                # update theta
                theta = (-theta_old + np.sqrt( theta_old**2. + 4. * theta_old)) / 2.

                alpha_y = (1. - theta) * alpha + theta * alpha_z

            
                # update xk
                alpha = alpha_y +0.

                # updating theta
                theta_old = theta

                # update yk_old
                alpha_yold = alpha_y



            else:
                for j in range(params["T_gd"]):

                    alpha_z[indices] = alpha[indices]

                    # update theta
                    theta = (-theta_old + np.sqrt( theta_old**2. + 4. * theta_old)) / 2.

                    alpha_y = (1. - theta) * alpha + theta * alpha_z

                    # refresh pred_train
                    pred_train = refresh_pred(pred_train, alpha_y, pred_train_weak, indices, alpha_yold) 

                    # compute gradient and loss at yk
                    loss, grad = compute_loss_grad(pred_train, alpha)
                    grad = grad + params['l2']  * alpha


                    if not params['is_line_search']:
                        # do one-step of gradient descent with original step-size (should be set to 1 if using line-search)
                        step_size = params["step_size"]
                    else:
                        step_size =  linesearch(1., params['linesearch_params'], grad, loss)



                    # update xk
                    alpha = alpha_y - step_size * grad

                    # update zk
                    alpha_z = alpha_z - (step_size_org / ((i+1.) * theta)) * grad

                    # updating theta
                    theta_old = theta

                    # update yk_old
                    alpha_yold = alpha_y


        



        pred_train = refresh_pred(pred_train, alpha, pred_train_weak)
        pred_test = refresh_pred(pred_test, alpha, pred_test_weak) 

        if(params['l'] == 'exp'):
            w = np.exp(-Y_train*pred_train)
            # normalize 
            d = w/np.sum(w)  
            v,g = exp_loss(pred_train, Y_train,alpha, params, compute_fval = True, compute_grad = False)

        if(params['l'] == 'logit'):
            p = expit(pred_train)
            v,g = logit_loss(pred_train, Y_train, alpha, params, compute_fval = True, compute_grad = False)
      
        if(params['l'] == 'lad'):
            v,g = LAD_loss(pred_train, Y_train,alpha, params, compute_fval = True, compute_grad = False)
            vtest, gtest = LAD_loss(pred_test, Y_test,alpha, params, compute_fval = True, compute_grad = False)
        
        if(params['l'] == 'l2'):
            v,g = L2_loss(pred_train, Y_train,alpha, params, compute_fval = True, compute_grad = False)
            vtest, gtest = L2_loss(pred_test, Y_test,alpha, params, compute_fval = True, compute_grad = False)


      

        err_train.append(get_error_rate(np.sign(pred_train), Y_train, params))
        err_test.append(get_error_rate(np.sign(pred_test), Y_test, params))
        loss_train.append(v)
        
        if params['l'] == 'l2' or params['l'] == 'lad':
            loss_test.append(vtest)

    return {'err_train': err_train, 'err_test': err_test, 'loss_train':loss_train, 'loss_test':loss_test}
