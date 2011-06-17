import numpy

def brute_force_training(net, training_set, validation_set, numtrials, classifier=False,
                         min_sigma = 0.001, max_sigma = 1.0, min_kernels = None, max_kernels = None):
    
    train_input, train_output =  training_set
    validate_input, validate_output = validation_set

    if classifier:    
        validate_output = validate_output.reshape( (validate_output.shape[0], 1) )
    else:
        validate_output = validate_output.reshape( (validate_output.shape[0], net.output_size) )

    if min_kernels is None:
        min_kernels = train_input.shape[0] / 20
    if max_kernels is None:
        max_kernels = train_input.shape[0] / 2
    
    best_err = numpy.finfo(numpy.float64).max
    for trial in xrange(numtrials):
        sigma = numpy.random.uniform(min_sigma, max_sigma)
        num_kernels = numpy.random.randint(min_kernels, max_kernels)
        
        net.sigma = sigma
        net.select_random_kernels(train_input, num_kernels)
        err = net.lsqtrain(train_input, train_output)
#        print "Training err: ", numpy.mean(err)
        
        netout = net(validate_input)
        
        
        if classifier:
            err = netout != validate_output            
        else:
            err = numpy.abs(netout - validate_output) 
        
        err = numpy.mean(err[:])
#        print "Validate err: ", err
        if err < best_err:            
            best_sigma = sigma
            best_num_kernels = num_kernels
            best_err = err
            print "New low error: ",best_err," at iteration ",trial
            print "Best sigma: ", best_sigma, " best num_kernels: ", best_num_kernels
            
    
    net.sigma = best_sigma
    net.select_random_kernels(train_input, best_num_kernels)
    err = net.lsqtrain(train_input, train_output)
    print "Final error on training is: ", numpy.mean(err[:])
    
    netout = net(validate_input)
    if classifier:
        err = netout != validate_output
    else:
        err = numpy.abs(netout - validate_output) 
        
    print "Final error on validation is: ", numpy.mean(err[:])
    