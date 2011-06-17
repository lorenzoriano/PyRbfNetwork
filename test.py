import numpy as np
import cPickle
import matplotlib.pylab as pylab

execfile("__init__.py")
import utils

print "Regression"
input1 = np.linspace(0, np.pi, 100)
input2 = 0.5 + np.linspace(-np.pi/2, np.pi/2, 100)

output = np.sin(input1) * np.cos(input2)
output = np.array(output, ndmin=2).T

tot_input = np.vstack( (input1, input2) ).T

normalizer = Normalizer()
normalizer.calculate_from_input(tot_input)
norm_input =  normalizer.normalize(tot_input)

for sgh in xrange(1):
    net = RbfNetwork(2, 1, 0.1)
    utils.brute_force_training(net, (norm_input, output+np.random.randn(*output.shape)*0.1), (norm_input,output), 500, classifier=False)

denorm_input = normalizer.denormalize(norm_input)
print "All close? ", np.allclose(denorm_input.ravel(), tot_input.ravel())

netout, conf = net.output_conf(norm_input)
#pylab.plot(output, label="true")
#pylab.plot(netout, label="rbfn")
#pylab.legend()

#pylab.figure()
#pylab.plot(conf)
#pylab.title("Regression")
#pylab.show()


s = cPickle.dumps(net)
net = cPickle.loads(s)
print "Pickle OK"

print "\nClassification"
inputs = np.random.rand(500,2)
outs = inputs[:,0] + inputs[:,1] > 0.5
for sgh in xrange(5):
    classifier = RbfClassifier(2,2,0.1)
    utils.brute_force_training(classifier, (inputs,outs), (inputs,outs), 5, classifier=True)

trials = 10
res = 0.0
for sgh in xrange(trials):
    vals = np.random.rand(2)
    outcome = np.sum(vals) > 0.5
    netout = classifier.output(vals)
    res += np.abs(netout - outcome)

print "Classification success: ", 1.0 - (res / trials)

inpt = [0.3, 0.1]
netout, conf = classifier.output_conf(inpt)
print "Input: ", inpt, " class: ", netout, " confidence: ", conf


print "Done"
