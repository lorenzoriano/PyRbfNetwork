import numpy as np
import cPickle

execfile("__init__.py")
import utils

print "Regression"
input1 = np.linspace(0, np.pi, 50)
input2 = np.linspace(-np.pi/2, np.pi/2, 50)

output = np.sin(input1) * np.cos(input2)
output = np.array(output, ndmin=2).T

tot_input = np.vstack( (input1, input2) ).T

for sgh in xrange(100):
    net = RbfNetwork(2, 1, 0.1)
    utils.brute_force_training(net, (tot_input,output), (tot_input,output), 5, classifier=False)

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

print "Done"
