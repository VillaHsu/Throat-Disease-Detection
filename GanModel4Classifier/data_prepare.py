#!/usr/bin/env python
#import bz2
import csv
import numpy as np
import sys
import theano

def load_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "../..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval
    
def load_ivectors_nohead(filename):
    """Loads ivectors

    Parameters
    ----------
    filename : string
        Path to ivector files (e.g. dev_ivectors.csv)

    Returns
    -------
    ids : list
        List of ivectorids
    durations : array, shaped('n_ivectors')
        Array of durations for each ivectorid
    languages : array, shaped('n_ivectors')
        Array of langs for each ivectorid (only applies to train)
    ivectors : array, shaped('n_ivectors', 600)
        Array of ivectors for each ivectorid
    """
    ids = []
    durations = []
    languages = []
    ivectors = []
    with open(filename, 'rU') as infile:
        #reader = csv.reader(infile, delimiter='\t')
        #reader.next()
        i = 0
        for row in csv.reader(infile, delimiter='\t'):
            i = i +1
            ids.append(row[0])
            durations.append(float(row[1]))
            languages.append(row[2])
            ivectors.append(np.asarray(row[3:], dtype=np.float32))

            sys.stdout.write("\r  %d   %s  " % (i, row[0]))
            sys.stdout.flush()

    print("\n   I-    Adding Transformed ivectors ")

    return ids, np.array(durations, dtype=np.float32), np.array(languages), np.vstack(ivectors)    

def load_ivectors_nohead2(filename):
    """Loads ivectors

    Parameters
    ----------
    filename : string
        Path to ivector files (e.g. dev_ivectors.csv)

    Returns
    -------
    ids : list
        List of ivectorids
    durations : array, shaped('n_ivectors')
        Array of durations for each ivectorid
    languages : array, shaped('n_ivectors')
        Array of langs for each ivectorid (only applies to train)
    ivectors : array, shaped('n_ivectors', 600)
        Array of ivectors for each ivectorid
    """
    ids = []
    durations = []
    languages = []
    ivectors = []
    with open(filename, 'rU') as infile:
        #reader = csv.reader(infile, delimiter='\t')
        #reader.next()

        for row in csv.reader(infile, delimiter='\t'):
            ids.append(row[0])
            durations.append(float(row[1]))
            languages.append(row[2])
            str=row[3]
            newStr=str.split(" ")
            ivectors.append(np.asarray(newStr, dtype=np.float32))

            #sys.stdout.write("\r     %s  " % row[0])
            #sys.stdout.flush()

    print "\n   I-    Adding Transformed ivectors "

    return np.array(ids), np.array(durations, dtype=np.float32), np.array(languages), np.vstack(ivectors)    
    
#Return language ID from lan_list        
def findLanID(lan_list, str):
    i = 0;
    while i < lan_list.size:
        if str==lan_list[i]:
            return i;
        i=i+1;
    return -1;

def shared_dataset(data_x, data_y, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    #data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32')    
