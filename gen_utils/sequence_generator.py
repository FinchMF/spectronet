import numpy as np 

def generate_from_seed(model, seed, sequence_length, data_variance, data_mean, verbose=True):
    # copy sequence
    seedSeq = seed.copy()

    if verbose:
        print('first')
        print(seedSeq.shape)
    # set container to collect sequence in
    output = []

    for it in range(sequence_length):
        # generate new step
        seedSeqNew = model.predict(seedSeq) 
        # append
        if it == 0:
            for i in range(seedSeqNew.shape[1]):
                output.append(seedSeqNew[0][i].copy())
        else:
            output.append(seedSeqNew[0][seedSeqNew.shape[1]-1].copy())
        # process to generate next sequence to predict with
        newSeq = seedSeqNew[0][seedSeqNew.shape[1]-1]
        if verbose:
            print('second')
            print(newSeq.shape)
        newSeq = np.reshape(newSeq, (1, 1, newSeq.shape[0]))
        if verbose:
            print('third')
            print(newSeq.shape)
        seedSeq = np.concatenate((seedSeq, newSeq), axis=1)
        if verbose:
            print('fourth')
            print(seedSeq.shape)

    # post process sequence
    for i in range(len(output)):

        output[i] *= data_variance
        output[i] += data_mean
        if verbose:
            print(output)
    
    return output