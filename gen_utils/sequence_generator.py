import numpy as np 

def generate_from_seed(model, seed, sequence_length, data_variance, data_mean, verbose=True):

    seedSeq = seed.copy()
    if verbose:
        print('first')
        print(seedSeq.shape)

    output = []

    #The generation algorithm is simple:
    #Step 1 - Given A = [X_0, X_1, ... X_n], generate X_n + 1
    #Step 2 - Concatenate X_n + 1 onto A
    #Step 3 - Repeat MAX_SEQ_LEN times
    for it in range(sequence_length):

        seedSeqNew = model.predict(seedSeq) #Step 1 Generate X_n + 1
        #Step 2 Append it to the sequence
        if it == 0:
            for i in range(seedSeqNew.shape[1]):
                output.append(seedSeqNew[0][i].copy())

        else:
            output.append(seedSeqNew[0][seedSeqNew.shape[1]-1].copy())
        
        newSeq = seedSeqNew[0][seedSeqNew.shape[1]-1]
        if verbose:
            print('second')
            print(newSeq.shape)
        newSeq = np.reshape(newSeq, (1, 1, newSeq.shape[0]))
        if verbose:
            print('third')
            print(newSeq.shape)

    #Finally, post-process the generated sequence so that we have valid frequencies
    #Essentially, we are undoing the data centering process
    for i in range(len(output)):

        output[i] *= data_variance
        output[i] += data_mean
        if verbose:
            print(output)
    
    return output