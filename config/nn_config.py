
def get_NN_config():

    nn_params = {

        'sampling_frequency': 44100,
        'hidden_dimension_size': 1024,
        'model_basename': './NNweights',
        'model_file': './data/NN',
        'data_dir': './data/audio/'
    }

    return nn_params