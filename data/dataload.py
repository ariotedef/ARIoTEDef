import numpy as np

def loadnpydata():

    test_label_infection = np.load('preprocessed/test_label_infection.npy')
    train_label_infection = np.load('preprocessed/train_label_infection.npy')
    test_data_infection = np.load('preprocessed/test_data_infection.npy')
    train_data_infection = np.load('preprocessed/train_data_infection.npy')

    test_label_reconnaissance = np.load('preprocessed/test_label_reconnaissance.npy')
    train_label_reconnaissance = np.load('preprocessed/train_label_reconnaissance.npy')
    test_data_reconnaissance = np.load('preprocessed/test_data_reconnaissance.npy')
    train_data_reconnaissance = np.load('preprocessed/train_data_reconnaissance.npy')

    test_label_attack = np.load('preprocessed/test_label_attack.npy')
    train_label_attack = np.load('preprocessed/train_label_attack.npy')
    test_data_attack = np.load('preprocessed/test_data_attack.npy')
    train_data_attack = np.load('preprocessed/train_data_attack.npy')




    multistep_dataset = {"infection": 
                            {
                            'train': [train_data_infection, train_label_infection], 
                            'test': [test_data_infection, test_label_infection]
                            },
                "attack": 
                            {
                            'train': [train_data_attack, train_label_attack], 
                            'test': [test_data_attack, test_label_attack]
                            },
                "reconnaissance": 
                            {
                            'train': [train_data_reconnaissance, train_label_reconnaissance], 
                            'test': [test_data_reconnaissance, test_label_reconnaissance]
                            }
                }
    
    return multistep_dataset