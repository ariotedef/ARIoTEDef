
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def SS_normalizedata(dataset_x):
    # dataset_min = np.min(dataset_x)
    # dataset_max = np.max(dataset_x)
    # print(f"dataset_min:{dataset_min}")
    # print(f"dataset_max:{dataset_max}") 
    
    # scaler = StandardScaler().fit(dataset_x)
    # dataset_x = scaler.transform(dataset_x)
    scaler = StandardScaler()
    dataset_x = scaler.fit_transform(dataset_x)
    
    # dataset_min = np.min(dataset_x)
    # dataset_max = np.max(dataset_x)
    # print(f"dataset_min:{dataset_min}")
    # print(f"dataset_max:{dataset_max}") 
          
    return dataset_x

def MinMax_normalizedata(dataset_x):
    # dataset_min = np.min(dataset_x)
    # dataset_max = np.max(dataset_x)
    # print(f"dataset_min:{dataset_min}")
    # print(f"dataset_max:{dataset_max}") 
    

    # 创建MinMaxScaler对象
    scaler = MinMaxScaler()
    # 拟合并转换数据
    dataset_x = scaler.fit_transform(dataset_x)


    # dataset_min = np.min(dataset_x)
    # dataset_max = np.max(dataset_x)
    # print(f"dataset_min:{dataset_min}")
    # print(f"dataset_max:{dataset_max}") 
          
    return dataset_x

def normalize_multistep_dataset(multistep_dataset):

    raw_trainset_min = np.min(multistep_dataset['reconnaissance']['train'][0])
    raw_trainset_max = np.max(multistep_dataset['reconnaissance']['train'][0])
    raw_testset_min = np.min(multistep_dataset['reconnaissance']['test'][0])
    raw_testset_max = np.max(multistep_dataset['reconnaissance']['test'][0])    
    print(f"reconnaissance trainset_min:{raw_trainset_min:.4f}")
    print(f"reconnaissance trainset_max:{raw_trainset_max:.4f}")
    print(f"reconnaissance testset_min:{raw_testset_min:.4f}")
    print(f"reconnaissance testset_max:{raw_testset_max:.4f}")  
        
    norm_train_data_reconnaissance = SS_normalizedata(multistep_dataset['reconnaissance']['train'][0]) 
    train_label_reconnaissance = multistep_dataset['reconnaissance']['train'][1]
    
    norm_test_data_reconnaissance = SS_normalizedata(multistep_dataset['reconnaissance']['test'][0])     
    test_label_reconnaissance = multistep_dataset['reconnaissance']['test'][1]
    
    
    raw_trainset_min = np.min(multistep_dataset['infection']['train'][0])
    raw_trainset_max = np.max(multistep_dataset['infection']['train'][0])
    raw_testset_min = np.min(multistep_dataset['infection']['test'][0])
    raw_testset_max = np.max(multistep_dataset['infection']['test'][0])    
    print(f"infection trainset_min:{raw_trainset_min:.4f}")
    print(f"infection trainset_max:{raw_trainset_max:.4f}")
    print(f"infection testset_min:{raw_testset_min:.4f}")
    print(f"infection testset_max:{raw_testset_max:.4f}")   
        
        
    norm_train_data_infection = SS_normalizedata(multistep_dataset['infection']['train'][0]) 
    train_label_infection = multistep_dataset['infection']['train'][1]
    
    norm_test_data_infection = SS_normalizedata(multistep_dataset['infection']['test'][0]) 
    test_label_infection = multistep_dataset['infection']['test'][1]


    raw_trainset_min = np.min(multistep_dataset['attack']['train'][0])
    raw_trainset_max = np.max(multistep_dataset['attack']['train'][0])
    raw_testset_min = np.min(multistep_dataset['attack']['test'][0])
    raw_testset_max = np.max(multistep_dataset['attack']['test'][0])    
    print(f"attack trainset_min:{raw_trainset_min:.4f}")
    print(f"attack trainset_max:{raw_trainset_max:.4f}")
    print(f"attack testset_min:{raw_testset_min:.4f}")
    print(f"attack testset_max:{raw_testset_max:.4f}")   
        
    norm_train_data_attack = SS_normalizedata(multistep_dataset['attack']['train'][0]) 
    train_label_attack = multistep_dataset['attack']['train'][1]
    
    norm_test_data_attack = SS_normalizedata(multistep_dataset['attack']['test'][0]) 
    test_label_attack = multistep_dataset['attack']['test'][1]



    
    norm_multistep_dataset = {"infection": 
                            {
                            'train': [norm_train_data_infection, train_label_infection], 
                            'test': [norm_test_data_infection, test_label_infection]
                            },
                "attack": 
                            {
                            'train': [norm_train_data_attack, train_label_attack], 
                            'test': [norm_test_data_attack, test_label_attack]
                            },
                "reconnaissance": 
                            {
                            'train': [norm_train_data_reconnaissance, train_label_reconnaissance], 
                            'test': [norm_test_data_reconnaissance, test_label_reconnaissance]
                            }
                }        
                
    return norm_multistep_dataset