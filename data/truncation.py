

def truncationdata(multistep_dataset):

    # Truncate to a multiple of 32

    new_infection_train_size = (len(multistep_dataset['infection']['train'][0]) // 32) * 32    
    truncate_train_data_infection = multistep_dataset['infection']['train'][0][:new_infection_train_size] 
    truncate_train_label_infection = multistep_dataset['infection']['train'][1][:new_infection_train_size] 

    new_infection_test_size = (len(multistep_dataset['infection']['test'][0]) // 32) * 32    
    truncate_test_data_infection = multistep_dataset['infection']['test'][0][:new_infection_test_size] 
    truncate_test_label_infection = multistep_dataset['infection']['test'][1][:new_infection_test_size] 
    
        
    new_attack_train_size = (len(multistep_dataset['attack']['train'][0]) // 32) * 32    
    truncate_train_data_attack = multistep_dataset['attack']['train'][0][:new_attack_train_size] 
    truncate_train_label_attack = multistep_dataset['attack']['train'][1][:new_attack_train_size] 

    new_attack_test_size = (len(multistep_dataset['attack']['test'][0]) // 32) * 32    
    truncate_test_data_attack = multistep_dataset['attack']['test'][0][:new_attack_test_size] 
    truncate_test_label_attack = multistep_dataset['attack']['test'][1][:new_attack_test_size] 
    
    
    new_reconnaissance_train_size = (len(multistep_dataset['reconnaissance']['train'][0]) // 32) * 32    
    truncate_train_data_reconnaissance = multistep_dataset['reconnaissance']['train'][0][:new_reconnaissance_train_size] 
    truncate_train_label_reconnaissance = multistep_dataset['reconnaissance']['train'][1][:new_reconnaissance_train_size] 

    new_reconnaissance_test_size = (len(multistep_dataset['reconnaissance']['test'][0]) // 32) * 32    
    truncate_test_data_reconnaissance = multistep_dataset['reconnaissance']['test'][0][:new_reconnaissance_test_size] 
    truncate_test_label_reconnaissance = multistep_dataset['reconnaissance']['test'][1][:new_reconnaissance_test_size]     
    
    truncate_multistep_dataset = {"infection": 
                            {
                            'train': [truncate_train_data_infection, truncate_train_label_infection], 
                            'test': [truncate_test_data_infection, truncate_test_label_infection]
                            },
                "attack": 
                            {
                            'train': [truncate_train_data_attack, truncate_train_label_attack], 
                            'test': [truncate_test_data_attack, truncate_test_label_attack]
                            },
                "reconnaissance": 
                            {
                            'train': [truncate_train_data_reconnaissance, truncate_train_label_reconnaissance], 
                            'test': [truncate_test_data_reconnaissance, truncate_test_label_reconnaissance]
                            }
                }        
                
    return truncate_multistep_dataset    