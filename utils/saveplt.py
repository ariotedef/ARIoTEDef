"""
Author: maggie
Date:   2021-06-15
@copyright
"""

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

def SaveTotalTimeCurve(model,dataset,exp_result_dir,global_cost_time,png_name):
    x = list(range(len(global_cost_time)))
    y = global_cost_time

    plt.title(f'{png_name}')
    plt.plot(x, y, color='black') 
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Total Cost Time (seconds)')
    plt.show()
    plt.savefig(f'{exp_result_dir}/{png_name}.png')
    plt.close()
    
def SaveTimeCurve(model,dataset,exp_result_dir,global_cost_time,png_name):
    x = list(range(len(global_cost_time)))
    y = global_cost_time

    plt.title(f'{png_name}')
    plt.plot(x, y, color='black') 
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Cost Time (seconds)')
    plt.show()
    plt.savefig(f'{exp_result_dir}/{png_name}.png')
    plt.close()
    
def SaveAccuracyCurve(model,dataset,exp_result_dir,global_train_acc,global_test_acc,png_name):
    train_x = list(range(len(global_train_acc)))
    train_y = global_train_acc
    test_x = list(range(len(global_test_acc)))
    test_y = global_test_acc
    plt.title(f'{png_name}')
    plt.plot(train_x, train_y, color='black', label='train acc') 
    plt.plot(test_x, test_y, color='red', label='test acc') 
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    
    plt.show()
    plt.savefig(f'{exp_result_dir}/{png_name}.png')
    plt.close()

def SaveLossCurve(model,dataset,exp_result_dir,global_train_loss,global_test_loss,png_name):
    train_x = list(range(len(global_train_loss)))
    train_y = global_train_loss
    test_x = list(range(len(global_test_loss)))
    test_y = global_test_loss
    plt.title(f'{png_name}')
    plt.plot(train_x, train_y, color='black', label='train loss')     
    plt.plot(test_x, test_y, color='red', label='test loss') 

    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    plt.savefig(f'{exp_result_dir}/{png_name}.png')
    plt.close()
    

def Save3AccuracyCurve(model,dataset,exp_result_dir,global_train_acc, global_cle_test_acc, global_adv_test_acc,png_name):
    train_x = list(range(len(global_train_acc)))
    train_y = global_train_acc
    cle_test_x = list(range(len(global_cle_test_acc)))
    cle_test_y = global_cle_test_acc
    adv_test_x = list(range(len(global_adv_test_acc)))
    adv_test_y = global_adv_test_acc    
    
    plt.title(f'{png_name}')
    plt.plot(train_x, train_y, color='black', label='train acc') 
    plt.plot(cle_test_x, cle_test_y, color='red', label='clean test acc') 
    plt.plot(adv_test_x, adv_test_y, color='blue', label='robust test acc') 
    
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    

    plt.show()
    plt.savefig(f'{exp_result_dir}/{png_name}.png')
    plt.close()

def Save3LossCurve(model,dataset,exp_result_dir,global_train_loss, global_cle_test_loss, global_adv_test_loss,png_name):
    train_x = list(range(len(global_train_loss)))
    train_y = global_train_loss
    cle_test_x = list(range(len(global_cle_test_loss)))
    cle_test_y = global_cle_test_loss
    adv_test_x = list(range(len(global_adv_test_loss)))
    adv_test_y = global_adv_test_loss    
    
    plt.title(f'{png_name}')
    plt.plot(train_x, train_y, color='black', label='train loss')                
    plt.plot(cle_test_x, cle_test_y, color='red', label='clean test loss') 
    
    plt.plot(adv_test_x, adv_test_y, color='blue', label='robust test loss') 
    
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    plt.savefig(f'{exp_result_dir}/{png_name}.png')
    plt.close()