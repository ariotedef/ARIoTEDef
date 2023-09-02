""" 
written by Mengdie Huang, Daniel de Mello, and Zilin Shen
Date: May 8, 2023
"""

import os
os.environ['TF_NUMA_NODES'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import numpy as np
from utils.printformat import print_header
from utils.argsparse import get_args
from utils.savedir import set_exp_result_dir
from utils.events import get_events_from_windows

from data.dataload import loadnpydata
from data.normalize import normalize_multistep_dataset
from models.createmodel import init_psdetector
from models.createmodel import init_seq2seq
import copy
import math
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle

# -----------parse parameters-----------
print("\n")
args = get_args()
print("args:",args)

# -----------set save path-----------
exp_result_dir = set_exp_result_dir(args)
os.makedirs(exp_result_dir, exist_ok=True)
print("exp_result_dir:",exp_result_dir)

# -----------get the preprocessed training and testing saved as .npy files
multistep_dataset = loadnpydata()
    
# -----------normalize sample values    
multistep_dataset = normalize_multistep_dataset(multistep_dataset)

# ----------------create per-step detectors----------------------
reconnaissance_detector, infection_detector, attack_detector = init_psdetector(multistep_dataset, args)

# ----------------train per-step detectors----------------------
print_header("Train Per-Step Detector")
for detector in [reconnaissance_detector, infection_detector, attack_detector]:
    
    if args.stdtrain_pedetector is True:
        
        print(f">>>>>>>> Training {detector.modelname} >>>>>>>>")  
        stdtrain_exp_result_dir = os.path.join(exp_result_dir,f'stdtrain-psdetector')
        os.makedirs(stdtrain_exp_result_dir, exist_ok=True)

        detector.stdtrain(timesteps=args.timesteps, exp_result_dir=stdtrain_exp_result_dir)
        
        print(f">>>>>>>> Evaluate {detector.modelname} on clean test data")
        test_acc, test_los, test_TP, test_FP, test_TN, test_FN, test_recall, test_precision, test_F1 = detector.test(testset_x=detector.dataset['test'][0], testset_y=detector.dataset['test'][1],timesteps=args.timesteps)
        
        metrics_dic = { 
                    'model': detector.modelname,
                    'clean test Accuracy': f'{test_acc*100:.2f}%',
                    'clean test Loss': test_los,
                    'clean test TP': test_TP,
                    'clean test FP': test_FP,
                    'clean test TN': test_TN,
                    'clean test FN': test_FN,
                    'clean test Recall': f'{test_recall*100:.2f}%',
                    'clean test Precision': f'{test_precision*100:.2f}%',
                    'clean test F1': f'{test_F1*100:.2f}%',
                    }
        
        print(f"{detector.modelname} metrics_dic:\n {metrics_dic}")   
        
        detector_save_path = f'{exp_result_dir}/{detector.modelname}-acc-{test_acc:.4f}.h5'
        detector.save_model(detector_save_path)
        
    elif args.stdtrain_pedetector is False:
        print(f">>>>>>>> Evaluate load {detector.modelname} on clean test data")

        pretrain_exp_result_dir = os.path.join(exp_result_dir,f'pretrain-psdetector')
        os.makedirs(pretrain_exp_result_dir, exist_ok=True)
                
        test_acc, test_los, test_TP, test_FP, test_TN, test_FN, test_recall, test_precision, test_F1 = detector.test(testset_x=detector.dataset['test'][0], testset_y=detector.dataset['test'][1],timesteps=args.timesteps)
        
        metrics_dic = { 
                    'model': detector.modelname,
                    'clean test Accuracy': f'{test_acc*100:.2f}%',
                    'clean test Loss': test_los,
                    'clean test TP': test_TP,
                    'clean test FP': test_FP,
                    'clean test TN': test_TN,
                    'clean test FN': test_FN,
                    'clean test Recall': f'{test_recall*100:.2f}%',
                    'clean test Precision': f'{test_precision*100:.2f}%',
                    'clean test F1': f'{test_F1*100:.2f}%',
                    }        
        print(f"{detector.modelname} metrics_dic:\n {metrics_dic}")   
             
# ----------------adversarial attack vanilla per-step detectors----------------------
print_header("Adversarial Attack Vanilla Per-Step Detector")
for detector in [reconnaissance_detector, infection_detector, attack_detector]:
   
    # generate adversarial mailicious testset
    print(f"Generate adversarial mailicious exapmples based white-box {detector.modelname}")

    adv_exp_result_dir = os.path.join(exp_result_dir,f'advattack')
    os.makedirs(adv_exp_result_dir, exist_ok=True)
        
    # adv_testset_x, adv_testset_y = detector.generate_advmail(timesteps=args.timesteps)
    adv_testset_x, adv_testset_y = detector.generate_advmail(timesteps=args.timesteps,cle_testset_x=detector.dataset['test'][0],cle_testset_y=detector.dataset['test'][1])
    
    print("adv_testset_x.shape:",adv_testset_x.shape)    
    print("adv_testset_y.shape:",adv_testset_y.shape)    

    adv_test_acc, adv_test_los, adv_test_TP, adv_test_FP, adv_test_TN, adv_test_FN, adv_test_recall, adv_test_precision, adv_test_F1 = detector.test(testset_x=adv_testset_x, testset_y=adv_testset_y, timesteps=args.timesteps)
    
    adv_metrics_dic = { 
                   'model': detector.modelname,
                   'adv test Accuracy': f'{adv_test_acc*100:.2f}%',
                   'adv test Loss': adv_test_los,
                   'adv test TP': adv_test_TP,
                   'adv test FP': adv_test_FP,
                   'adv test TN': adv_test_TN,
                   'adv test FN': adv_test_FN,
                   'adv test Recall': f'{adv_test_recall*100:.2f}%',
                   'adv test Precision': f'{adv_test_precision*100:.2f}%',
                   'adv test F1': f'{adv_test_F1*100:.2f}%',
                }
     
    print(f"{detector.modelname} adv_metrics_dic:\n {adv_metrics_dic}")           

# ----------------create seq2seq----------------------
infection_seq2seq = init_seq2seq(multistep_dataset, args)

# ----------------train seq2seq----------------------
print_header("Train Infection Seq2Seq")

for seq2seq in [infection_seq2seq]:
 
    if args.stdtrain_seq2seq is True:
        
        # create trainset
        cle_train_windows_x = seq2seq.dataset['train'][0]
        cle_train_windows_y = seq2seq.dataset['train'][1]
        

        cle_train_windows_x = cle_train_windows_x.reshape((cle_train_windows_x.shape[0], args.timesteps, int(math.ceil(cle_train_windows_x.shape[1] / args.timesteps))))
        print("cle_train_windows_x.shape:",cle_train_windows_x.shape)

        seq2seq_train_events = get_events_from_windows(reconnaissance_detector, infection_detector, attack_detector, cle_train_windows_x)
   
        seq2seq.def_model(input_length=args.sequence_length, output_length =args.sequence_length)
        
        stdtrain_exp_result_dir = os.path.join(exp_result_dir,f'stdtrain-seq2seq')
        os.makedirs(stdtrain_exp_result_dir, exist_ok=True)
        
        print(f">>>>>>>> Training {seq2seq.modelname} >>>>>>>>")      
        seq2seq.stdtrain(events=seq2seq_train_events, labels=cle_train_windows_y, exp_result_dir=stdtrain_exp_result_dir)
            
        print(f">>>>>>>> Evaluate {seq2seq.modelname} on clean test data")
        
        # create testset
        cle_test_windows_x = seq2seq.dataset['test'][0]
        cle_test_windows_y = seq2seq.dataset['test'][1]

        cle_test_windows_x = cle_test_windows_x.reshape((cle_test_windows_x.shape[0], args.timesteps, int(math.ceil(cle_test_windows_x.shape[1] / args.timesteps))))
        print("cle_test_windows_x.shape:",cle_test_windows_x.shape)
        
        cle_test_events_x = get_events_from_windows(reconnaissance_detector, infection_detector, attack_detector, cle_test_windows_x)
        print("cle_test_events_x.shape:",cle_test_events_x.shape)
       
        test_acc, test_los, test_TP, test_FP, test_TN, test_FN, test_recall, test_precision, test_F1 = seq2seq.test(events=cle_test_events_x, labels=cle_test_windows_y)
        
        metrics_dic = { 
                    'model': seq2seq.modelname,
                    'clean test Accuracy': f'{test_acc*100:.2f}%',
                    'clean test Loss': test_los,
                    'clean test TP': test_TP,
                    'clean test FP': test_FP,
                    'clean test TN': test_TN,
                    'clean test FN': test_FN,
                    'clean test Recall': f'{test_recall*100:.2f}%',
                    'clean test Precision': f'{test_precision*100:.2f}%',
                    'clean test F1': f'{test_F1*100:.2f}%',
                    }
        
        print(f"{seq2seq.modelname} metrics_dic:\n {metrics_dic}")       

        seq2seq_save_path = f'{exp_result_dir}/{seq2seq.modelname}-acc-{test_acc:.4f}.h5'
        seq2seq.save_model(seq2seq_save_path)

    elif args.stdtrain_seq2seq is False:
        print(f">>>>>>>> Evaluate load {seq2seq.modelname} on clean test data")
        
        pretrain_exp_result_dir = os.path.join(exp_result_dir,f'pretrain-seq2seq')
        os.makedirs(pretrain_exp_result_dir, exist_ok=True)

        
        # create testset
        cle_test_windows_x = seq2seq.dataset['test'][0]
        cle_test_windows_y = seq2seq.dataset['test'][1]
        # print("cle_test_windows_x.shape:",cle_test_windows_x.shape)
        # print("cle_test_windows_y.shape:",cle_test_windows_y.shape)
        cle_test_windows_x = cle_test_windows_x.reshape((cle_test_windows_x.shape[0], args.timesteps, int(math.ceil(cle_test_windows_x.shape[1] / args.timesteps))))
        print("cle_test_windows_x.shape:",cle_test_windows_x.shape)

        cle_test_events_x = get_events_from_windows(reconnaissance_detector, infection_detector, attack_detector, cle_test_windows_x)
        print("cle_test_events_x.shape:",cle_test_events_x.shape)       
        print("cle_test_events_x[:2]:",cle_test_events_x[:2])
        
        test_acc, test_los, test_TP, test_FP, test_TN, test_FN, test_recall, test_precision, test_F1 = seq2seq.test(events=cle_test_events_x, labels=cle_test_windows_y)
        
        metrics_dic = { 
                    'model': seq2seq.modelname,
                    'clean test Accuracy': f'{test_acc*100:.2f}%',
                    'clean test Loss': test_los,
                    'clean test TP': test_TP,
                    'clean test FP': test_FP,
                    'clean test TN': test_TN,
                    'clean test FN': test_FN,
                    'clean test Recall': f'{test_recall*100:.2f}%',
                    'clean test Precision': f'{test_precision*100:.2f}%',
                    'clean test F1': f'{test_F1*100:.2f}%',
                    }
        
        print(f"{seq2seq.modelname} metrics_dic:\n {metrics_dic}")         

# ----------------retrain vanilla per-step detectors----------------------
print_header("Retrain Vanilla Infection Detector")

for detector in [infection_detector]:

    for seq2seq in [infection_seq2seq]:


        if args.advset_mode == 'advset1':

            #---------------------------------------------
            #
            # prepare for plt
            #
            #---------------------------------------------         
            test_acc_list = [] 
            test_los_list = []
            test_TP_list = []
            test_FP_list = []
            test_TN_list = []
            test_FN_list = []
            test_recall_list = []
            test_precision_list = []
            test_FPR_list = []
            test_F1_list = []
            cost_time_list =[]
            
            test_FPrate_list=[]
            test_FNrate_list=[]

            adv_test_acc_list = [] 
            adv_test_los_list = []
            adv_test_TN_list = []
            adv_test_FN_list = []
            adv_test_recall_list = []
            adv_test_precision_list = []
            adv_test_F1_list = []
            adv_test_FNrate_list=[]
            
            detector_tagged_mal_event_num_list=[]
            detector_tagged_ben_event_num_list=[]
            seq2seq_tagged_mal_event_num_list=[]
            seq2seq_tagged_ben_event_num_list=[]
            #---------------------------------------------
            #
            # evaluate vanillia infection detector
            #
            #---------------------------------------------    
            print(f">>>>>>>> Evaluate vanillia {detector.modelname} on clean test data >>>>>>>>")
            test_acc, test_los, test_TP, test_FP, test_TN, test_FN, test_recall, test_precision, test_F1 = detector.test(testset_x=detector.dataset['test'][0], testset_y=detector.dataset['test'][1],timesteps=args.timesteps)
            FPrate = round((test_FP/(test_FP+test_TN)), 4)
            FNrate = round((test_FN/(test_FN+test_TP)), 4)
            
            metrics_dic = { 
                        'model': detector.modelname,
                        'clean test Accuracy': f'{test_acc*100:.2f}%',
                        'clean test Loss': test_los,
                        'clean test TP': test_TP,
                        'clean test FP': test_FP,
                        'clean test TN': test_TN,
                        'clean test FN': test_FN,
                        'clean test Recall': f'{test_recall*100:.2f}%',
                        'clean test Precision': f'{test_precision*100:.2f}%',
                        'clean test F1': f'{test_F1*100:.2f}%',
                        'clean test FPrate':f'{FPrate*100:.2f}%',
                        'clean test FNrate':f'{FNrate*100:.2f}%',
                        }
            print(f"vanillia {detector.modelname} metrics_dic:\n {metrics_dic}")          
            
            test_FPrate_list.append(FPrate*100)
            test_FNrate_list.append(FNrate*100)
            test_acc_list.append(test_acc*100)
            test_los_list.append(test_los)
            test_TP_list.append(test_TP)
            test_FP_list.append(test_FP)
            test_TN_list.append(test_TN)
            test_FN_list.append(test_FN)
            test_recall_list.append(test_recall*100)
            test_precision_list.append(test_precision*100)
            test_F1_list.append(test_F1*100)
            cost_time_list.append(0)        
            
                    
            print(f">>>>>>>> Evaluate vanillia {detector.modelname} on adversarial test data >>>>>>>>")
            adv_exp_result_dir = os.path.join(exp_result_dir,f'advattack')
            os.makedirs(adv_exp_result_dir, exist_ok=True)
            adv_testset_x, adv_testset_y = detector.generate_advmail(timesteps=args.timesteps,cle_testset_x=detector.dataset['test'][0],cle_testset_y=detector.dataset['test'][1])
            
            print("adv_testset_x.shape:",adv_testset_x.shape)    
            print("adv_testset_y.shape:",adv_testset_y.shape)   
            
            adv_test_acc, adv_test_los, adv_test_TP, adv_test_FP, adv_test_TN, adv_test_FN, adv_test_recall, adv_test_precision, adv_test_F1 = detector.test(testset_x=adv_testset_x, testset_y=adv_testset_y, timesteps=args.timesteps)
            adv_FNrate = round((adv_test_FN/(adv_test_FN+adv_test_TP)), 4)
            
            adv_metrics_dic = { 
                        'model': detector.modelname,
                        'adv test Accuracy': f'{adv_test_acc*100:.2f}%',
                        'adv test Loss': adv_test_los,
                        'adv test TP': adv_test_TP,
                        'adv test FP': adv_test_FP,
                        'adv test TN': adv_test_TN,
                        'adv test FN': adv_test_FN,
                        'adv test Recall': f'{adv_test_recall*100:.2f}%',
                        'adv test Precision': f'{adv_test_precision*100:.2f}%',
                        'adv test F1': f'{adv_test_F1*100:.2f}%',
                        'adv test FNrate': f'{adv_FNrate*100:.2f}%',  
                        }
            print(f"Vanillia {detector.modelname} adv_metrics_dic:\n {adv_metrics_dic}")          
            
            adv_test_FNrate_list.append(adv_FNrate*100)
            adv_test_acc_list.append(adv_test_acc*100)
            adv_test_los_list.append(adv_test_los)
            adv_test_TN_list.append(adv_test_TN)
            adv_test_FN_list.append(adv_test_FN)
            adv_test_recall_list.append(adv_test_recall*100)
            adv_test_precision_list.append(adv_test_precision*100)
            adv_test_F1_list.append(adv_test_F1*100)

            #---------------------------------------------
            #
            # multi rounds retraining
            #
            #---------------------------------------------                    
            for r in range(args.relabel_rounds):
                
                curround_exp_result_dir = os.path.join(exp_result_dir,f'round-{r+1}')
                os.makedirs(curround_exp_result_dir, exist_ok=True)

                print(f">>>>>>>> create {r+1} round retraining dataset >>>>>>>>")    

                #---------------------------------------------
                #
                # create event testset
                #
                #---------------------------------------------               
                
                #----------create clean windows testset----------
                cle_test_x = detector.dataset['test'][0]
                cle_test_y = detector.dataset['test'][1]

                #----------create adv windows testset----------         
                adv_exp_result_dir = os.path.join(curround_exp_result_dir,f'advattack')
                os.makedirs(adv_exp_result_dir, exist_ok=True)
                    
                adv_testset_x, adv_testset_y = detector.generate_advmail(timesteps=args.timesteps,cle_testset_x=detector.dataset['test'][0],cle_testset_y=detector.dataset['test'][1])                

                if args.retrain_testset_mode == 'cle_adv':
                    #----------create adv+clean windows testset----------
                    test_x = np.concatenate((adv_testset_x,cle_test_x))
                    test_y = np.concatenate((adv_testset_y,cle_test_y))            
                    print("retrain_test_(cle_adv)_x.shape:",test_x.shape)
                    print("retrain_test_(cle_adv)_y.shape:",test_y.shape)                 

                elif args.retrain_testset_mode == 'adv':
                    test_x = adv_testset_x
                    test_y = adv_testset_y   
                    print("retrain_test_(adv)_x.shape:",test_x.shape)
                    print("retrain_test_(adv)_y.shape:",test_y.shape)   
                                    
                elif args.retrain_testset_mode == 'cle':
                    test_x = cle_test_x
                    test_y = cle_test_y 
                    print("retrain_test_(cle)_x.shape:",test_x.shape)
                    print("retrain_test_(cle)_y.shape:",test_y.shape)   
                                            
                                                    
                #----------create adv+clean events testset----------
                test_windows_x = copy.deepcopy(test_x)
                test_windows_y = copy.deepcopy(test_y)           
                print("test_windows_x.shape:",test_windows_x.shape)
                print("test_windows_y.shape:",test_windows_y.shape) 
        
                                
                test_windows_x = test_windows_x.reshape((test_windows_x.shape[0], args.timesteps, int(math.ceil(test_windows_x.shape[1] / args.timesteps))))
                
                print("test_windows_x.shape:",test_windows_x.shape)

                
                test_events_x = get_events_from_windows(reconnaissance_detector, infection_detector, attack_detector, test_windows_x)
                print("test_events_x.shape:",test_events_x.shape)


                #---------------------------------------------
                #
                # analyze event testset
                #
                #---------------------------------------------  

                #----------detector predict test_windows_x----------
                print("test_windows_x.shape:",test_windows_x.shape)
 
                
                detector_tagged_mal_windows_probs, detector_tagged_mal_windows_idxs, detector_tagged_ben_windows_probs,detector_tagged_ben_windows_idxs = detector.analysis(test_windows_x)
                
                detector_tagged_mal_event_num_list.append(detector_tagged_mal_windows_idxs.shape[0])
                detector_tagged_ben_event_num_list.append(detector_tagged_ben_windows_idxs.shape[0])
                
                # #----------update seq2seq-------------------------
                if args.retrain_seq2seq is True: 
                    if r>=1: #非初次retrain
                        # create trainset
                        cle_train_windows_x = seq2seq.dataset['train'][0]
                        cle_train_windows_y = seq2seq.dataset['train'][1]
                        
                    
                        cle_train_windows_x = cle_train_windows_x.reshape((cle_train_windows_x.shape[0], args.timesteps, int(math.ceil(cle_train_windows_x.shape[1] / args.timesteps))))
                        print("cle_train_windows_x.shape:",cle_train_windows_x.shape)
                        
                        seq2seq_train_events = get_events_from_windows(reconnaissance_detector, infection_detector, attack_detector, cle_train_windows_x)
                        print("seq2seq_train_events.shape:",seq2seq_train_events.shape)
  
                        seq2seq.def_model(input_length=args.sequence_length, output_length =args.sequence_length)
                        
                        retrain_exp_result_dir = os.path.join(curround_exp_result_dir,f'retrain-seq2seq')
                        os.makedirs(retrain_exp_result_dir, exist_ok=True)
                        
                        print(f">>>>>>>> Retrain {seq2seq.modelname} >>>>>>>>")      
                        seq2seq.retrain(events=seq2seq_train_events, labels=cle_train_windows_y, exp_result_dir=retrain_exp_result_dir)
                            
                    
                        print(f">>>>>>>> Evaluate {seq2seq.modelname} on clean test data")
                        
                        # create testset
                        cle_test_windows_x = seq2seq.dataset['test'][0]
                        cle_test_windows_y = seq2seq.dataset['test'][1]
                 
                        cle_test_windows_x = cle_test_windows_x.reshape((cle_test_windows_x.shape[0], args.timesteps, int(math.ceil(cle_test_windows_x.shape[1] / args.timesteps))))
                        print("cle_test_windows_x.shape:",cle_test_windows_x.shape)
          
                        cle_test_events_x = get_events_from_windows(reconnaissance_detector, infection_detector, attack_detector, cle_test_windows_x)
                        print("cle_test_events_x.shape:",cle_test_events_x.shape)
              
                        test_acc, test_los, test_TP, test_FP, test_TN, test_FN, test_recall, test_precision, test_F1 = seq2seq.test(events=cle_test_events_x, labels=cle_test_windows_y)
                        
                        metrics_dic = { 
                                    'model': seq2seq.modelname,
                                    'clean test Accuracy': f'{test_acc*100:.2f}%',
                                    'clean test Loss': test_los,
                                    'clean test TP': test_TP,
                                    'clean test FP': test_FP,
                                    'clean test TN': test_TN,
                                    'clean test FN': test_FN,
                                    'clean test Recall': f'{test_recall*100:.2f}%',
                                    'clean test Precision': f'{test_precision*100:.2f}%',
                                    'clean test F1': f'{test_F1*100:.2f}%',
                                    }
                        
                        print(f"{seq2seq.modelname} metrics_dic:\n {metrics_dic}")                   
                
                
                #----------seq2seq predict test_events_x----------
                print("test_events_x.shape:",test_events_x.shape)                    
                
                seq2seq_tagged_mal_event_probs, seq2seq_tagged_mal_event_idxs, seq2seq_tagged_ben_event_probs, seq2seq_tagged_ben_event_idxs = seq2seq.analysis(test_events_x, test_windows_y)
                
                seq2seq_tagged_mal_event_num_list.append(seq2seq_tagged_mal_event_idxs.shape[0])
                seq2seq_tagged_ben_event_num_list.append(seq2seq_tagged_ben_event_idxs.shape[0])
                
                #---------------------------------------------
                #
                # retrain strategy
                #
                #---------------------------------------------  
                set_B_pos_idx = []
                set_C_neg_idx = []

                # way 1
                for idx in seq2seq_tagged_mal_event_idxs:                
                    set_B_pos_idx.append(idx)                       #   B
                    
                for idx in detector_tagged_mal_windows_idxs:            #   A
                    if idx not in seq2seq_tagged_mal_event_idxs:        
                        set_C_neg_idx.append(idx)                   #   C       
                
                            
                print(f">>>>>>>> Prepare {r+1} round Retraining clean dataset >>>>>>>>")    

              
                set_B_pos_x=[]
                set_B_pos_y=[]
                set_C_neg_x=[]
                set_C_neg_y=[]   
                
                print("args.strategy:",args.strategy)
                if args.strategy =='strategy1':     
                    for idx, l in enumerate(test_y): 
                        if idx in set_B_pos_idx:                   
                            set_B_pos_x.append(test_x[idx])
                            set_B_pos_y.append(1)
                            
                        elif idx in set_C_neg_idx:            
                            set_C_neg_x.append(test_x[idx])
                            set_C_neg_y.append(0)
                            
                    set_B_pos_x = np.array(set_B_pos_x)        
                    set_B_pos_y = np.array(set_B_pos_y)        
                    set_C_neg_x = np.array(set_C_neg_x)        
                    set_C_neg_y = np.array(set_C_neg_y)        
                            
                    print("set_B_pos_x.shape:",set_B_pos_x.shape)
                    print("set_B_pos_y.shape:",set_B_pos_y.shape)
                    print("set_C_neg_x.shape:",set_C_neg_x.shape)
                    print("set_C_neg_y.shape:",set_C_neg_y.shape)
                                    
                    if set_B_pos_x.shape[0] == 0:
                        set_BC_x = set_C_neg_x
                        set_BC_y = set_C_neg_y
                    else:                    
                        if set_C_neg_x.shape[0] == 0:
                            set_BC_x = set_B_pos_x
                            set_BC_y = set_B_pos_y                        
                        else:    
                            set_BC_x = np.concatenate((set_B_pos_x,set_C_neg_x))
                            set_BC_y = np.concatenate((set_B_pos_y,set_C_neg_y))

                        
                    print("set_BC_x.shape:",set_BC_x.shape)
                    print("set_BC_y.shape:",set_BC_y.shape)  
                        
                elif args.strategy =='strategy2': 
                    for idx, l in enumerate(test_y): 
                        if idx in set_B_pos_idx:                   
                            set_B_pos_x.append(test_x[idx])
                            set_B_pos_y.append(1)


                    set_B_pos_x = np.array(set_B_pos_x)        
                    set_B_pos_y = np.array(set_B_pos_y)        
                    print("set_B_pos_x.shape:",set_B_pos_x.shape)
                    print("set_B_pos_y.shape:",set_B_pos_y.shape)
                        
                    set_BC_x = set_B_pos_x
                    set_BC_y = set_B_pos_y
                    print("set_BC_x.shape:",set_BC_x.shape)
                    print("set_BC_y.shape:",set_BC_y.shape)  
                    
                                                
                elif args.strategy =='strategy3': 
                    for idx, l in enumerate(test_y):                       
                        if idx in set_C_neg_idx:            
                            set_C_neg_x.append(test_x[idx])
                            set_C_neg_y.append(0)
                                    
                    set_C_neg_x = np.array(set_C_neg_x)        
                    set_C_neg_y = np.array(set_C_neg_y)        
                    print("set_C_neg_x.shape:",set_C_neg_x.shape)
                    print("set_C_neg_y.shape:",set_C_neg_y.shape)
                                
                    set_BC_x = set_C_neg_x
                    set_BC_y = set_C_neg_y
                    print("set_BC_x.shape:",set_BC_x.shape)
                    print("set_BC_y.shape:",set_BC_y.shape)              

                elif args.strategy =='strategy4-groundtruth':
                    
                    set_B_pos_x = adv_testset_x 
                    set_B_pos_y = adv_testset_y
                    
                    set_B_pos_x = np.array(set_B_pos_x)        
                    set_B_pos_y = np.array(set_B_pos_y)        
                    print("set_B_pos_x.shape:",set_B_pos_x.shape)
                    print("set_B_pos_y.shape:",set_B_pos_y.shape)
                        
                    set_BC_x = set_B_pos_x
                    set_BC_y = set_B_pos_y
                    print("set_BC_x.shape:",set_BC_x.shape)
                    print("set_BC_y.shape:",set_BC_y.shape)  
                        
                elif args.strategy =='strategy5-groundtruth-C':
                    
                    set_B_pos_x = adv_testset_x 
                    set_B_pos_y = adv_testset_y
                    
                    set_B_pos_x = np.array(set_B_pos_x)        
                    set_B_pos_y = np.array(set_B_pos_y)        
                    print("set_B_pos_x.shape:",set_B_pos_x.shape)
                    print("set_B_pos_y.shape:",set_B_pos_y.shape)
        
                    for idx, l in enumerate(test_y): 
                        if idx in set_C_neg_idx:            
                            set_C_neg_x.append(test_x[idx])
                            set_C_neg_y.append(0)
                                
                    set_C_neg_x = np.array(set_C_neg_x)        
                    set_C_neg_y = np.array(set_C_neg_y)        
                            
                    print("set_C_neg_x.shape:",set_C_neg_x.shape)
                    print("set_C_neg_y.shape:",set_C_neg_y.shape)
                                    
                
                    if set_B_pos_x.shape[0] == 0:
                        set_BC_x = set_C_neg_x
                        set_BC_y = set_C_neg_y
                    else:                    
                        if set_C_neg_x.shape[0] == 0:
                            set_BC_x = set_B_pos_x
                            set_BC_y = set_B_pos_y                        
                        else:    
                            set_BC_x = np.concatenate((set_B_pos_x,set_C_neg_x))
                            set_BC_y = np.concatenate((set_B_pos_y,set_C_neg_y))
                        
                    print("set_BC_x.shape:",set_BC_x.shape)
                    print("set_BC_y.shape:",set_BC_y.shape)              

                        
                if set_BC_x.shape[0] == 0:
                    retrainset_x = detector.dataset['train'][0]
                    retrainset_y = detector.dataset['train'][1]                      
                else:        
                    retrainset_x = np.concatenate((set_BC_x, detector.dataset['train'][0]))
                    retrainset_y = np.concatenate((set_BC_y, detector.dataset['train'][1]))   
                
                print("retrainset_x.shape:",retrainset_x.shape)
                print("retrainset_y.shape:",retrainset_y.shape)            

                #---------------------------------------------
                #
                # retrain 
                #
                #---------------------------------------------  
                
                rou_cost_time=detector.retrain(retrainset_x=retrainset_x, retrainset_y=retrainset_y, timesteps=args.timesteps, curround_exp_result_dir=curround_exp_result_dir)


                #---------------------------------------------
                #
                # evaluate retrained infection detector
                #
                #---------------------------------------------         
                print(f">>>>>>>> Evaluate {r+1} round retrained {detector.modelname} on clean test data")
                test_acc, test_los, test_TP, test_FP, test_TN, test_FN, test_recall, test_precision, test_F1 = detector.test(testset_x=detector.dataset['test'][0], testset_y=detector.dataset['test'][1],timesteps=args.timesteps)
                FPrate = round((test_FP/(test_FP+test_TN)), 4)
                FNrate = round((test_FN/(test_FN+test_TP)), 4)
                
                metrics_dic = { 
                            'model': detector.modelname,
                            'clean test Accuracy': f'{test_acc*100:.2f}%',
                            'clean test Loss': test_los,
                            'clean test TP': test_TP,
                            'clean test FP': test_FP,
                            'clean test TN': test_TN,
                            'clean test FN': test_FN,
                            'clean test Recall': f'{test_recall*100:.2f}%',
                            'clean test Precision': f'{test_precision*100:.2f}%',
                            'clean test F1': f'{test_F1*100:.2f}%',
                            'clean test FPrate': f'{FPrate*100:.2f}%',
                            'clean test FNrate': f'{FNrate*100:.2f}%',    
                            }
                print(f"{detector.modelname} metrics_dic:\n {metrics_dic}")              
                
                test_FPrate_list.append(FPrate*100)
                test_FNrate_list.append(FNrate*100)
                test_acc_list.append(test_acc*100)
                test_los_list.append(test_los)
                test_TP_list.append(test_TP)
                test_FP_list.append(test_FP)
                test_TN_list.append(test_TN)
                test_FN_list.append(test_FN)
                test_recall_list.append(test_recall*100)
                test_precision_list.append(test_precision*100)
                test_F1_list.append(test_F1*100)
                cost_time_list.append(rou_cost_time)

                print(f">>>>>>>> Evaluate {r+1} round retrained {detector.modelname} on adversarial test data")
                adv_exp_result_dir = os.path.join(curround_exp_result_dir,f'advattack')
                os.makedirs(adv_exp_result_dir, exist_ok=True)
                adv_testset_x, adv_testset_y = detector.generate_advmail(timesteps=args.timesteps,cle_testset_x=detector.dataset['test'][0],cle_testset_y=detector.dataset['test'][1])
                
        

                adv_test_acc, adv_test_los, adv_test_TP, adv_test_FP, adv_test_TN, adv_test_FN, adv_test_recall, adv_test_precision, adv_test_F1 = detector.test(testset_x=adv_testset_x, testset_y=adv_testset_y, timesteps=args.timesteps)
                adv_FNrate = round((adv_test_FN/(adv_test_FN+adv_test_TP)), 4)

                adv_metrics_dic = { 
                            'model': detector.modelname,
                            'adv test Accuracy': f'{adv_test_acc*100:.2f}%',
                            'adv test Loss': adv_test_los,
                            'adv test TP': adv_test_TP,
                            'adv test FP': adv_test_FP,
                            'adv test TN': adv_test_TN,
                            'adv test FN': adv_test_FN,
                            'adv test Recall': f'{adv_test_recall*100:.2f}%',
                            'adv test Precision': f'{adv_test_precision*100:.2f}%',
                            'adv test F1': f'{adv_test_F1*100:.2f}%',
                            'adv test FNrate': f'{adv_FNrate*100:.2f}%',
                            }
                print(f"{r+1}th-round retrained {detector.modelname} adv_metrics_dic:\n {adv_metrics_dic}")    
                
                adv_test_FNrate_list.append(adv_FNrate*100)
                adv_test_acc_list.append(adv_test_acc*100)
                adv_test_los_list.append(adv_test_los)
                adv_test_TN_list.append(adv_test_TN)
                adv_test_FN_list.append(adv_test_FN)
                adv_test_recall_list.append(adv_test_recall*100)
                adv_test_precision_list.append(adv_test_precision*100)
                adv_test_F1_list.append(adv_test_F1*100)
            
            
            #---------------------------------------------
            #
            # draw figure
            #
            #---------------------------------------------

            #---------------evaluate on clean testset---------------
            retrain_cle_exp_result_dir = os.path.join(exp_result_dir,f'retrain-evaluate-cle')
            os.makedirs(retrain_cle_exp_result_dir, exist_ok=True)

            loss_png_name = f'Test loss of retrained {detector.modelname} on clean testset'
            accuracy_png_name = f'Accuracy of retrained {detector.modelname} on clean testset'
            costtime_png_name = f'Cost Time of retrain {detector.modelname}'
            fn_fp_png_name = f'FP and FN of retrained {detector.modelname} on clean testset'
            recall_png_name = f'Recall of retrained {detector.modelname} on clean testset'
            f1_png_name = f'F1 of retrained {detector.modelname} on clean testset'
            fnrate_fprate_png_name = f'FP rate and FN rate of retrained {detector.modelname} on clean testset'
            num_tagged_mal_png_name = f'Number of tagged malicious amples'
            num_tagged_ben_png_name = f'Number of tagged benign samples'
            num_tagged_png_name = f'Number of tagged malicious or benign samples'        

            plt.plot(list(range(len(test_los_list))), test_los_list, label='Test Loss', marker='o')
            plt.xlabel('Round')
            plt.ylabel('Loss on Clean Test Set')
            plt.xticks(range(min(list(range(len(test_los_list)))), max(list(range(len(test_los_list))))+1, math.ceil(len(test_los_list)/10)))
            plt.xlim(left=0)
            plt.ylim(bottom=0)
            plt.title(f'{loss_png_name}')
            plt.savefig(f'{retrain_cle_exp_result_dir}/{loss_png_name}.png')
            plt.close()
                    
            plt.plot(list(range(len(test_acc_list))), test_acc_list, label='Test Accuracy', marker='o')
            plt.xlabel('Round')
            plt.ylabel('Accuracy (%) on Clean Test Set')
            plt.xticks(range(min(list(range(len(test_acc_list)))), max(list(range(len(test_acc_list))))+1, math.ceil(len(test_acc_list)/10)))
            plt.ylim(0, 100)
            plt.xlim(left=0)
            plt.ylim(bottom=0)
            plt.title(f'{accuracy_png_name}')        
            plt.savefig(f'{retrain_cle_exp_result_dir}/{accuracy_png_name}.png')
            plt.close()

            plt.plot(list(range(len(cost_time_list))), cost_time_list, label='Cost Time', marker='o')
            plt.xlabel('Round')
            plt.ylabel('Cost Time (second)')
            plt.xticks(range(min(list(range(len(cost_time_list)))), max(list(range(len(cost_time_list))))+1, math.ceil(len(cost_time_list)/10)))       
            plt.xlim(left=0)
            plt.ylim(bottom=0) 
            plt.title(f'{costtime_png_name}')        
            plt.savefig(f'{retrain_cle_exp_result_dir}/{costtime_png_name}.png')
            plt.close()

            plt.plot(list(range(len(test_FP_list))), test_FP_list, label='Test FP', marker='o')
            plt.plot(list(range(len(test_FN_list))), test_FN_list, label='Test FN', marker='s')
            plt.xlabel('Round')
            plt.ylabel('FP and FN on Clean Test Set')
            plt.xticks(range(min(list(range(len(test_FP_list)))), max(list(range(len(test_FP_list))))+1, math.ceil(len(test_FP_list)/10)))        
            plt.xlim(left=0)
            plt.ylim(bottom=0)
            plt.legend(loc='best',frameon=True)
            plt.title(f'{fn_fp_png_name}')        
            plt.savefig(f'{retrain_cle_exp_result_dir}/{fn_fp_png_name}.png')
            plt.close()

            plt.plot(list(range(len(test_FPrate_list))), test_FPrate_list, label='Test FP rate', marker='o')
            plt.plot(list(range(len(test_FNrate_list))), test_FNrate_list, label='Test FN rate', marker='s')
            plt.xlabel('Round')
            plt.ylabel('FP Rate and FN Rate (%) on Clean Test Set')
            plt.ylim(0, 100)
            plt.xticks(range(min(list(range(len(test_FPrate_list)))), max(list(range(len(test_FPrate_list))))+1, math.ceil(len(test_FPrate_list)/10))  )   
            plt.xlim(left=0)
            plt.ylim(bottom=0)   
            plt.legend(loc='best',frameon=True)
            plt.title(f'{fnrate_fprate_png_name}')        
            plt.savefig(f'{retrain_cle_exp_result_dir}/{fnrate_fprate_png_name}.png')
            plt.close()
            
            
            
            plt.plot(list(range(len(test_recall_list))), test_recall_list, label='Test Recall', marker='o')
            plt.xlabel('Round')
            plt.ylabel('Recall (%) on Test Set')
            plt.ylim(0, 100)
            plt.xticks(range(min(list(range(len(test_recall_list)))), max(list(range(len(test_recall_list))))+1, math.ceil(len(test_recall_list)/10))  )   
            plt.xlim(left=0)
            plt.ylim(bottom=0)   
            plt.title(f'{recall_png_name}')        
            plt.savefig(f'{retrain_cle_exp_result_dir}/{recall_png_name}.png')
            plt.close()        

            plt.plot(list(range(len(test_F1_list))), test_F1_list, label='Test F1', marker='o')
            plt.xlabel('Round')
            plt.ylabel('F1 (%) on Test Set')
            plt.ylim(0, 100)
            plt.xticks(range(min(list(range(len(test_F1_list)))), max(list(range(len(test_F1_list))))+1, math.ceil(len(test_F1_list)/10))   ) 
            plt.xlim(left=0)
            plt.ylim(bottom=0)            
            plt.title(f'{f1_png_name}')        
            plt.savefig(f'{retrain_cle_exp_result_dir}/{f1_png_name}.png')
            plt.close()                                 



            #---------------evaluate on adv testset--------------------------------
            retrain_adv_exp_result_dir = os.path.join(exp_result_dir,f'retrain-evaluate-adv')
            os.makedirs(retrain_adv_exp_result_dir, exist_ok=True)

            adv_loss_png_name = f'Test loss of retrained {detector.modelname} on {args.attack} examples'
            adv_accuracy_png_name = f'Accuracy of retrained {detector.modelname} on {args.attack} examples'
            adv_fn_png_name = f'FN of retrained {detector.modelname} on {args.attack} examples'
            adv_recall_png_name = f'Recall of retrained {detector.modelname} on {args.attack} examples'
            adv_f1_png_name = f'F1 of retrained {detector.modelname} on {args.attack} examples'
            adv_fnrate_png_name = f'FN rate of retrained {detector.modelname} on {args.attack} examples'
            adv_precision_png_name = f'Precision of retrained {detector.modelname} on {args.attack} examples'
            

            plt.plot(list(range(len(adv_test_los_list))), adv_test_los_list, label='Test Loss', marker='o')
            plt.xlabel('Round')
            plt.ylabel('Loss on Adversarial Test Set')
            plt.xticks(range(min(list(range(len(adv_test_los_list)))), max(list(range(len(adv_test_los_list))))+1, math.ceil(len(adv_test_los_list)/10))   )  
            plt.xlim(left=0)
            plt.ylim(bottom=0)           
            plt.title(f'{adv_loss_png_name}')
            plt.savefig(f'{retrain_adv_exp_result_dir}/{adv_loss_png_name}.png')
            plt.close()
                    
            plt.plot(list(range(len(adv_test_acc_list))), adv_test_acc_list, label='Test Accuracy', marker='o')
            plt.xlabel('Round')
            plt.ylabel('Accuracy (%) on Adversarial Test Set')
            plt.ylim(0, 100)
            plt.xticks(range(min(list(range(len(adv_test_acc_list)))), max(list(range(len(adv_test_acc_list))))+1, math.ceil(len(adv_test_acc_list)/10))     )
            plt.xlim(left=0)
            plt.ylim(bottom=0)                     
            plt.title(f'{adv_accuracy_png_name}')        
            plt.savefig(f'{retrain_adv_exp_result_dir}/{adv_accuracy_png_name}.png')
            plt.close()

            plt.plot(list(range(len(adv_test_FN_list))), adv_test_FN_list, label='Test False Negative', marker='o')
            plt.xlabel('Round')
            plt.ylabel('FN on Adversarial Test Set')
            plt.xticks(range(min(list(range(len(adv_test_FN_list)))), max(list(range(len(adv_test_FN_list))))+1, math.ceil(len(adv_test_FN_list)/10)))                
            plt.xlim(left=0)
            plt.ylim(bottom=0)  
            plt.title(f'{adv_fn_png_name}')        
            plt.savefig(f'{retrain_adv_exp_result_dir}/{adv_fn_png_name}.png')
            plt.close()

            plt.plot(list(range(len(adv_test_FNrate_list))), adv_test_FNrate_list, label='Test False Negative rate', marker='o')
            plt.xlabel('Round')
            plt.ylabel('FN Rate (%) on Adversarial Test Set')
            plt.ylim(0, 100)
            plt.xticks(range(min(list(range(len(adv_test_FNrate_list)))), max(list(range(len(adv_test_FNrate_list))))+1, math.ceil(len(adv_test_FNrate_list)/10)) )      
            plt.xlim(left=0)
            plt.ylim(bottom=0)                   
            plt.title(f'{adv_fnrate_png_name}')        
            plt.savefig(f'{retrain_adv_exp_result_dir}/{adv_fnrate_png_name}.png')
            plt.close()
            
            
            plt.plot(list(range(len(adv_test_recall_list))), adv_test_recall_list, label='Test Recall', marker='o')
            plt.xlabel('Round')
            plt.ylabel('Recall (%) on Adversarial Test Set')
            plt.ylim(0, 100)
            plt.xticks(range(min(list(range(len(adv_test_recall_list)))), max(list(range(len(adv_test_recall_list))))+1, math.ceil(len(adv_test_recall_list)/10))    )
            plt.xlim(left=0)
            plt.ylim(bottom=0)                      
            plt.title(f'{adv_recall_png_name}')        
            plt.savefig(f'{retrain_adv_exp_result_dir}/{adv_recall_png_name}.png')
            plt.close()        

            plt.plot(list(range(len(adv_test_precision_list))), adv_test_precision_list, label='Test Precision', marker='o')
            plt.xlabel('Round')
            plt.ylabel('Precision (%) on Adversarial Test Set')
            plt.ylim(0, 100)
            plt.xticks(range(min(list(range(len(adv_test_precision_list)))), max(list(range(len(adv_test_precision_list))))+1, math.ceil(len(adv_test_precision_list)/10))    )
            plt.xlim(left=0)
            plt.ylim(bottom=0)                      
            plt.title(f'{adv_precision_png_name}')        
            plt.savefig(f'{retrain_adv_exp_result_dir}/{adv_precision_png_name}.png')
            plt.close()      
            
            plt.plot(list(range(len(adv_test_F1_list))), adv_test_F1_list, label='Test F1', marker='o')
            plt.xlabel('Round')
            plt.ylabel('F1 (%) on Adversarial Test Set')
            plt.ylim(0, 100)
            plt.xticks(range(min(list(range(len(adv_test_F1_list)))), max(list(range(len(adv_test_F1_list))))+1, math.ceil(len(adv_test_F1_list)/10))      )                        
            plt.xlim(left=0)
            plt.ylim(bottom=0)            
            plt.title(f'{adv_f1_png_name}')        
            plt.savefig(f'{retrain_adv_exp_result_dir}/{adv_f1_png_name}.png')
            plt.close()                                 
    

            #-------------------tagged num---------------------
            plt.plot(list(range(1, len(seq2seq_tagged_mal_event_num_list)+1)), seq2seq_tagged_mal_event_num_list, label='tagged by seq2seq', marker='o')
            plt.plot(list(range(1, len(detector_tagged_mal_event_num_list)+1)), detector_tagged_mal_event_num_list, label='tagged by detector', marker='*')
                            
            plt.xlabel('Round')
            plt.ylabel('Number of Tagged Malicious Events')
            plt.xticks(range(1, len(seq2seq_tagged_mal_event_num_list)+1, math.ceil(len(seq2seq_tagged_mal_event_num_list)/10))  )   

            plt.legend(loc='best',frameon=True)
            plt.title(f'{num_tagged_mal_png_name}')        
            plt.savefig(f'{retrain_cle_exp_result_dir}/{num_tagged_mal_png_name}.png')
            plt.close()
                    
                    
                    
                    
            plt.plot(list(range(1, len(seq2seq_tagged_ben_event_num_list)+1)), seq2seq_tagged_ben_event_num_list, label='tagged by seq2seq ', marker='s')
            plt.plot(list(range(1, len(detector_tagged_ben_event_num_list)+1)), detector_tagged_ben_event_num_list, label='tagged by detector ', marker='^')
                            
            plt.xlabel('Round')
            plt.ylabel('Number of Tagged Benign Events')
            plt.xticks(range(1, len(seq2seq_tagged_ben_event_num_list)+1, math.ceil(len(seq2seq_tagged_ben_event_num_list)/10))  )   

            plt.legend(loc='best',frameon=True)
            plt.title(f'{num_tagged_ben_png_name}')        
            plt.savefig(f'{retrain_cle_exp_result_dir}/{num_tagged_ben_png_name}.png')
            plt.close()  
            
            
            
            
            plt.plot(list(range(1, len(seq2seq_tagged_mal_event_num_list)+1)), seq2seq_tagged_mal_event_num_list, label='tagged malicious by seq2seq', marker='o')
            plt.plot(list(range(1, len(seq2seq_tagged_ben_event_num_list)+1)), seq2seq_tagged_ben_event_num_list, label='tagged benign by seq2seq ', marker='s')        
            plt.plot(list(range(1, len(detector_tagged_mal_event_num_list)+1)), detector_tagged_mal_event_num_list, label='tagged malicious by detector', marker='*')        
            plt.plot(list(range(1, len(detector_tagged_ben_event_num_list)+1)), detector_tagged_ben_event_num_list, label='tagged benign by detector ', marker='^')

            plt.xlabel('Round')
            plt.ylabel('Number of Tagged Malicious/Benign Events')
            plt.xticks(range(1, len(seq2seq_tagged_ben_event_num_list)+1, math.ceil(len(seq2seq_tagged_ben_event_num_list)/10))  )   

            plt.legend(loc='best',frameon=True)
            plt.title(f'{num_tagged_png_name}')        
            plt.savefig(f'{retrain_cle_exp_result_dir}/{num_tagged_png_name}.png')
            plt.close()  
            



                    
            #---------------save xlsx data---------------

            cle_figure_xlsx_result_dir = os.path.join(retrain_cle_exp_result_dir,f'figure-xlsx')
            os.makedirs(cle_figure_xlsx_result_dir, exist_ok=True)        
            pd.DataFrame(test_FPrate_list).to_excel(f'{cle_figure_xlsx_result_dir}/test_FPrate_list.xlsx')
            pd.DataFrame(test_FNrate_list).to_excel(f'{cle_figure_xlsx_result_dir}/test_FNrate_list.xlsx')
            pd.DataFrame(test_acc_list).to_excel(f'{cle_figure_xlsx_result_dir}/test_acc_list.xlsx')
            pd.DataFrame(test_los_list).to_excel(f'{cle_figure_xlsx_result_dir}/test_los_list.xlsx')
            pd.DataFrame(test_TP_list).to_excel(f'{cle_figure_xlsx_result_dir}/test_TP_list.xlsx')
            pd.DataFrame(test_FP_list).to_excel(f'{cle_figure_xlsx_result_dir}/test_FP_list.xlsx')
            pd.DataFrame(test_TN_list).to_excel(f'{cle_figure_xlsx_result_dir}/test_TN_list.xlsx')
            pd.DataFrame(test_FN_list).to_excel(f'{cle_figure_xlsx_result_dir}/test_FN_list.xlsx')
            pd.DataFrame(test_recall_list).to_excel(f'{cle_figure_xlsx_result_dir}/test_recall_list.xlsx')
            pd.DataFrame(test_precision_list).to_excel(f'{cle_figure_xlsx_result_dir}/test_precision_list.xlsx')
            pd.DataFrame(test_F1_list).to_excel(f'{cle_figure_xlsx_result_dir}/test_F1_list.xlsx')
            pd.DataFrame(cost_time_list).to_excel(f'{cle_figure_xlsx_result_dir}/cost_time_list.xlsx')                
            pd.DataFrame(seq2seq_tagged_mal_event_num_list).to_excel(f'{cle_figure_xlsx_result_dir}/seq2seq_tagged_mal_event_num_list.xlsx')
            pd.DataFrame(seq2seq_tagged_ben_event_num_list).to_excel(f'{cle_figure_xlsx_result_dir}/seq2seq_tagged_ben_event_num_list.xlsx')      
            
            
                

            adv_figure_xlsx_result_dir = os.path.join(retrain_adv_exp_result_dir,f'figure-xlsx')
            os.makedirs(adv_figure_xlsx_result_dir, exist_ok=True)          
            pd.DataFrame(adv_test_FNrate_list).to_excel(f'{adv_figure_xlsx_result_dir}/adv_test_FNrate_list.xlsx')
            pd.DataFrame(adv_test_acc_list).to_excel(f'{adv_figure_xlsx_result_dir}/adv_test_acc_list.xlsx')
            pd.DataFrame(adv_test_los_list).to_excel(f'{adv_figure_xlsx_result_dir}/adv_test_los_list.xlsx')
            pd.DataFrame(adv_test_TN_list).to_excel(f'{adv_figure_xlsx_result_dir}/adv_test_TN_list.xlsx')
            pd.DataFrame(adv_test_FN_list).to_excel(f'{adv_figure_xlsx_result_dir}/adv_test_FN_list.xlsx')
            pd.DataFrame(adv_test_recall_list).to_excel(f'{adv_figure_xlsx_result_dir}/adv_test_recall_list.xlsx')
            pd.DataFrame(adv_test_precision_list).to_excel(f'{adv_figure_xlsx_result_dir}/adv_test_precision_list.xlsx')
            pd.DataFrame(adv_test_F1_list).to_excel(f'{adv_figure_xlsx_result_dir}/adv_test_F1_list.xlsx')
            
                        
                


        elif args.advset_mode == 'advset2':
          
            #---------------------------------------------
            #
            # extract mal samples form original nfection test set
            #
            #--------------------------------------------- 
            
            cle_testset_x=detector.dataset['test'][0]
            cle_testset_y=detector.dataset['test'][1]
            print(f'Original clean test set lengh: {len(cle_testset_y)}') # 318


            # extract malicious set
            condition = cle_testset_y.astype(bool)     
            malicious_cle_testset_y = np.extract(condition,cle_testset_y)
            print("malicious_cle_testset_y.shape:",malicious_cle_testset_y.shape)

            cond=np.expand_dims(condition,1)
            cond_expend = np.full((cle_testset_x.shape[0], cle_testset_x.shape[1]), False, dtype=bool)
            cond = np.logical_or(cond_expend, cond)        
            malicious_cle_testset_x = np.extract(cond,cle_testset_x)
            malicious_cle_testset_x = np.reshape(malicious_cle_testset_x, (malicious_cle_testset_y.shape[0], cle_testset_x.shape[1]))
      
                    
            # extract benign set            
            benign_cle_testset_y = np.extract(1-condition,cle_testset_y)
            print("benign_cle_testset_y.shape:",benign_cle_testset_y.shape)
           
            cond=np.expand_dims(1-condition,1)
            cond_expend = np.full((cle_testset_x.shape[0], cle_testset_x.shape[1]), False, dtype=bool)
            cond = np.logical_or(cond_expend, cond)       
            benign_cle_testset_x = np.extract(cond,cle_testset_x)
            benign_cle_testset_x = np.reshape(benign_cle_testset_x, (benign_cle_testset_y.shape[0], cle_testset_x.shape[1]))
      
            #---------------------------------------------
            #
            # split infection test set into N+1 parts
            #
            #---------------------------------------------   
            
            # split only mal infection test set into N+1 parts
            sub_mal_cle_testset_size = len(malicious_cle_testset_y) // (args.relabel_rounds+1)
            sub_mal_cle_testsets_x=[]
            sub_mal_cle_testsets_y=[]
            for i in range(args.relabel_rounds+1):
                subdataset_x = malicious_cle_testset_x[i * sub_mal_cle_testset_size: (i + 1) * sub_mal_cle_testset_size]
                subdataset_y = malicious_cle_testset_y[i * sub_mal_cle_testset_size: (i + 1) * sub_mal_cle_testset_size]
                
                sub_mal_cle_testsets_x.append(subdataset_x)
                sub_mal_cle_testsets_y.append(subdataset_y)
                print(f"Number of mal samples in {i+1} sub_mal_cle_testset - : {len(sub_mal_cle_testsets_x[i])}")
         

            # split only ben infection test set into N+1 parts
            sub_ben_cle_testset_size = len(benign_cle_testset_y) // (args.relabel_rounds+1)
            sub_ben_cle_testsets_x=[]
            sub_ben_cle_testsets_y=[]
            for i in range(args.relabel_rounds+1):
                subdataset_x = benign_cle_testset_x[i * sub_ben_cle_testset_size: (i + 1) * sub_ben_cle_testset_size]
                subdataset_y = benign_cle_testset_y[i * sub_ben_cle_testset_size: (i + 1) * sub_ben_cle_testset_size]
                
                sub_ben_cle_testsets_x.append(subdataset_x)
                sub_ben_cle_testsets_y.append(subdataset_y)
                print(f"Number of ben samples in {i+1} sub_ben_cle_testset - : {len(sub_ben_cle_testsets_x[i])}")
            
            
            sub_cle_testsets_x=[]
            sub_cle_testsets_y=[]
            for i in range(args.relabel_rounds+1):
                subdataset_x = np.concatenate((sub_mal_cle_testsets_x[i],sub_ben_cle_testsets_x[i]))
                subdataset_y = np.concatenate((sub_mal_cle_testsets_y[i],sub_ben_cle_testsets_y[i]))
                subdataset_x, subdataset_y = shuffle(subdataset_x, subdataset_y)
                
                sub_cle_testsets_x.append(subdataset_x)
                sub_cle_testsets_y.append(subdataset_y)
                                
                print(f"Number of samples in {i+1} sub_cle_testset - : {len(sub_cle_testsets_x[i])}")
             
         

            #---------------------------------------------
            #
            # prepare for plt
            #
            #---------------------------------------------         

            
                        
            test_acc_on_former_set_list = [] 
            test_los_on_former_set_list = []
            test_TP_on_former_set_list = []
            test_FP_on_former_set_list = []
            test_TN_on_former_set_list = []
            test_FN_on_former_set_list = []
            test_recall_on_former_set_list = []
            test_precision_on_former_set_list = []
            test_FPR_on_former_set_list = []
            test_F1_on_former_set_list = []  
            test_FPrate_on_former_set_list=[]
            test_FNrate_on_former_set_list=[]
            
                                  
            test_acc_on_later_set_list = [] 
            test_los_on_later_set_list = []
            test_TP_on_later_set_list = []
            test_FP_on_later_set_list = []
            test_TN_on_later_set_list = []
            test_FN_on_later_set_list = []
            test_recall_on_later_set_list = []
            test_precision_on_later_set_list = []
            test_FPR_on_later_set_list = []
            test_F1_on_later_set_list = []   
            test_FPrate_on_later_set_list=[]
            test_FNrate_on_later_set_list=[]
            
            cost_time_list =[]
            


            adv_test_acc_list = [] 
            adv_test_los_list = []
            adv_test_TN_list = []
            adv_test_FN_list = []
            adv_test_recall_list = []
            adv_test_precision_list = []
            adv_test_F1_list = []
            adv_test_FNrate_list=[]
            
            detector_tagged_mal_event_num_list=[]
            detector_tagged_ben_event_num_list=[]
            seq2seq_tagged_mal_event_num_list=[]
            seq2seq_tagged_ben_event_num_list=[]
            
            
            #---------------------------------------------
            #
            # evaluate vanillia infection detector on subtestset1
            #
            #---------------------------------------------    
            print(f">>>>>>>> Evaluate vanillia {detector.modelname} on 1st sub clean test data (former)>>>>>>>>")
            
            test_acc, test_los, test_TP, test_FP, test_TN, test_FN, test_recall, test_precision, test_F1 = detector.test(testset_x=sub_cle_testsets_x[0], testset_y=sub_cle_testsets_y[0], timesteps=args.timesteps)
            
            FPrate = round((test_FP/(test_FP+test_TN)), 4)
            FNrate = round((test_FN/(test_FN+test_TP)), 4)
            
            metrics_dic = { 
                        'model': detector.modelname,
                        'clean test Accuracy': f'{test_acc*100:.2f}%',
                        'clean test Loss': test_los,
                        'clean test TP': test_TP,
                        'clean test FP': test_FP,
                        'clean test TN': test_TN,
                        'clean test FN': test_FN,
                        'clean test Recall': f'{test_recall*100:.2f}%',
                        'clean test Precision': f'{test_precision*100:.2f}%',
                        'clean test F1': f'{test_F1*100:.2f}%',
                        'clean test FPrate':f'{FPrate*100:.2f}%',
                        'clean test FNrate':f'{FNrate*100:.2f}%',
                        }
            print(f"vanillia {detector.modelname} on 1st clean sub testset metrics_dic:\n {metrics_dic}")          
            
            test_FPrate_on_former_set_list.append(FPrate*100)
            test_FNrate_on_former_set_list.append(FNrate*100)
            
            test_acc_on_former_set_list.append(test_acc*100)
            test_los_on_former_set_list.append(test_los)
            test_TP_on_former_set_list.append(test_TP)
            test_FP_on_former_set_list.append(test_FP)
            test_TN_on_former_set_list.append(test_TN)
            test_FN_on_former_set_list.append(test_FN)
            test_recall_on_former_set_list.append(test_recall*100)
            test_precision_on_former_set_list.append(test_precision*100)
            test_F1_on_former_set_list.append(test_F1*100)
             
            cost_time_list.append(0)        
            #------------------------------------
            
            print(f">>>>>>>> Evaluate vanillia {detector.modelname} on 2nd sub clean test data (later)>>>>>>>>")
            
            test_acc, test_los, test_TP, test_FP, test_TN, test_FN, test_recall, test_precision, test_F1 = detector.test(testset_x=sub_cle_testsets_x[1], testset_y=sub_cle_testsets_y[1], timesteps=args.timesteps)
            
            FPrate = round((test_FP/(test_FP+test_TN)), 4)
            FNrate = round((test_FN/(test_FN+test_TP)), 4)
            
            metrics_dic = { 
                        'model': detector.modelname,
                        'clean test Accuracy': f'{test_acc*100:.2f}%',
                        'clean test Loss': test_los,
                        'clean test TP': test_TP,
                        'clean test FP': test_FP,
                        'clean test TN': test_TN,
                        'clean test FN': test_FN,
                        'clean test Recall': f'{test_recall*100:.2f}%',
                        'clean test Precision': f'{test_precision*100:.2f}%',
                        'clean test F1': f'{test_F1*100:.2f}%',
                        'clean test FPrate':f'{FPrate*100:.2f}%',
                        'clean test FNrate':f'{FNrate*100:.2f}%',
                        }
            print(f"vanillia {detector.modelname} on 2nd clean sub testset metrics_dic:\n {metrics_dic}")          
            
            test_FPrate_on_later_set_list.append(FPrate*100)
            test_FNrate_on_later_set_list.append(FNrate*100)
            
            test_acc_on_later_set_list.append(test_acc*100)
            test_los_on_later_set_list.append(test_los)
            test_TP_on_later_set_list.append(test_TP)
            test_FP_on_later_set_list.append(test_FP)
            test_TN_on_later_set_list.append(test_TN)
            test_FN_on_later_set_list.append(test_FN)
            test_recall_on_later_set_list.append(test_recall*100)
            test_precision_on_later_set_list.append(test_precision*100)
            test_F1_on_later_set_list.append(test_F1*100)            
            #------------------------------------
            
            
            
                
            print(f">>>>>>>> Evaluate vanillia {detector.modelname} on 1st adversarial test data >>>>>>>>")
            adv_exp_result_dir = os.path.join(exp_result_dir,f'advattack')
            os.makedirs(adv_exp_result_dir, exist_ok=True)
            adv_testset_x, adv_testset_y = detector.generate_advmail(timesteps=args.timesteps,cle_testset_x=sub_cle_testsets_x[0],cle_testset_y=sub_cle_testsets_y[0])
            
            print("adv_testset_x.shape:",adv_testset_x.shape)    
            print("adv_testset_y.shape:",adv_testset_y.shape)   
            
            adv_test_acc, adv_test_los, adv_test_TP, adv_test_FP, adv_test_TN, adv_test_FN, adv_test_recall, adv_test_precision, adv_test_F1 = detector.test(testset_x=adv_testset_x, testset_y=adv_testset_y, timesteps=args.timesteps)
            adv_FNrate = round((adv_test_FN/(adv_test_FN+adv_test_TP)), 4)
            
            adv_metrics_dic = { 
                        'model': detector.modelname,
                        'adv test Accuracy': f'{adv_test_acc*100:.2f}%',
                        'adv test Loss': adv_test_los,
                        'adv test TP': adv_test_TP,
                        'adv test FP': adv_test_FP,
                        'adv test TN': adv_test_TN,
                        'adv test FN': adv_test_FN,
                        'adv test Recall': f'{adv_test_recall*100:.2f}%',
                        'adv test Precision': f'{adv_test_precision*100:.2f}%',
                        'adv test F1': f'{adv_test_F1*100:.2f}%',
                        'adv test FNrate': f'{adv_FNrate*100:.2f}%',  
                        }
            print(f"Vanillia {detector.modelname} adv_metrics_dic:\n {adv_metrics_dic}")          
            
            adv_test_FNrate_list.append(adv_FNrate*100)
            adv_test_acc_list.append(adv_test_acc*100)
            adv_test_los_list.append(adv_test_los)
            adv_test_TN_list.append(adv_test_TN)
            adv_test_FN_list.append(adv_test_FN)
            adv_test_recall_list.append(adv_test_recall*100)
            adv_test_precision_list.append(adv_test_precision*100)
            adv_test_F1_list.append(adv_test_F1*100)








            #---------------------------------------------
            #
            # multi rounds retraining
            #
            #---------------------------------------------                    
            for r in range(args.relabel_rounds):
                
                curround_exp_result_dir = os.path.join(exp_result_dir,f'round-{r+1}')
                os.makedirs(curround_exp_result_dir, exist_ok=True)

                print(f">>>>>>>> create {r+1} round retraining dataset >>>>>>>>")    

                #---------------------------------------------
                #
                # create event testset
                #
                #---------------------------------------------               
                
                #----------create clean windows testset----------
                cle_test_x = sub_cle_testsets_x[r]
                cle_test_y = sub_cle_testsets_y[r]
                
                print("cle_test_x.shape:",cle_test_x.shape)
                print("cle_test_y.shape:",cle_test_y.shape)
         

                #----------create adv windows testset----------         
                adv_exp_result_dir = os.path.join(curround_exp_result_dir,f'advattack')
                os.makedirs(adv_exp_result_dir, exist_ok=True)
                    
                adv_testset_x, adv_testset_y = detector.generate_advmail(timesteps=args.timesteps,cle_testset_x=sub_cle_testsets_x[r], cle_testset_y=sub_cle_testsets_y[r])  
                
                
                print("adv_testset_x.shape:",adv_testset_x.shape)    
                print("adv_testset_y.shape:",adv_testset_y.shape)    
                    

                if args.retrain_testset_mode == 'cle_adv':
                    #----------create adv+clean windows testset----------
                    test_x = np.concatenate((adv_testset_x, cle_test_x))
                    test_y = np.concatenate((adv_testset_y, cle_test_y))            
                    print("retrain_test_(cle_adv)_x.shape:", test_x.shape)
                    print("retrain_test_(cle_adv)_y.shape:", test_y.shape)                 
                
                elif args.retrain_testset_mode == 'adv':
                    test_x = adv_testset_x
                    test_y = adv_testset_y   
                    print("retrain_test_(adv)_x.shape:",test_x.shape)
                    print("retrain_test_(adv)_y.shape:",test_y.shape)   
             
                                    
                elif args.retrain_testset_mode == 'cle':
                    test_x = cle_test_x
                    test_y = cle_test_y 
                    print("retrain_test_(cle)_x.shape:",test_x.shape)
                    print("retrain_test_(cle)_y.shape:",test_y.shape)   
                
                                            
                                    
                #----------create adv+clean events testset----------
                test_windows_x = copy.deepcopy(test_x)
                test_windows_y = copy.deepcopy(test_y)           
                print("test_windows_x.shape:",test_windows_x.shape)
                print("test_windows_y.shape:",test_windows_y.shape) 
         
                                
                test_windows_x = test_windows_x.reshape((test_windows_x.shape[0], args.timesteps, int(math.ceil(test_windows_x.shape[1] / args.timesteps))))
                
                print("test_windows_x.shape:",test_windows_x.shape)
          
                
                test_events_x = get_events_from_windows(reconnaissance_detector, infection_detector, attack_detector, test_windows_x)
                print("test_events_x.shape:",test_events_x.shape)
            




                #---------------------------------------------
                #
                # analyze event testset
                #
                #---------------------------------------------  

                #----------detector predict test_windows_x----------
                print("test_windows_x.shape:",test_windows_x.shape)
     
                
                detector_tagged_mal_windows_probs, detector_tagged_mal_windows_idxs, detector_tagged_ben_windows_probs,detector_tagged_ben_windows_idxs = detector.analysis(test_windows_x)
                
                detector_tagged_mal_event_num_list.append(detector_tagged_mal_windows_idxs.shape[0])
                detector_tagged_ben_event_num_list.append(detector_tagged_ben_windows_idxs.shape[0])
                


                # #----------update seq2seq-------------------------
                if args.retrain_seq2seq is True: 
                    if r>=1: 
                        # create trainset
                        cle_train_windows_x = seq2seq.dataset['train'][0]
                        cle_train_windows_y = seq2seq.dataset['train'][1]
                        
                      
                        cle_train_windows_x = cle_train_windows_x.reshape((cle_train_windows_x.shape[0], args.timesteps, int(math.ceil(cle_train_windows_x.shape[1] / args.timesteps))))
                        print("cle_train_windows_x.shape:",cle_train_windows_x.shape)
                    
                        

                        seq2seq_train_events = get_events_from_windows(reconnaissance_detector, infection_detector, attack_detector, cle_train_windows_x)
                        print("seq2seq_train_events.shape:",seq2seq_train_events.shape)
                 
                        seq2seq.def_model(input_length=args.sequence_length, output_length =args.sequence_length)
                        
                        retrain_exp_result_dir = os.path.join(curround_exp_result_dir,f'retrain-seq2seq')
                        os.makedirs(retrain_exp_result_dir, exist_ok=True)
                        
                        print(f">>>>>>>> Retrain {seq2seq.modelname} >>>>>>>>")      
                        seq2seq.retrain(events=seq2seq_train_events, labels=cle_train_windows_y, exp_result_dir=retrain_exp_result_dir)
                            
                    
                        print(f">>>>>>>> Evaluate {seq2seq.modelname} on clean test data")
                        
                        # create testset
                        cle_test_windows_x = seq2seq.dataset['test'][0]
                        cle_test_windows_y = seq2seq.dataset['test'][1]
                  
                        cle_test_windows_x = cle_test_windows_x.reshape((cle_test_windows_x.shape[0], args.timesteps, int(math.ceil(cle_test_windows_x.shape[1] / args.timesteps))))
                        print("cle_test_windows_x.shape:",cle_test_windows_x.shape)
                     
                        
                        cle_test_events_x = get_events_from_windows(reconnaissance_detector, infection_detector, attack_detector, cle_test_windows_x)
                        print("cle_test_events_x.shape:",cle_test_events_x.shape)
                     
                        test_acc, test_los, test_TP, test_FP, test_TN, test_FN, test_recall, test_precision, test_F1 = seq2seq.test(events=cle_test_events_x, labels=cle_test_windows_y)
                        
                        metrics_dic = { 
                                    'model': seq2seq.modelname,
                                    'clean test Accuracy': f'{test_acc*100:.2f}%',
                                    'clean test Loss': test_los,
                                    'clean test TP': test_TP,
                                    'clean test FP': test_FP,
                                    'clean test TN': test_TN,
                                    'clean test FN': test_FN,
                                    'clean test Recall': f'{test_recall*100:.2f}%',
                                    'clean test Precision': f'{test_precision*100:.2f}%',
                                    'clean test F1': f'{test_F1*100:.2f}%',
                                    }
                        
                        print(f"{seq2seq.modelname} metrics_dic:\n {metrics_dic}")                   
                

                #----------seq2seq predict test_events_x----------
                print("test_events_x.shape:",test_events_x.shape)            
                      
                
                seq2seq_tagged_mal_event_probs, seq2seq_tagged_mal_event_idxs, seq2seq_tagged_ben_event_probs, seq2seq_tagged_ben_event_idxs = seq2seq.analysis(test_events_x, test_windows_y) 
                
                seq2seq_tagged_mal_event_num_list.append(seq2seq_tagged_mal_event_idxs.shape[0])
                seq2seq_tagged_ben_event_num_list.append(seq2seq_tagged_ben_event_idxs.shape[0])
                

                #---------------------------------------------
                #
                # retrain strategy
                #
                #---------------------------------------------  
                set_B_pos_idx = []
                set_C_neg_idx = []

                for idx in seq2seq_tagged_mal_event_idxs:                
                    set_B_pos_idx.append(idx)                       #   B
                    
                for idx in detector_tagged_mal_windows_idxs:            #   A
                    if idx not in seq2seq_tagged_mal_event_idxs:        
                        set_C_neg_idx.append(idx)                   #   C       
                

                            
                print(f">>>>>>>> Prepare {r+1} round Retraining clean dataset >>>>>>>>")    

                set_B_pos_x=[]
                set_B_pos_y=[]
                set_C_neg_x=[]
                set_C_neg_y=[]   
                
                print("args.strategy:",args.strategy)
                if args.strategy =='strategy1':     
                    for idx, l in enumerate(test_y): 
                        if idx in set_B_pos_idx:                   
                            set_B_pos_x.append(test_x[idx])
                            set_B_pos_y.append(1)
                            
                        elif idx in set_C_neg_idx:            
                            set_C_neg_x.append(test_x[idx])
                            set_C_neg_y.append(0)
                            
                    set_B_pos_x = np.array(set_B_pos_x)        
                    set_B_pos_y = np.array(set_B_pos_y)        
                    set_C_neg_x = np.array(set_C_neg_x)        
                    set_C_neg_y = np.array(set_C_neg_y)        
                            
                    print("set_B_pos_x.shape:",set_B_pos_x.shape)
                    print("set_B_pos_y.shape:",set_B_pos_y.shape)
                    print("set_C_neg_x.shape:",set_C_neg_x.shape)
                    print("set_C_neg_y.shape:",set_C_neg_y.shape)
                                    
                    if set_B_pos_x.shape[0] == 0:
                        set_BC_x = set_C_neg_x
                        set_BC_y = set_C_neg_y
                    else:                    
                        if set_C_neg_x.shape[0] == 0:
                            set_BC_x = set_B_pos_x
                            set_BC_y = set_B_pos_y                        
                        else:    
                            set_BC_x = np.concatenate((set_B_pos_x,set_C_neg_x))
                            set_BC_y = np.concatenate((set_B_pos_y,set_C_neg_y))

                        
                    print("set_BC_x.shape:",set_BC_x.shape)
                    print("set_BC_y.shape:",set_BC_y.shape)  
                        
                elif args.strategy =='strategy2': 
                    for idx, l in enumerate(test_y): 
                        if idx in set_B_pos_idx:                   
                            set_B_pos_x.append(test_x[idx])
                            set_B_pos_y.append(1)


                    set_B_pos_x = np.array(set_B_pos_x)        
                    set_B_pos_y = np.array(set_B_pos_y)        
                    print("set_B_pos_x.shape:",set_B_pos_x.shape)
                    print("set_B_pos_y.shape:",set_B_pos_y.shape)
                        
                    set_BC_x = set_B_pos_x
                    set_BC_y = set_B_pos_y
                    print("set_BC_x.shape:",set_BC_x.shape)
                    print("set_BC_y.shape:",set_BC_y.shape)  
                    
                                                
                elif args.strategy =='strategy3': 
                    for idx, l in enumerate(test_y):                       
                        if idx in set_C_neg_idx:            
                            set_C_neg_x.append(test_x[idx])
                            set_C_neg_y.append(0)
                                    
                    set_C_neg_x = np.array(set_C_neg_x)        
                    set_C_neg_y = np.array(set_C_neg_y)        
                    print("set_C_neg_x.shape:",set_C_neg_x.shape)
                    print("set_C_neg_y.shape:",set_C_neg_y.shape)
                                
                    set_BC_x = set_C_neg_x
                    set_BC_y = set_C_neg_y
                    print("set_BC_x.shape:",set_BC_x.shape)
                    print("set_BC_y.shape:",set_BC_y.shape)              

                elif args.strategy =='strategy4-groundtruth':
                    
                    set_B_pos_x = adv_testset_x 
                    set_B_pos_y = adv_testset_y
                    
                    set_B_pos_x = np.array(set_B_pos_x)        
                    set_B_pos_y = np.array(set_B_pos_y)        
                    print("set_B_pos_x.shape:",set_B_pos_x.shape)
                    print("set_B_pos_y.shape:",set_B_pos_y.shape)
                        
                    set_BC_x = set_B_pos_x
                    set_BC_y = set_B_pos_y
                    print("set_BC_x.shape:",set_BC_x.shape)
                    print("set_BC_y.shape:",set_BC_y.shape)  
                        
                elif args.strategy =='strategy5-groundtruth-C':
                    
                    set_B_pos_x = adv_testset_x 
                    set_B_pos_y = adv_testset_y
                    
                    set_B_pos_x = np.array(set_B_pos_x)        
                    set_B_pos_y = np.array(set_B_pos_y)        
                    print("set_B_pos_x.shape:",set_B_pos_x.shape)
                    print("set_B_pos_y.shape:",set_B_pos_y.shape)
        
                    for idx, l in enumerate(test_y): 
                        if idx in set_C_neg_idx:            
                            set_C_neg_x.append(test_x[idx])
                            set_C_neg_y.append(0)
                                
                    set_C_neg_x = np.array(set_C_neg_x)        
                    set_C_neg_y = np.array(set_C_neg_y)        
                            
                    print("set_C_neg_x.shape:",set_C_neg_x.shape)
                    print("set_C_neg_y.shape:",set_C_neg_y.shape)
                                    
                
                    if set_B_pos_x.shape[0] == 0:
                        set_BC_x = set_C_neg_x
                        set_BC_y = set_C_neg_y
                    else:                    
                        if set_C_neg_x.shape[0] == 0:
                            set_BC_x = set_B_pos_x
                            set_BC_y = set_B_pos_y                        
                        else:    
                            set_BC_x = np.concatenate((set_B_pos_x,set_C_neg_x))
                            set_BC_y = np.concatenate((set_B_pos_y,set_C_neg_y))
                        
                    print("set_BC_x.shape:",set_BC_x.shape)
                    print("set_BC_y.shape:",set_BC_y.shape)              

                        
                if set_BC_x.shape[0] == 0:
                    retrainset_x = detector.dataset['train'][0]
                    retrainset_y = detector.dataset['train'][1]                      
                else:        
                    retrainset_x = np.concatenate((set_BC_x, detector.dataset['train'][0]))
                    retrainset_y = np.concatenate((set_BC_y, detector.dataset['train'][1]))   
                
                print("retrainset_x.shape:",retrainset_x.shape)
                print("retrainset_y.shape:",retrainset_y.shape)            

                #---------------------------------------------
                #
                # retrain 
                #
                #---------------------------------------------  
                
                rou_cost_time=detector.retrain(retrainset_x=retrainset_x, retrainset_y=retrainset_y, timesteps=args.timesteps, curround_exp_result_dir=curround_exp_result_dir)



                #---------------------------------------------
                #
                # evaluate retrained infection detector
                #
                #---------------------------------------------         
                print(f">>>>>>>> Evaluate {r+1} round retrained {detector.modelname} on {r+1+1} sub clean test data (former)>>>>>>>>")
                test_acc, test_los, test_TP, test_FP, test_TN, test_FN, test_recall, test_precision, test_F1 = detector.test(testset_x=sub_cle_testsets_x[r+1], testset_y=sub_cle_testsets_y[r+1],timesteps=args.timesteps)
                
                
                
                FPrate = round((test_FP/(test_FP+test_TN)), 4)
                FNrate = round((test_FN/(test_FN+test_TP)), 4)
                
                metrics_dic = { 
                            'model': detector.modelname,
                            'clean test Accuracy': f'{test_acc*100:.2f}%',
                            'clean test Loss': test_los,
                            'clean test TP': test_TP,
                            'clean test FP': test_FP,
                            'clean test TN': test_TN,
                            'clean test FN': test_FN,
                            'clean test Recall': f'{test_recall*100:.2f}%',
                            'clean test Precision': f'{test_precision*100:.2f}%',
                            'clean test F1': f'{test_F1*100:.2f}%',
                            'clean test FPrate': f'{FPrate*100:.2f}%',
                            'clean test FNrate': f'{FNrate*100:.2f}%',    
                            }
                print(f"{r+1} round retrained {detector.modelname} on {r+1+1} clean sub testset metrics_dic:\n {metrics_dic}")              
                
                test_FPrate_on_former_set_list.append(FPrate*100)
                test_FNrate_on_former_set_list.append(FNrate*100)
                test_acc_on_former_set_list.append(test_acc*100)
                test_los_on_former_set_list.append(test_los)
                test_TP_on_former_set_list.append(test_TP)
                test_FP_on_former_set_list.append(test_FP)
                test_TN_on_former_set_list.append(test_TN)
                test_FN_on_former_set_list.append(test_FN)
                test_recall_on_former_set_list.append(test_recall*100)
                test_precision_on_former_set_list.append(test_precision*100)
                test_F1_on_former_set_list.append(test_F1*100)
                
                cost_time_list.append(rou_cost_time)

                #------------------------------------
                print(f">>>>>>>> Evaluate {r+1} round retrained {detector.modelname} on {r+1+2} sub clean test data (later)>>>>>>>>")
                
                if r+1 >= args.relabel_rounds:
                    test_acc, test_los, test_TP, test_FP, test_TN, test_FN, test_recall, test_precision, test_F1=0, 0, 0, 0, 0, 0, 0, 0, 0
                else:
                    test_acc, test_los, test_TP, test_FP, test_TN, test_FN, test_recall, test_precision, test_F1 = detector.test(testset_x=sub_cle_testsets_x[r+2], testset_y=sub_cle_testsets_y[r+2],timesteps=args.timesteps)
                    
                
                if (test_FP+test_TN) == 0:
                    FPrate = 0
                else:
                    FPrate = round((test_FP/(test_FP+test_TN)), 4)
                if (test_FN+test_TP) == 0:    
                    FNrate = 0
                else:  
                    FNrate = round((test_FN/(test_FN+test_TP)), 4)
                
                metrics_dic = { 
                            'model': detector.modelname,
                            'clean test Accuracy': f'{test_acc*100:.2f}%',
                            'clean test Loss': test_los,
                            'clean test TP': test_TP,
                            'clean test FP': test_FP,
                            'clean test TN': test_TN,
                            'clean test FN': test_FN,
                            'clean test Recall': f'{test_recall*100:.2f}%',
                            'clean test Precision': f'{test_precision*100:.2f}%',
                            'clean test F1': f'{test_F1*100:.2f}%',
                            'clean test FPrate': f'{FPrate*100:.2f}%',
                            'clean test FNrate': f'{FNrate*100:.2f}%',    
                            }
                print(f"{r+1} round retrained {detector.modelname} on {r+1+2} clean sub testset metrics_dic:\n {metrics_dic}")              
                
                test_FPrate_on_later_set_list.append(FPrate*100)
                test_FNrate_on_later_set_list.append(FNrate*100)
                test_acc_on_later_set_list.append(test_acc*100)
                test_los_on_later_set_list.append(test_los)
                test_TP_on_later_set_list.append(test_TP)
                test_FP_on_later_set_list.append(test_FP)
                test_TN_on_later_set_list.append(test_TN)
                test_FN_on_later_set_list.append(test_FN)
                test_recall_on_later_set_list.append(test_recall*100)
                test_precision_on_later_set_list.append(test_precision*100)
                test_F1_on_later_set_list.append(test_F1*100)


                #-----------------------










                print(f">>>>>>>> Evaluate {r+1} round retrained {detector.modelname} on {r+1+1} adversarial test data")
                adv_exp_result_dir = os.path.join(curround_exp_result_dir,f'advattack')
                os.makedirs(adv_exp_result_dir, exist_ok=True)
                adv_testset_x, adv_testset_y = detector.generate_advmail(timesteps=args.timesteps,cle_testset_x=sub_cle_testsets_x[r+1],cle_testset_y=sub_cle_testsets_y[r+1])
                
         

                adv_test_acc, adv_test_los, adv_test_TP, adv_test_FP, adv_test_TN, adv_test_FN, adv_test_recall, adv_test_precision, adv_test_F1 = detector.test(testset_x=adv_testset_x, testset_y=adv_testset_y, timesteps=args.timesteps)
                adv_FNrate = round((adv_test_FN/(adv_test_FN+adv_test_TP)), 4)

                adv_metrics_dic = { 
                            'model': detector.modelname,
                            'adv test Accuracy': f'{adv_test_acc*100:.2f}%',
                            'adv test Loss': adv_test_los,
                            'adv test TP': adv_test_TP,
                            'adv test FP': adv_test_FP,
                            'adv test TN': adv_test_TN,
                            'adv test FN': adv_test_FN,
                            'adv test Recall': f'{adv_test_recall*100:.2f}%',
                            'adv test Precision': f'{adv_test_precision*100:.2f}%',
                            'adv test F1': f'{adv_test_F1*100:.2f}%',
                            'adv test FNrate': f'{adv_FNrate*100:.2f}%',
                            }
                print(f"{r+1}th-round retrained {detector.modelname} adv_metrics_dic:\n {adv_metrics_dic}")    
                
                adv_test_FNrate_list.append(adv_FNrate*100)
                adv_test_acc_list.append(adv_test_acc*100)
                adv_test_los_list.append(adv_test_los)
                adv_test_TN_list.append(adv_test_TN)
                adv_test_FN_list.append(adv_test_FN)
                adv_test_recall_list.append(adv_test_recall*100)
                adv_test_precision_list.append(adv_test_precision*100)
                adv_test_F1_list.append(adv_test_F1*100)
            
            
            #---------------------------------------------
            #
            # draw figure
            #
            #---------------------------------------------

            #---------------evaluate on clean testset---------------
            retrain_cle_exp_result_dir = os.path.join(exp_result_dir,f'retrain-evaluate-cle')
            os.makedirs(retrain_cle_exp_result_dir, exist_ok=True)

            loss_png_name = f'Test loss of retrained {detector.modelname} on clean testset'
            accuracy_png_name = f'Accuracy of retrained {detector.modelname} on clean testset'
            costtime_png_name = f'Cost Time of retrain {detector.modelname}'
            fn_fp_png_name = f'FP and FN of retrained {detector.modelname} on clean testset'
            recall_png_name = f'Recall of retrained {detector.modelname} on clean testset'
            f1_png_name = f'F1 of retrained {detector.modelname} on clean testset'
            fnrate_fprate_png_name = f'FP rate and FN rate of retrained {detector.modelname} on clean testset'
            
            num_tagged_mal_png_name = f'Number of tagged malicious amples'
            num_tagged_ben_png_name = f'Number of tagged benign samples'
            num_tagged_png_name = f'Number of tagged malicious or benign samples'        

            #---------------evaluate on adv testset--------------------------------
            retrain_adv_exp_result_dir = os.path.join(exp_result_dir,f'retrain-evaluate-adv')
            os.makedirs(retrain_adv_exp_result_dir, exist_ok=True)

            adv_loss_png_name = f'Test loss of retrained {detector.modelname} on {args.attack} examples'
            adv_accuracy_png_name = f'Accuracy of retrained {detector.modelname} on {args.attack} examples'
            adv_fn_png_name = f'FN of retrained {detector.modelname} on {args.attack} examples'
            adv_recall_png_name = f'Recall of retrained {detector.modelname} on {args.attack} examples'
            adv_f1_png_name = f'F1 of retrained {detector.modelname} on {args.attack} examples'
            adv_fnrate_png_name = f'FN rate of retrained {detector.modelname} on {args.attack} examples'
            adv_precision_png_name = f'Precision of retrained {detector.modelname} on {args.attack} examples'
            
     

            plt.bar(list(range(len(adv_test_los_list))), adv_test_los_list, width=0.2, label='Test Loss')
            plt.xlabel('Round')
            plt.ylabel('Loss on Adversarial Test Set')
            # 
            plt.xticks(range(min(list(range(len(adv_test_los_list)))), max(list(range(len(adv_test_los_list))))+1, 1)   )  
            # plt.xlim(left=0)
            plt.ylim(bottom=0)           
            # plt.legend()
            plt.title(f'{adv_loss_png_name}')
            plt.savefig(f'{retrain_adv_exp_result_dir}/{adv_loss_png_name}.png')
            plt.close()
                    
            plt.bar(list(range(len(adv_test_acc_list))), adv_test_acc_list, width=0.2, label='Test Accuracy')
            plt.xlabel('Round')
            plt.ylabel('Accuracy (%) on Adversarial Test Set')
            plt.ylim(0, 100)
            plt.xticks(range(min(list(range(len(adv_test_acc_list)))), max(list(range(len(adv_test_acc_list))))+1, 1)     )
            plt.ylim(bottom=0)                     
            plt.title(f'{adv_accuracy_png_name}')        
            plt.savefig(f'{retrain_adv_exp_result_dir}/{adv_accuracy_png_name}.png')
            plt.close()

            plt.bar(list(range(len(adv_test_FN_list))), adv_test_FN_list, width=0.2, label='Test False Negative')
            plt.xlabel('Round')
            plt.ylabel('FN on Adversarial Test Set')
            plt.xticks(range(min(list(range(len(adv_test_FN_list)))), max(list(range(len(adv_test_FN_list))))+1, 1))                
            plt.ylim(bottom=0)  
            plt.title(f'{adv_fn_png_name}')        
            plt.savefig(f'{retrain_adv_exp_result_dir}/{adv_fn_png_name}.png')
            plt.close()

            plt.bar(list(range(len(adv_test_FNrate_list))), adv_test_FNrate_list, width=0.2, label='Test False Negative rate')
            plt.xlabel('Round')
            plt.ylabel('FN Rate (%) on Adversarial Test Set')
            plt.ylim(0, 100)
            plt.xticks(range(min(list(range(len(adv_test_FNrate_list)))), max(list(range(len(adv_test_FNrate_list))))+1, 1) )      
            plt.ylim(bottom=0)                   
            plt.title(f'{adv_fnrate_png_name}')        
            plt.savefig(f'{retrain_adv_exp_result_dir}/{adv_fnrate_png_name}.png')
            plt.close()
            
            
            plt.bar(list(range(len(adv_test_recall_list))), adv_test_recall_list, width=0.2, label='Test Recall')
            plt.xlabel('Round')
            plt.ylabel('Recall (%) on Adversarial Test Set')
            plt.ylim(0, 100)
            plt.xticks(range(min(list(range(len(adv_test_recall_list)))), max(list(range(len(adv_test_recall_list))))+1, 1)    )
            plt.ylim(bottom=0)                      
            plt.title(f'{adv_recall_png_name}')        
            plt.savefig(f'{retrain_adv_exp_result_dir}/{adv_recall_png_name}.png')
            plt.close()        

            plt.bar(list(range(len(adv_test_precision_list))), adv_test_precision_list, width=0.2, label='Test Precision')
            plt.xlabel('Round')
            plt.ylabel('Precision (%) on Adversarial Test Set')
            plt.ylim(0, 100)
            plt.xticks(range(min(list(range(len(adv_test_precision_list)))), max(list(range(len(adv_test_precision_list))))+1, 1)    )
            plt.ylim(bottom=0)                      
            plt.title(f'{adv_precision_png_name}')        
            plt.savefig(f'{retrain_adv_exp_result_dir}/{adv_precision_png_name}.png')
            plt.close()      
            
            plt.bar(list(range(len(adv_test_F1_list))), adv_test_F1_list, width=0.2, label='Test F1')
            plt.xlabel('Round')
            plt.ylabel('F1 (%) on Adversarial Test Set')
            plt.ylim(0, 100)
            plt.xticks(range(min(list(range(len(adv_test_F1_list)))), max(list(range(len(adv_test_F1_list))))+1, 1)      )                        
            plt.ylim(bottom=0)            
            plt.title(f'{adv_f1_png_name}')        
            plt.savefig(f'{retrain_adv_exp_result_dir}/{adv_f1_png_name}.png')
            plt.close()                                 
    

            #-------------------tagged num---------------------

            
            bar_width = 0.2  
            x_positions = np.arange(1, len(seq2seq_tagged_mal_event_num_list)+1)

            fig, ax = plt.subplots()
            bar1 = ax.bar(x_positions - bar_width/2, seq2seq_tagged_mal_event_num_list, bar_width, label='tagged by seq2seq')
            bar2 = ax.bar(x_positions + bar_width/2, detector_tagged_mal_event_num_list, bar_width, label='tagged by detector')
                            
            plt.xlabel('Round')
            plt.ylabel('Number of Tagged Malicious Events')
            plt.xticks(range(1, len(seq2seq_tagged_mal_event_num_list)+1, 1)  )   

            plt.legend(loc='best',frameon=True)
            plt.title(f'{num_tagged_mal_png_name}')        
            plt.savefig(f'{retrain_cle_exp_result_dir}/{num_tagged_mal_png_name}.png')
            plt.close()
              
              
              

            bar_width = 0.2  
            x_positions = np.arange(1, len(seq2seq_tagged_ben_event_num_list)+1)

            fig, ax = plt.subplots()
            bar1 = ax.bar(x_positions - bar_width/2, seq2seq_tagged_ben_event_num_list, bar_width, label='tagged by seq2seq')
            bar2 = ax.bar(x_positions + bar_width/2, detector_tagged_ben_event_num_list, bar_width, label='tagged by detector') 
            
                                        
            plt.xlabel('Round')
            plt.ylabel('Number of Tagged Benign Events')
            plt.xticks(range(1, len(seq2seq_tagged_ben_event_num_list)+1, 1)  )   

            plt.legend(loc='best',frameon=True)
            plt.title(f'{num_tagged_ben_png_name}')        
            plt.savefig(f'{retrain_cle_exp_result_dir}/{num_tagged_ben_png_name}.png')
            plt.close()  
            
            
       

            bar_width = 0.2
            x_positions = np.arange(1, len(seq2seq_tagged_mal_event_num_list)+1)

            fig, ax = plt.subplots()
            bar_Y1 = ax.bar(x_positions - bar_width*1.5, seq2seq_tagged_mal_event_num_list, bar_width, label='tagged malicious by seq2seq')
            bar_Y2 = ax.bar(x_positions - bar_width*0.5, seq2seq_tagged_ben_event_num_list, bar_width, label='tagged benign by seq2seq')
            bar_Y3 = ax.bar(x_positions + bar_width*0.5, detector_tagged_mal_event_num_list, bar_width, label='tagged malicious by detector')
            bar_Y4 = ax.bar(x_positions + bar_width*1.5, detector_tagged_ben_event_num_list, bar_width, label='tagged benign by detector')


            plt.xlabel('Round')
            plt.ylabel('Number of Tagged Malicious/Benign Events')
            plt.xticks(range(1, len(seq2seq_tagged_ben_event_num_list)+1, 1)  )   

            plt.legend(loc='best',frameon=True)
            plt.title(f'{num_tagged_png_name}')        
            plt.savefig(f'{retrain_cle_exp_result_dir}/{num_tagged_png_name}.png')
            plt.close()  
            





















            #------------------------clean plt换成bar---------------            
            bar_width = 0.2  
            x_positions = np.arange(len(test_los_on_former_set_list))
            fig, ax = plt.subplots()
            bar1 = ax.bar(x_positions - bar_width/2, test_los_on_former_set_list, bar_width, label='Loss on Former Test Set')
            bar2 = ax.bar(x_positions + bar_width/2, test_los_on_later_set_list, bar_width, label='Loss on Later Test Set')             
            
            
            plt.xlabel('Round')
            plt.ylabel('Loss on Clean Test Set')
            plt.xticks(range(min(list(range(len(test_los_on_former_set_list)))), max(list(range(len(test_los_on_former_set_list))))+1, 1))
            plt.ylim(bottom=0)
            plt.legend(loc='best',frameon=True)
            plt.title(f'{loss_png_name}')
            plt.savefig(f'{retrain_cle_exp_result_dir}/{loss_png_name}.png')
            plt.close()
                    
                    
                    
            #-----------special----------                    
            bar_width = 0.2  
            x_positions = np.arange(len(test_acc_on_former_set_list))
            fig, ax = plt.subplots()
            bar1 = ax.bar(x_positions - bar_width/2, test_acc_on_former_set_list, bar_width, label='Accuracy on Former Test Set')
            bar2 = ax.bar(x_positions + bar_width/2, test_acc_on_later_set_list, bar_width, label='Accuracy on Later Test Set')      
                        
            plt.xlabel('Round')
            plt.ylabel('Accuracy (%) on Clean Test Set')
            plt.xticks(range(min(list(range(len(test_acc_on_former_set_list)))), max(list(range(len(test_acc_on_former_set_list))))+1, 1))
            plt.ylim(0, 100)
            plt.ylim(bottom=0)
            plt.legend(loc='lower right',frameon=True)
            plt.title(f'{accuracy_png_name}')        
            plt.savefig(f'{retrain_cle_exp_result_dir}/{accuracy_png_name}.png')
            plt.close()
            

            plt.bar(list(range(len(cost_time_list))), cost_time_list, width=0.2, label='Cost Time')
            plt.xlabel('Round')
            plt.ylabel('Cost Time (second)')
            plt.xticks(range(min(list(range(len(cost_time_list)))), max(list(range(len(cost_time_list))))+1, 1)      )                
            plt.ylim(bottom=0) 
            plt.xlim(left=0)
            plt.title(f'{costtime_png_name}')        
            plt.savefig(f'{retrain_cle_exp_result_dir}/{costtime_png_name}.png')
            plt.close()

            bar_width = 0.2
            x_positions = np.arange(len(test_FP_on_former_set_list))

            fig, ax = plt.subplots()
            bar_Y1 = ax.bar(x_positions - bar_width*1.5, test_FP_on_former_set_list, bar_width, label='FP on Former Test Set')
            bar_Y2 = ax.bar(x_positions - bar_width*0.5, test_FN_on_former_set_list, bar_width, label='FN on Former Test Set')
            bar_Y3 = ax.bar(x_positions + bar_width*0.5, test_FP_on_later_set_list, bar_width, label='FP on Later Test Set')
            bar_Y4 = ax.bar(x_positions + bar_width*1.5, test_FN_on_later_set_list, bar_width, label='FN on Later Test Set')
            
        
            plt.xlabel('Round')
            plt.ylabel('FP and FN on Clean Test Set')
            plt.xticks(range(min(list(range(len(test_FP_on_former_set_list)))), max(list(range(len(test_FP_on_former_set_list))))+1, 1))        
            plt.ylim(bottom=0)
            plt.legend(loc='best',frameon=True)
            plt.title(f'{fn_fp_png_name}')        
            plt.savefig(f'{retrain_cle_exp_result_dir}/{fn_fp_png_name}.png')
            plt.close()       
            bar_width = 0.2
            x_positions = np.arange(len(test_FPrate_on_former_set_list))

            fig, ax = plt.subplots()
            bar_Y1 = ax.bar(x_positions - bar_width*1.5, test_FPrate_on_former_set_list, bar_width, label='FP rate on Former Test Set')
            bar_Y2 = ax.bar(x_positions - bar_width*0.5, test_FNrate_on_former_set_list, bar_width, label='FN rate on Former Test Set')
            bar_Y3 = ax.bar(x_positions + bar_width*0.5, test_FPrate_on_later_set_list, bar_width, label='FP rate on Later Test Set')
            bar_Y4 = ax.bar(x_positions + bar_width*1.5, test_FNrate_on_later_set_list, bar_width, label='FN rate on Later Test Set')                        
                        
                        
                        
            plt.xlabel('Round')
            plt.ylabel('FP Rate and FN Rate (%) on Clean Test Set')
            plt.ylim(0, 100)
            plt.xticks(range(min(list(range(len(test_FPrate_on_former_set_list)))), max(list(range(len(test_FPrate_on_former_set_list))))+1, 1)  )   
            plt.ylim(bottom=0)   
            plt.legend(loc='best',frameon=True)
            plt.title(f'{fnrate_fprate_png_name}')        
            plt.savefig(f'{retrain_cle_exp_result_dir}/{fnrate_fprate_png_name}.png')
            plt.close()
            
            
            
            
            bar_width = 0.2  
            x_positions = np.arange(len(test_recall_on_former_set_list))
            fig, ax = plt.subplots()
            bar1 = ax.bar(x_positions - bar_width/2, test_recall_on_former_set_list, bar_width, label='Recall on Former Test Set')
            bar2 = ax.bar(x_positions + bar_width/2, test_recall_on_later_set_list, bar_width, label='Recall on Later Test Set')      
            
                        
            plt.xlabel('Round')
            plt.ylabel('Recall (%) on Test Set')
            plt.ylim(0, 100)
            plt.xticks(range(min(list(range(len(test_recall_on_former_set_list)))), max(list(range(len(test_recall_on_former_set_list))))+1, 1)  )   
            plt.ylim(bottom=0)   
            plt.legend(loc='best',frameon=True)
            plt.title(f'{recall_png_name}')        
            plt.savefig(f'{retrain_cle_exp_result_dir}/{recall_png_name}.png')
            plt.close()        





            
            bar_width = 0.2  
            x_positions = np.arange(len(test_F1_on_former_set_list))
            fig, ax = plt.subplots()
            bar1 = ax.bar(x_positions - bar_width/2, test_F1_on_former_set_list, bar_width, label='F1 on Former Test Set')
            bar2 = ax.bar(x_positions + bar_width/2, test_F1_on_later_set_list, bar_width, label='F1 on Later Test Set')      
                        
            
            plt.xlabel('Round')
            plt.ylabel('F1 (%) on Test Set')
            plt.ylim(0, 100)
            plt.xticks(range(min(list(range(len(test_F1_on_former_set_list)))), max(list(range(len(test_F1_on_former_set_list))))+1, 1)   ) 
            plt.ylim(bottom=0)            
            plt.legend(loc='best',frameon=True)
            plt.title(f'{f1_png_name}')        
            plt.savefig(f'{retrain_cle_exp_result_dir}/{f1_png_name}.png')
            plt.close()                                 
     
                    
            #---------------save xlsx data---------------

            cle_figure_xlsx_result_dir = os.path.join(retrain_cle_exp_result_dir,f'figure-xlsx')
            os.makedirs(cle_figure_xlsx_result_dir, exist_ok=True)        
            
            
            pd.DataFrame(test_FPrate_on_former_set_list).to_excel(f'{cle_figure_xlsx_result_dir}/test_FPrate_on_former_set_list.xlsx')
            pd.DataFrame(test_FNrate_on_former_set_list).to_excel(f'{cle_figure_xlsx_result_dir}/test_FNrate_on_former_set_list.xlsx')
            pd.DataFrame(test_acc_on_former_set_list).to_excel(f'{cle_figure_xlsx_result_dir}/test_acc_on_former_set_list.xlsx')
            pd.DataFrame(test_los_on_former_set_list).to_excel(f'{cle_figure_xlsx_result_dir}/test_los_on_former_set_list.xlsx')
            pd.DataFrame(test_TP_on_former_set_list).to_excel(f'{cle_figure_xlsx_result_dir}/test_TP_on_former_set_list.xlsx')
            pd.DataFrame(test_FP_on_former_set_list).to_excel(f'{cle_figure_xlsx_result_dir}/test_FP_on_former_set_list.xlsx')
            pd.DataFrame(test_TN_on_former_set_list).to_excel(f'{cle_figure_xlsx_result_dir}/test_TN_on_former_set_list.xlsx')
            pd.DataFrame(test_FN_on_former_set_list).to_excel(f'{cle_figure_xlsx_result_dir}/test_FN_on_former_set_list.xlsx')
            pd.DataFrame(test_recall_on_former_set_list).to_excel(f'{cle_figure_xlsx_result_dir}/test_on_former_set_recall_list.xlsx')
            pd.DataFrame(test_precision_on_former_set_list).to_excel(f'{cle_figure_xlsx_result_dir}/test_precision_on_former_set_list.xlsx')
            pd.DataFrame(test_F1_on_former_set_list).to_excel(f'{cle_figure_xlsx_result_dir}/test_F1_on_former_set_list.xlsx')
            

            pd.DataFrame(test_FPrate_on_later_set_list).to_excel(f'{cle_figure_xlsx_result_dir}/test_FPrate_on_later_set_list.xlsx')
            pd.DataFrame(test_FNrate_on_later_set_list).to_excel(f'{cle_figure_xlsx_result_dir}/test_FNrate_on_later_set_list.xlsx')
            pd.DataFrame(test_acc_on_later_set_list).to_excel(f'{cle_figure_xlsx_result_dir}/test_acc_on_later_set_list.xlsx')
            pd.DataFrame(test_los_on_later_set_list).to_excel(f'{cle_figure_xlsx_result_dir}/test_los_on_later_set_list.xlsx')
            pd.DataFrame(test_TP_on_later_set_list).to_excel(f'{cle_figure_xlsx_result_dir}/test_TP_on_later_set_list.xlsx')
            pd.DataFrame(test_FP_on_later_set_list).to_excel(f'{cle_figure_xlsx_result_dir}/test_FP_on_later_set_list.xlsx')
            pd.DataFrame(test_TN_on_later_set_list).to_excel(f'{cle_figure_xlsx_result_dir}/test_TN_on_later_set_list.xlsx')
            pd.DataFrame(test_FN_on_later_set_list).to_excel(f'{cle_figure_xlsx_result_dir}/test_FN_on_later_set_list.xlsx')
            pd.DataFrame(test_recall_on_later_set_list).to_excel(f'{cle_figure_xlsx_result_dir}/test_on_later_set_recall_list.xlsx')
            pd.DataFrame(test_precision_on_later_set_list).to_excel(f'{cle_figure_xlsx_result_dir}/test_precision_on_later_set_list.xlsx')
            pd.DataFrame(test_F1_on_later_set_list).to_excel(f'{cle_figure_xlsx_result_dir}/test_F1_on_later_set_list.xlsx')
            
                        
            
            pd.DataFrame(cost_time_list).to_excel(f'{cle_figure_xlsx_result_dir}/cost_time_list.xlsx')                
            pd.DataFrame(seq2seq_tagged_mal_event_num_list).to_excel(f'{cle_figure_xlsx_result_dir}/seq2seq_tagged_mal_event_num_list.xlsx')
            pd.DataFrame(seq2seq_tagged_ben_event_num_list).to_excel(f'{cle_figure_xlsx_result_dir}/seq2seq_tagged_ben_event_num_list.xlsx')      
            
            
                

            adv_figure_xlsx_result_dir = os.path.join(retrain_adv_exp_result_dir,f'figure-xlsx')
            os.makedirs(adv_figure_xlsx_result_dir, exist_ok=True)          
            pd.DataFrame(adv_test_FNrate_list).to_excel(f'{adv_figure_xlsx_result_dir}/adv_test_FNrate_list.xlsx')
            pd.DataFrame(adv_test_acc_list).to_excel(f'{adv_figure_xlsx_result_dir}/adv_test_acc_list.xlsx')
            pd.DataFrame(adv_test_los_list).to_excel(f'{adv_figure_xlsx_result_dir}/adv_test_los_list.xlsx')
            pd.DataFrame(adv_test_TN_list).to_excel(f'{adv_figure_xlsx_result_dir}/adv_test_TN_list.xlsx')
            pd.DataFrame(adv_test_FN_list).to_excel(f'{adv_figure_xlsx_result_dir}/adv_test_FN_list.xlsx')
            pd.DataFrame(adv_test_recall_list).to_excel(f'{adv_figure_xlsx_result_dir}/adv_test_recall_list.xlsx')
            pd.DataFrame(adv_test_precision_list).to_excel(f'{adv_figure_xlsx_result_dir}/adv_test_precision_list.xlsx')
            pd.DataFrame(adv_test_F1_list).to_excel(f'{adv_figure_xlsx_result_dir}/adv_test_F1_list.xlsx')
            
                        
        
    print(f"finish retrain the {detector.modelname}")