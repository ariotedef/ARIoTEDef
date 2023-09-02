from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os
os.environ['TF_NUMA_NODES'] = '1'
from keras.callbacks import EarlyStopping
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import math
import numpy as np
import time
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy

from keras.layers import RepeatVector, TimeDistributed, BatchNormalization, Activation, Input, dot, concatenate, Attention
from tensorflow.keras.optimizers import Adam
from keras.models import Model    
from tensorflow.keras import optimizers

import sys
sys.stdout.flush()
print("正在导入art库...", flush=True)
from art.estimators.classification.keras import KerasClassifier
from art.attacks.evasion.projected_gradient_descent.projected_gradient_descent import ProjectedGradientDescent
from art.attacks.evasion.fast_gradient import FastGradientMethod
from art.attacks.evasion.zoo import ZooAttack   
from art.attacks.evasion.boundary import BoundaryAttack
from art.attacks.evasion.square_attack import SquareAttack 
from art.attacks.evasion.feature_adversaries.feature_adversaries_numpy import FeatureAdversariesNumpy
from art.attacks.evasion.feature_adversaries.feature_adversaries_pytorch import FeatureAdversariesPyTorch
from art.attacks.evasion.feature_adversaries.feature_adversaries_tensorflow import FeatureAdversariesTensorFlowV2
from art.attacks.evasion.hop_skip_jump import HopSkipJump
from art.attacks.evasion.carlini import CarliniL2Method


def apply_custom_threshold(predictions, threshold=0.5):
    return (predictions >= threshold).astype(int)


def calculate_tp_tn_fp_fn(y_true, y_pred):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            tp += 1
        elif true == 0 and pred == 0:
            tn += 1
        elif true == 0 and pred == 1:
            fp += 1
        elif true == 1 and pred == 0:
            fn += 1

    return tp, tn, fp, fn

def accuracy_score(tp,tn,fp,fn):
    if (tp+tn+fp+fn)==0:
        print("tp+tn+fp+fn=0")
        accuracy=0
    else:
        accuracy=(tp+tn)/(tp+tn+fp+fn)
    return accuracy

def recall_score(tp,fn):
    if (tp+fn)==0:
        print("tp+fn=0")
        recall=0
    else:
        recall=tp/(tp+fn)
    return recall
    
def precision_score(tp,fp):
    if (tp+fp)==0:
        print("tp+fp=0")
        precision=0
    else:    
        precision=tp/(tp+fp)
        
    return precision

def f1_score(precision,recall):
    if (precision + recall)==0:
        print("precision + recall=0")
        f1=0        
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    return f1
    

        
def seq2seq_model(input_shape, output_shape, hidden_units):
    train_input = Input(shape=input_shape)
    train_output = Input(shape=output_shape)    
    print("train_input.shape:", train_input.shape)
    print("train_output.shape:", train_output.shape)
    
    encoder_stack_h, encoder_last_h, encoder_last_c = LSTM(
            units=128, activation='relu', dropout=0.2, recurrent_dropout=0.2,
            return_sequences=True, return_state=True)(train_input)

    encoder_last_h = BatchNormalization(momentum=0.6)(encoder_last_h)
    encoder_last_c = BatchNormalization(momentum=0.6)(encoder_last_c)

    decoder_input = RepeatVector(train_output.shape[1])(encoder_last_h)
    decoder_stack_h = LSTM(units=128, activation='relu', dropout=0.2, recurrent_dropout=0.2,
            return_state=False, return_sequences=True)(decoder_input, initial_state=[encoder_last_h, encoder_last_c])
    
    
    print("encoder_stack_h.shape:",encoder_stack_h.shape)
    print("decoder_stack_h.shape:",decoder_stack_h.shape)
    
    attention = dot([decoder_stack_h, encoder_stack_h], axes=[2,2])
    attention = Activation('softmax')(attention)

    context = dot([attention, encoder_stack_h], axes=[2,1])
    context = BatchNormalization(momentum=0.6)(context)

    decoder_combined_context = concatenate([context, decoder_stack_h])
    out = TimeDistributed(Dense(train_output.shape[2]))(decoder_combined_context)
    out = Activation('sigmoid')(out)

    print("train_input.shape:",train_input.shape)                
    print("out.shape:",out.shape)       
    
    
    model = Model(inputs=train_input, outputs=out)
    
    return model

        
class EpochTimer(Callback):
    def __init__(self):
        self.epoch_times = []
    
    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = time.time() - self.start_time
        self.epoch_times.append(elapsed_time)

class PSDetector():
    def __init__(self, name, args):
        self.modelname = name
        self.args = args
        
    def add_dataset(self, dataset):
        self.dataset = dataset

        self.trainset_min = np.min(self.dataset['train'][0])
        self.trainset_max = np.max(self.dataset['train'][0])
        self.testset_min = np.min(self.dataset['test'][0])
        self.testset_max = np.max(self.dataset['test'][0])    
            
        print(f"{self.modelname} trainset_min:{self.trainset_min:.4f}")
        print(f"{self.modelname} trainset_max:{self.trainset_max:.4f}")
        print(f"{self.modelname} testset_min:{self.testset_min:.4f}")
        print(f"{self.modelname} testset_max:{self.testset_max:.4f}")            
                
    def def_model(self, input_dim=41, output_dim=1, timesteps=1):  

        model = Sequential()

        model.add(LSTM(units=128, activation='relu', return_sequences=True, input_shape=(timesteps, int(input_dim / timesteps))))
        model.add(LSTM(units=128, activation='relu', return_sequences=True))                        
        model.add(Dense(units=output_dim, activation='sigmoid'))
        model.add(Flatten())
        self.model = model

    def stdtrain(self, timesteps, exp_result_dir):
        
        if tf.test.is_built_with_cuda() and tf.config.list_physical_devices('GPU'):
            print("Cuda and GPU are available")
   
        trainset_x = self.dataset['train'][0]
        trainset_y = self.dataset['train'][1]
        
        print("trainset_x.shape:",trainset_x.shape)
        print("trainset_y.shape:",trainset_y.shape)
        
        trainset_x, trainset_y = shuffle(trainset_x, trainset_y)
        
        trainset_x = trainset_x.reshape((trainset_x.shape[0], timesteps, int(math.ceil(trainset_x.shape[1] / timesteps))))


        condition = trainset_y.astype(bool)
        malicious_trainset_y = np.extract(condition,trainset_y)
        print("malicious_trainset_y.shape:",malicious_trainset_y.shape)
        print("malicious_trainset_y:",malicious_trainset_y)
    
        benign_trainset_y = np.extract(1-condition,trainset_y)
        print("benign_trainset_y.shape:",benign_trainset_y.shape)
        print("benign_trainset_y:",benign_trainset_y)        
                
        self.model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=self.args.lr), metrics=['accuracy'])
        early_stop = EarlyStopping(monitor='val_loss', patience=self.args.patience, verbose=1)    
        timer_callback = EpochTimer()
        callbacks = [early_stop, timer_callback]
        
        print("trainset_x.shape:",trainset_x.shape)
        print("trainset_y.shape:",trainset_y.shape)


        history=self.model.fit(x=trainset_x, y=trainset_y, batch_size=self.args.batchsize, epochs=self.args.ps_epochs, verbose=2, callbacks=callbacks, validation_split=0.2)       


        epo_train_loss = history.history['loss']
        epo_val_loss = history.history['val_loss']
        epo_train_acc = history.history['accuracy']
        epo_val_acc = history.history['val_accuracy']
        epo_cost_time = timer_callback.epoch_times

        epo_train_acc = [accuracy * 100 for accuracy in epo_train_acc]
        epo_val_acc = [accuracy * 100 for accuracy in epo_val_acc]

        #--------save plt---------            
        loss_png_name = f'Loss of standard trained {self.modelname}'
        accuracy_png_name = f'Accuracy of standard trained {self.modelname}'        
        time_png_name = f'Cost time of standard trained {self.modelname}'
         
        plt.plot(list(range(1, len(epo_train_loss)+1)), epo_train_loss, label='Train Loss', marker='o')
        plt.plot(list(range(1, len(epo_val_loss)+1)), epo_val_loss, label='Validation Loss', marker='s')

        if len(epo_train_loss) <= 20:
            plt.xticks(range(1, len(epo_train_loss)+1, 1))
        else:
            plt.xticks(range(1, len(epo_train_loss)+1, 2))
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='best',frameon=True)
        plt.title(f'{loss_png_name}')
        plt.savefig(f'{exp_result_dir}/{loss_png_name}.png')
        plt.close()
                
        plt.plot(list(range(1, len(epo_train_acc)+1)), epo_train_acc, label='Train Accuracy', marker='o')
        plt.plot(list(range(1, len(epo_val_acc)+1)), epo_val_acc, label='Validation Accuracy', marker='s')     
       
        if len(epo_train_acc) <= 20:
            plt.xticks(range(1, len(epo_train_acc)+1, 1))
        else:
            plt.xticks(range(1, len(epo_train_acc)+1, 2))   
                    
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend(loc='best',frameon=True)
        plt.title(f'{accuracy_png_name}')        
        plt.savefig(f'{exp_result_dir}/{accuracy_png_name}.png')
        plt.close()

        plt.plot(list(range(1, len(epo_cost_time)+1)), epo_cost_time, marker='o')
        
        if len(epo_cost_time) <= 20:
            plt.xticks(range(1, len(epo_cost_time)+1, 1))
        else:
            plt.xticks(range(1, len(epo_cost_time)+1, 2))   
            
        plt.xlabel('Epoch')
        plt.ylabel('Cost Time (seconds)')
        plt.title(f'{time_png_name}')        
        plt.savefig(f'{exp_result_dir}/{time_png_name}.png')
        plt.close()
             
    def evaluate(self, testset_x, testset_y):
        
        test_los, _ = self.model.evaluate(testset_x, testset_y)
        output = self.model.predict(testset_x)
        
        predicts = []
        for p in output:
            ret = (p[0] > 0.5).astype("int32")
            predicts.append(ret)
            
        output = np.array(predicts)
        
        print("confusion_matrix(testset_y, output).ravel():",confusion_matrix(testset_y, output).ravel())
        
        test_TP, test_TN, test_FP, test_FN = calculate_tp_tn_fp_fn(y_true=testset_y, y_pred=output)

        test_acc = accuracy_score(tp=test_TP, tn=test_TN, fp=test_FP, fn=test_FN)
        test_recall = recall_score(tp=test_TP, fn=test_FN)
        test_precision = precision_score(tp=test_TP, fp=test_FP)
        test_F1 = f1_score(precision=test_precision, recall=test_recall)
                
        return round(test_acc, 4), round(test_los, 4), test_TP, test_FP, test_TN, test_FN, round(test_recall, 4), round(test_precision, 4), round(test_F1, 4)
    
    def test(self, testset_x, testset_y, timesteps):
        if tf.test.is_built_with_cuda() and tf.config.list_physical_devices('GPU'):
            print("Cuda and GPU are available")
 
        print("testset_x.shape:",testset_x.shape)
        print("testset_y.shape:",testset_y.shape)
        
        testset_x = testset_x.reshape((testset_x.shape[0], timesteps, int(math.ceil(testset_x.shape[1] / timesteps))))


        print("testset_x.shape:",testset_x.shape)
        print("testset_y.shape:",testset_y.shape)
                
        test_acc, test_los, test_TP, test_FP, test_TN, test_FN, test_recall, test_precision, test_F1 = self.evaluate(testset_x, testset_y)

        
        return test_acc, test_los, test_TP, test_FP, test_TN, test_FN, test_recall, test_precision, test_F1
  
        
    def generate_advmail(self,timesteps,cle_testset_x,cle_testset_y):        
        if tf.test.is_built_with_cuda() and tf.config.list_physical_devices('GPU'):
            print("Cuda and GPU are available")

        
        condition = cle_testset_y.astype(bool)

        malicious_cle_testset_y = np.extract(condition,cle_testset_y)
        print("malicious_cle_testset_y.shape:",malicious_cle_testset_y.shape)
        print("malicious_cle_testset_y:",malicious_cle_testset_y)

        benign_cle_testset_y = np.extract(1-condition,cle_testset_y)
        print("benign_cle_testset_y.shape:",benign_cle_testset_y.shape)
        print("benign_cle_testset_y:",benign_cle_testset_y)                                   
        
        cond=np.expand_dims(condition,1)
        cond_expend = np.full((cle_testset_x.shape[0], cle_testset_x.shape[1]), False, dtype=bool)
        cond = np.logical_or(cond_expend, cond)        
   
        
        malicious_cle_testset_x = np.extract(cond,cle_testset_x)
     
        malicious_cle_testset_x = np.reshape(malicious_cle_testset_x, (malicious_cle_testset_y.shape[0], cle_testset_x.shape[1]))        

        
        
        malicious_cle_testset_x = malicious_cle_testset_x.reshape((malicious_cle_testset_x.shape[0], timesteps, int(math.ceil(malicious_cle_testset_x.shape[1] / timesteps))))
        print("malicious_cle_testset_x.shape:",malicious_cle_testset_x.shape)

             
        art_classifier = KerasClassifier(model=self.model, clip_values=(self.testset_min, self.testset_max), use_logits=False)

        print(f'eps={self.args.eps},eps_step={self.args.eps_step},max_iter={self.args.max_iter}')


        if self.args.attack == 'pgd':
            if self.args.targeted:
                attack = ProjectedGradientDescent(estimator=art_classifier, eps=self.args.eps, eps_step=self.args.eps_step, max_iter=self.args.max_iter, targeted=True)
                adv_testset_x = attack.generate(x=malicious_cle_testset_x, y=np.zeros(len(malicious_cle_testset_x)))                
            else:    
                attack = ProjectedGradientDescent(estimator=art_classifier, eps=self.args.eps, eps_step=self.args.eps_step, max_iter=self.args.max_iter, targeted=False)
                adv_testset_x = attack.generate(x=malicious_cle_testset_x)
            
        elif self.args.attack == 'fgsm':
            
            if self.args.targeted:
                attack = FastGradientMethod(estimator=art_classifier, eps=self.args.eps, eps_step=self.args.eps, targeted=True)
                adv_testset_x = attack.generate(x=malicious_cle_testset_x, y=np.zeros(len(malicious_cle_testset_x)))
            else:
                attack = FastGradientMethod(estimator=art_classifier, eps=self.args.eps, eps_step=self.args.eps, targeted=False)
                adv_testset_x = attack.generate(x=malicious_cle_testset_x)
                            
        elif self.args.attack == 'boundary':
            if self.args.targeted:
                attack = BoundaryAttack(estimator=art_classifier, targeted=True, delta=self.args.eps, epsilon=self.args.eps, max_iter=self.args.max_iter, num_trial=100000, init_size=100000)    
                adv_testset_x = attack.generate(x=malicious_cle_testset_x, y=np.zeros(len(malicious_cle_testset_x)))
            else:    
                attack = BoundaryAttack(estimator=art_classifier, targeted=False, delta=self.args.eps, epsilon=self.args.eps, max_iter=self.args.max_iter, num_trial=100000, init_size=100000)    
                adv_testset_x = attack.generate(x=malicious_cle_testset_x)
                              
        elif self.args.attack == 'hopskipjump':
    
            if self.args.targeted: 
                attack = HopSkipJump(classifier=art_classifier, targeted=True, norm="inf", max_iter=self.args.max_iter, init_eval=10000, init_size=10000)         
                adv_testset_x = attack.generate(x=malicious_cle_testset_x, y=np.zeros(len(malicious_cle_testset_x)))
            else:
                attack = HopSkipJump(classifier=art_classifier, targeted=False, norm="inf", max_iter=self.args.max_iter, init_eval=10000, init_size=10000)         
                adv_testset_x = attack.generate(x=malicious_cle_testset_x)
            
        print("self.args.attack:",self.args.attack)        
        print("self.args.targeted:",self.args.targeted)            
        print(f'eps={self.args.eps},eps_step={self.args.eps_step},max_iter={self.args.max_iter}')
            
 
        print("malicious_cle_testset_x.shape:", malicious_cle_testset_x.shape)
        
        adv_testset_y = malicious_cle_testset_y
        

        
        adv_testset_x = adv_testset_x.reshape((adv_testset_x.shape[0],adv_testset_x.shape[2]))
 
    
        return adv_testset_x, adv_testset_y

    def retrain(self, retrainset_x, retrainset_y, timesteps, curround_exp_result_dir):
        
        if tf.test.is_built_with_cuda() and tf.config.list_physical_devices('GPU'):
            print("Cuda and GPU are available")

        print(f"prepare retraining set for learning {self.modelname} ")
        
        trainset_x = retrainset_x
        trainset_y = retrainset_y

        trainset_x, trainset_y = shuffle(trainset_x, trainset_y)
  
        trainset_x = trainset_x.reshape((trainset_x.shape[0], timesteps, int(math.ceil(trainset_x.shape[1] / timesteps))))     
      
        early_stop = EarlyStopping(monitor='val_loss', patience=self.args.patience, verbose=1)    
      
        timer_callback = EpochTimer()
        callbacks = [early_stop, timer_callback]
        history=self.model.fit(x=trainset_x, y=trainset_y, batch_size=self.args.batchsize, epochs=self.args.ps_epochs, verbose=2, callbacks=callbacks, validation_split=0.2)       

        epo_train_loss = history.history['loss']
        epo_val_loss = history.history['val_loss']
        epo_train_acc = history.history['accuracy']
        epo_val_acc = history.history['val_accuracy']
        epo_cost_time = timer_callback.epoch_times

        epo_train_acc = [accuracy * 100 for accuracy in epo_train_acc]
        epo_val_acc = [accuracy * 100 for accuracy in epo_val_acc]
        
        
        rou_cost_time = sum(epo_cost_time)
        #--------save plt---------            
        loss_png_name = f'Loss of retrained {self.modelname}'
        accuracy_png_name = f'Accuracy of retrained {self.modelname}'        
        time_png_name = f'Cost time of retrained {self.modelname}'
                        
        plt.plot(list(range(1, len(epo_train_loss)+1)), epo_train_loss, label='Train Loss', marker='o')
        plt.plot(list(range(1, len(epo_val_loss)+1)), epo_val_loss, label='Validation Loss', marker='s')
         
        if len(epo_val_loss) <= 20:
            plt.xticks(range(1, len(epo_val_loss)+1, 1))
        else:
            plt.xticks(range(1, len(epo_val_loss)+1, 2))   
            
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='best',frameon=True)
        plt.title(f'{loss_png_name}')
        plt.savefig(f'{curround_exp_result_dir}/{loss_png_name}.png')
        plt.close()
                
        plt.plot(list(range(1, len(epo_train_acc)+1)), epo_train_acc, label='Train Accuracy', marker='o')
        plt.plot(list(range(1, len(epo_val_acc)+1)), epo_val_acc, label='Validation Accuracy', marker='s')
         
        if len(epo_train_acc) <= 20:
            plt.xticks(range(1, len(epo_train_acc)+1, 1))
        else:
            plt.xticks(range(1, len(epo_train_acc)+1, 2))   
                    
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend(loc='best',frameon=True)
        plt.title(f'{accuracy_png_name}')        
        plt.savefig(f'{curround_exp_result_dir}/{accuracy_png_name}.png')
        plt.close()

        plt.plot(list(range(1, len(epo_cost_time)+1)), epo_cost_time, marker='o')
         
        if len(epo_cost_time) <= 20:
            plt.xticks(range(1, len(epo_cost_time)+1, 1))
        else:
            plt.xticks(range(1, len(epo_cost_time)+1, 2))   
                    
        plt.xlabel('Epoch')
        plt.ylabel('Cost Time (seconds)')
        plt.title(f'{time_png_name}')        
        plt.savefig(f'{curround_exp_result_dir}/{time_png_name}.png')
        plt.close()
        
        return rou_cost_time
        
    def load_model(self, model_path):
        from keras.models import load_model
        self.model = load_model(model_path)
        
    def save_model(self, save_path):
        self.model.save(save_path)
        
    def analysis(self, test_windows_x):
        detector_probs = self.model.predict(test_windows_x)
        print("detector_probs.shape",detector_probs.shape)
        detector_probs = np.array(detector_probs).squeeze()

        detector_tagged_mal_windows_idxs = []
        detector_tagged_mal_windows_probs = []
        detector_tagged_ben_windows_idxs = []
        detector_tagged_ben_windows_probs = []
                
        for idx, pred in enumerate(detector_probs):
            if pred>0.5:    # predicted label = infection
                detector_tagged_mal_windows_idxs.append(idx)
                detector_tagged_mal_windows_probs.append(pred)
            elif pred<=0.5:
                detector_tagged_ben_windows_idxs.append(idx)
                detector_tagged_ben_windows_probs.append(pred)                

        detector_tagged_mal_windows_probs = np.array(detector_tagged_mal_windows_probs)            
        detector_tagged_mal_windows_idxs = np.array(detector_tagged_mal_windows_idxs)   
        print("detector_tagged_mal_windows_idxs.shape",detector_tagged_mal_windows_idxs.shape)
        
        detector_tagged_ben_windows_probs = np.array(detector_tagged_ben_windows_probs)            
        detector_tagged_ben_windows_idxs = np.array(detector_tagged_ben_windows_idxs)   
        print("detector_tagged_ben_windows_idxs.shape",detector_tagged_ben_windows_idxs.shape)
                 
        return detector_tagged_mal_windows_probs, detector_tagged_mal_windows_idxs, detector_tagged_ben_windows_probs,detector_tagged_ben_windows_idxs

class Seq2Seq():
    def __init__(self, name, args):
        self.modelname = name
        self.args = args

    def add_dataset(self, dataset):
        self.dataset = dataset

        self.trainset_min = np.min(self.dataset['train'][0])
        self.trainset_max = np.max(self.dataset['train'][0])
        self.testset_min = np.min(self.dataset['test'][0])
        self.testset_max = np.max(self.dataset['test'][0])    
            
        print(f"{self.modelname} trainset_min:{self.trainset_min:.4f}")
        print(f"{self.modelname} trainset_max:{self.trainset_max:.4f}")
        print(f"{self.modelname} testset_min:{self.testset_min:.4f}")
        print(f"{self.modelname} testset_max:{self.testset_max:.4f}")  

    def probability_based_embedding(self, p, d):
        embedding_event=np.round(p, d)
        
        return embedding_event

    def truncate(self, x, y, idxs_order, slen=100):
        in_, out_, truncated_idxs = [], [], []

        for i in range(len(x) - slen + 1):
            in_.append(x[i:(i+slen)])
            out_.append(y[i:(i+slen)])
            truncated_idxs.append(idxs_order[i:(i+slen)])
        return np.array(in_), np.array(out_), np.array(truncated_idxs)

    def permute_truncated(self, X_in, X_out, truncated_idxs, slen=10, inplace=False):
        enable_permute_prints = False
        if not inplace:
            X_in = copy.copy(X_in)
            truncated_idxs = copy.copy(truncated_idxs)
        for x_seq_in, x_seq_out, seq_idxs in zip(X_in, X_out, truncated_idxs):
            repeating_seq = []
            permute_idxs = []
            i = 0
            current_label = x_seq_out[i]
            repeating_seq.append(i)
            i+=1
            while i < slen:
                prev_label = current_label
                current_label = x_seq_out[i]
                if i < 20 and enable_permute_prints:
                    print(i, current_label, prev_label)

                if prev_label != current_label: 
                    if i < 20 and enable_permute_prints:
                        print(repeating_seq)
                    
                    np.random.seed(self.args.seed)    
                    permute_idxs = permute_idxs + list(np.random.permutation(repeating_seq))
                    repeating_seq = []
                    repeating_seq.append(i)
                    i+=1
                else:
                    repeating_seq.append(i)
                    i+=1 
                if i < 20 and enable_permute_prints:
                    print(repeating_seq)
            
            np.random.seed(self.args.seed)    
            permute_idxs = permute_idxs + list(np.random.permutation(repeating_seq))
            if i < 20 and enable_permute_prints:
                print("permuting {} with idxs {}".format(x_seq_in, permute_idxs))
                print("permuting {} with idxs {}".format(seq_idxs, permute_idxs))
            self.permute_sublist_inplace(x_seq_in, permute_idxs)    
            self.permute_sublist_inplace(seq_idxs, permute_idxs)
        if not inplace:
            return X_in, truncated_idxs
    
    def evaluate(self, testset_x, testset_y):
        
        print("test seq2seq on clean test")

        print("testset_x.shape:",testset_x.shape)
        print("testset_y.shape:",testset_y.shape)
        print("testset_y[:3]:",testset_y[:3])
        
                
        output = self.model.predict(testset_x)
        print("output.shape:",output.shape)
        print("output[:3]:",output[:3])

         
       
        y_pred_binary = apply_custom_threshold(output, threshold=self.args.seq2seq_threshold)
        y_test_binary = apply_custom_threshold(testset_y, threshold=self.args.seq2seq_threshold)        
        
        #-----------------------------------------------------------------------------------------------
        print("y_pred_binary.shape:",y_pred_binary.shape)
        print("y_pred_binary[:3]:",y_pred_binary[:3])
        print("y_test_binary.shape:",y_test_binary.shape)                  
        print("y_test_binary[:3]:",y_test_binary[:3])                  
  
        

        y_test_binary_2d = y_test_binary.squeeze()
        y_pred_binary_2d = y_pred_binary.squeeze()
        
        y_test_binary_1d = []
        for row in y_test_binary_2d:
            counts = np.bincount(row)
            most_frequent_index = np.argmax(counts)
            y_test_binary_1d.append(most_frequent_index)
        y_test_binary_1d = np.array(y_test_binary_1d)


        y_pred_binary_1d = []
        for row in y_pred_binary_2d:
            counts = np.bincount(row)
            most_frequent_index = np.argmax(counts)
            y_pred_binary_1d.append(most_frequent_index)
        y_pred_binary_1d = np.array(y_pred_binary_1d)
                

        print("confusion_matrix(y_test_binary_1d, y_pred_binary_1d).ravel():",confusion_matrix(y_test_binary_1d, y_pred_binary_1d).ravel())
        test_TP, test_TN, test_FP, test_FN = calculate_tp_tn_fp_fn(y_true=y_test_binary_1d, y_pred=y_pred_binary_1d)

        test_acc = accuracy_score(tp=test_TP, tn=test_TN, fp=test_FP, fn=test_FN)
        test_recall = recall_score(tp=test_TP, fn=test_FN)
        test_precision = precision_score(tp=test_TP, fp=test_FP)
        test_F1 = f1_score(precision=test_precision, recall=test_recall)
        
        
        test_los, _ = self.model.evaluate(x=testset_x, y=y_test_binary)
    
        return round(test_acc, 4), round(test_los, 4), test_TP, test_FP, test_TN, test_FN, round(test_recall, 4), round(test_precision, 4), round(test_F1, 4)
      
    def test(self, events, labels):
        print("events.shape:",events.shape)
        print("labels.shape:",labels.shape)  
        
        slen = self.args.sequence_length

        if tf.test.is_built_with_cuda() and tf.config.list_physical_devices('GPU'):
            print("Cuda and GPU are available")

        testset_x, testset_y = [], []
        idx_order = []
        idx = 0        

        print("self.args.use_prob_embedding:",self.args.use_prob_embedding)
        print("self.args.roundvalue_d:",self.args.roundvalue_d)        
        for idx, (event, label) in enumerate(zip(events, labels)):
            
            if self.args.use_prob_embedding:
                event = self.probability_based_embedding(event, self.args.roundvalue_d)   
            
            testset_x.append(event)
            testset_y.append([label])
            idx_order.append(idx)
                    
        testset_x, testset_y, truncated_idxs = self.truncate(testset_x, testset_y, idx_order, slen=slen)

        print("testset_x.shape:", testset_x.shape)
        print("testset_y.shape:", testset_y.shape)

        test_acc, test_los, test_TP, test_FP, test_TN, test_FN, test_recall, test_precision, test_F1 = self.evaluate(testset_x, testset_y)
        
        
        return test_acc, test_los, test_TP, test_FP, test_TN, test_FN, test_recall, test_precision, test_F1
            
    def analysis(self, events, labels):
        if tf.test.is_built_with_cuda() and tf.config.list_physical_devices('GPU'):
            print("Cuda and GPU are available")

        slen = self.args.sequence_length

        print("events.shape:", events.shape)
        print("labels.shape:", labels.shape)

        testset_x, testset_y = [], []
        idx_order = []
        idx = 0
        
        print("self.args.use_prob_embedding:",self.args.use_prob_embedding)
        print("self.args.roundvalue_d:",self.args.roundvalue_d)        
        for idx, (event, label) in enumerate(zip(events, labels)):
            if self.args.use_prob_embedding:
                event = self.probability_based_embedding(event, self.args.roundvalue_d)            
             
            testset_x.append(event)        
            testset_y.append([label])
            idx_order.append(idx)
                    
        testset_x, testset_y, truncated_idxs = self.truncate(testset_x, testset_y, idx_order, slen=slen)

        print("testset_x.shape:",testset_x.shape)
        y_pred = self.model.predict(testset_x)
        print("y_pred.shape:",y_pred.shape)
        
        idx = 0
        predictions = {}   
        for pred in y_pred: 

            '''iterates through remaining truncated seq len if surpassing the limit'''
            if len(y_pred) - idx > slen: 
                lst = range(len(pred)) 
            else:
                lst = range(len(y_pred)-idx)

            '''acumulates truncated predictions: e.g. {idx1: [1,1,0,0], idx2: [1,0,0,1], idx3: [0,0,1,1], ...]'''
            for i in lst:       #y_i
                if idx + i not in predictions:
                    predictions[idx + i] = []
                predictions[idx + i].append(pred[i][0])     
            idx += 1
        
        '''looks like it takes the average of predictions for each truncated sequence? not sure'''
        results = []
        for idx in range(len(events) - slen + 1): 
            res = sum(predictions[idx])/len(predictions[idx])   
            results.append(res)
        
        seq2seq_tagged_mal_event_probs = []
        seq2seq_tagged_mal_event_idxs = [] 
        seq2seq_tagged_ben_event_probs=[]
        seq2seq_tagged_ben_event_idxs=[]
        
        print("self.args.seq2seq_threshold:",self.args.seq2seq_threshold)
        for idx in range(len(results)):
            if results[idx] > self.args.seq2seq_threshold:
                seq2seq_tagged_mal_event_probs.append(results[idx])
                seq2seq_tagged_mal_event_idxs.append(idx)
            else: 
                seq2seq_tagged_ben_event_probs.append(results[idx])
                seq2seq_tagged_ben_event_idxs.append(idx)
                
        seq2seq_tagged_mal_event_probs = np.array(seq2seq_tagged_mal_event_probs)
        seq2seq_tagged_mal_event_idxs = np.array(seq2seq_tagged_mal_event_idxs)
        seq2seq_tagged_ben_event_probs = np.array(seq2seq_tagged_ben_event_probs)
        seq2seq_tagged_ben_event_idxs = np.array(seq2seq_tagged_ben_event_idxs)
        
                        
        print("seq2seq_tagged_mal_event_idxs.shape:",seq2seq_tagged_mal_event_idxs.shape)   
        print("seq2seq_tagged_ben_event_idxs.shape:",seq2seq_tagged_ben_event_idxs.shape)   
             
        return seq2seq_tagged_mal_event_probs, seq2seq_tagged_mal_event_idxs, seq2seq_tagged_ben_event_probs, seq2seq_tagged_ben_event_idxs

    def def_model(self, input_length, output_length, input_dim=4, output_dim=1, hidden_units=128): 
        print("--------------------create seq2seq------------------------")        
    
        train_input = Input(shape=(input_length, input_dim))
        train_output = Input(shape=(output_length, output_dim))        
                    
        print("train_input.shape:", train_input.shape)
        print("train_output.shape:", train_output.shape)

        
        encoder_stack_h, encoder_last_h, encoder_last_c = LSTM(
                units=128, activation='relu', dropout=0.2, recurrent_dropout=0.2,
                return_sequences=True, return_state=True)(train_input)

        encoder_last_h = BatchNormalization(momentum=0.6)(encoder_last_h)
        encoder_last_c = BatchNormalization(momentum=0.6)(encoder_last_c)

        decoder_input = RepeatVector(train_output.shape[1])(encoder_last_h)
        decoder_stack_h = LSTM(units=128, activation='relu', dropout=0.2, recurrent_dropout=0.2,
                return_state=False, return_sequences=True)(decoder_input, initial_state=[encoder_last_h, encoder_last_c])
        
        
        print("encoder_stack_h.shape:",encoder_stack_h.shape)
        print("decoder_stack_h.shape:",decoder_stack_h.shape)
        
        attention = dot([decoder_stack_h, encoder_stack_h], axes=[2,2])
        attention = Activation('softmax')(attention)

        context = dot([attention, encoder_stack_h], axes=[2,1])
        context = BatchNormalization(momentum=0.6)(context)

        decoder_combined_context = concatenate([context, decoder_stack_h])
        out = TimeDistributed(Dense(train_output.shape[2]))(decoder_combined_context)
        out = Activation('sigmoid')(out)

        print("train_input.shape:",train_input.shape)                
        print("out.shape:",out.shape)       

        
        self.model = Model(inputs=train_input, outputs=out)
        print("--------------------end create seq2seq------------------------")    
                                
    def stdtrain(self, events, labels, exp_result_dir):

        if tf.test.is_built_with_cuda() and tf.config.list_physical_devices('GPU'):
            print("Cuda and GPU are available")

        print("events.shape:",events.shape)
        print("labels.shape:",labels.shape)
        
        slen = self.args.sequence_length
        print("self.args.sequence_length:",self.args.sequence_length)

        trainset_x, trainset_y = [], []
        idx_order = []
        idx = 0
        
        print("self.args.use_prob_embedding:",self.args.use_prob_embedding)
        print("self.args.roundvalue_d:",self.args.roundvalue_d)
        for idx, (event, label) in enumerate(zip(events, labels)):   
            if self.args.use_prob_embedding:
                event = self.probability_based_embedding(event, self.args.roundvalue_d)
                
            trainset_x.append(event)
            trainset_y.append([label])
            idx_order.append(idx)
            
        trainset_x, trainset_y, truncated_idxs = self.truncate(trainset_x, trainset_y, idx_order, slen=slen)
        trainset_x, trainset_y = shuffle(trainset_x, trainset_y)

        print("trainset_x.shape:",trainset_x.shape)
        print("trainset_y.shape:",trainset_y.shape)
        

        self.model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=self.args.lr), metrics=['accuracy'])
        
        early_stop = EarlyStopping(monitor='val_loss', patience=self.args.patience, verbose=1)    
        timer_callback = EpochTimer()
        callbacks = [early_stop, timer_callback]

        history = self.model.fit(x=trainset_x, y=trainset_y, batch_size=self.args.seq2seq_batchsize, epochs=self.args.seq2seq_epochs, verbose=2, callbacks=callbacks, validation_split=0.2)
          

        epo_train_loss = history.history['loss']
        epo_val_loss = history.history['val_loss']
        epo_train_acc = history.history['accuracy']
        epo_val_acc = history.history['val_accuracy']
        epo_cost_time = timer_callback.epoch_times

        epo_train_acc = [accuracy * 100 for accuracy in epo_train_acc]
        epo_val_acc = [accuracy * 100 for accuracy in epo_val_acc]
        
        #--------save plt---------            
        loss_png_name = f'Loss of standard trained {self.modelname}'
        accuracy_png_name = f'Accuracy of standard trained {self.modelname}'        
        time_png_name = f'Cost time of standard trained {self.modelname}'
                   
        plt.plot(list(range(1, len(epo_train_loss)+1)), epo_train_loss, label='Train Loss', marker='o')
        plt.plot(list(range(1, len(epo_val_loss)+1)), epo_val_loss, label='Validation Loss', marker='s')
        
        if len(epo_train_loss) <= 20:
            plt.xticks(range(1, len(epo_train_loss)+1, 1))
        else:
            plt.xticks(range(1, len(epo_train_loss)+1, 2))   
                    
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='best',frameon=True)
        plt.title(f'{loss_png_name}')
        plt.show()
        plt.savefig(f'{exp_result_dir}/{loss_png_name}.png')
        plt.close()
                
        plt.plot(list(range(1, len(epo_train_acc)+1)), epo_train_acc, label='Train Accuracy', marker='o')
        plt.plot(list(range(1, len(epo_val_acc)+1)), epo_val_acc, label='Validation Accuracy', marker='s')
   
        if len(epo_train_acc) <= 20:
            plt.xticks(range(1, len(epo_train_acc)+1, 1))
        else:
            plt.xticks(range(1, len(epo_train_acc)+1, 2))   
                                       
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend(loc='best',frameon=True)
        plt.title(f'{accuracy_png_name}')        
        plt.show()
        plt.savefig(f'{exp_result_dir}/{accuracy_png_name}.png')
        plt.close()

        plt.plot(list(range(1, len(epo_cost_time)+1)), epo_cost_time, marker='o')
             
        if len(epo_cost_time) <= 20:
            plt.xticks(range(1, len(epo_cost_time)+1, 1))
        else:
            plt.xticks(range(1, len(epo_cost_time)+1, 2))   

        plt.xlabel('Epoch')
        plt.ylabel('Cost Time (seconds)')
        plt.title(f'{time_png_name}')        
        plt.savefig(f'{exp_result_dir}/{time_png_name}.png')
        plt.close()
                    
    def save_model(self, save_path):
        self.model.save(save_path)

    def load_model(self, model_path):
        from keras.models import load_model
        self.model = load_model(model_path)
 
    def retrain(self, events, labels, exp_result_dir):
        if tf.test.is_built_with_cuda() and tf.config.list_physical_devices('GPU'):
            print("Cuda and GPU are available")

        print("events.shape:",events.shape)
        print("labels.shape:",labels.shape)

        
        slen = self.args.sequence_length
        print("self.args.sequence_length:",self.args.sequence_length)

        trainset_x, trainset_y = [], []
        idx_order = []
        idx = 0

        print("self.args.use_prob_embedding:",self.args.use_prob_embedding)
        print("self.args.roundvalue_d:",self.args.roundvalue_d)        
        for idx, (event, label) in enumerate(zip(events, labels)):   
            if self.args.use_prob_embedding:
                event = self.probability_based_embedding(event, self.args.roundvalue_d)   
                
            trainset_x.append(event)
            trainset_y.append([label])
            idx_order.append(idx)
            
        trainset_x, trainset_y, truncated_idxs = self.truncate(trainset_x, trainset_y, idx_order, slen=slen)
        trainset_x, trainset_y = shuffle(trainset_x, trainset_y)

        print("trainset_x.shape:",trainset_x.shape)
        print("trainset_y.shape:",trainset_y.shape)

        self.model.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr=self.args.lr), metrics=['accuracy'])
        print("compile success")
        early_stop = EarlyStopping(monitor='val_loss', patience=self.args.patience, verbose=1)    
        timer_callback = EpochTimer()
        callbacks = [early_stop, timer_callback]

        self.model.trainable = True
        print("trainable success")

        history = self.model.fit(x=trainset_x, y=trainset_y, batch_size=self.args.seq2seq_batchsize, epochs=self.args.seq2seq_epochs, verbose=2, callbacks=callbacks, validation_split=0.2)


        epo_train_loss = history.history['loss']
        epo_val_loss = history.history['val_loss']
        epo_train_acc = history.history['accuracy']
        epo_val_acc = history.history['val_accuracy']
        epo_cost_time = timer_callback.epoch_times

        epo_train_acc = [accuracy * 100 for accuracy in epo_train_acc]
        epo_val_acc = [accuracy * 100 for accuracy in epo_val_acc]
        
        #--------save plt---------            
        loss_png_name = f'Loss of retrained {self.modelname}'
        accuracy_png_name = f'Accuracy of retrained {self.modelname}'        
        time_png_name = f'Cost time of retrained {self.modelname}'
                   
        plt.plot(list(range(1, len(epo_train_loss)+1)), epo_train_loss, label='Train Loss', marker='o')
        plt.plot(list(range(1, len(epo_val_loss)+1)), epo_val_loss, label='Validation Loss', marker='s')
        
        if len(epo_train_loss) <= 20:
            plt.xticks(range(1, len(epo_train_loss)+1, 1))
        else:
            plt.xticks(range(1, len(epo_train_loss)+1, 2))   
            
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='best',frameon=True)
        plt.title(f'{loss_png_name}')
        plt.show()
        plt.savefig(f'{exp_result_dir}/{loss_png_name}.png')
        plt.close()
                
        plt.plot(list(range(1, len(epo_train_acc)+1)), epo_train_acc, label='Train Accuracy', marker='o')
        plt.plot(list(range(1, len(epo_val_acc)+1)), epo_val_acc, label='Validation Accuracy', marker='s')
  
        if len(epo_train_acc) <= 20:
            plt.xticks(range(1, len(epo_train_acc)+1, 1))
        else:
            plt.xticks(range(1, len(epo_train_acc)+1, 2))  
                               
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend(loc='best',frameon=True)
        plt.title(f'{accuracy_png_name}')        
        plt.show()
        plt.savefig(f'{exp_result_dir}/{accuracy_png_name}.png')
        plt.close()        
        
        plt.plot(list(range(1, len(epo_cost_time)+1)), epo_cost_time, marker='o')
    
        if len(epo_cost_time) <= 20:
            plt.xticks(range(1, len(epo_cost_time)+1, 1))
        else:
            plt.xticks(range(1, len(epo_cost_time)+1, 2))   

        plt.xlabel('Epoch')
        plt.ylabel('Cost Time (seconds)')
        plt.title(f'{time_png_name}')        
        plt.savefig(f'{exp_result_dir}/{time_png_name}.png')
        plt.close()        