import numpy as np
from utils.softmax import softmax

def get_events_from_windows(reconnaissance_detector, infection_detector, attack_detector, cle_train_windows_x):
    
    events = []
    for detector in [reconnaissance_detector, infection_detector, attack_detector]:
        malicious_class_score = detector.model.predict(cle_train_windows_x)  

        malicious_events_proportion = np.sum(np.array(malicious_class_score)>0.5)/len(malicious_class_score)
        print(f'proportion of malicious events tagged by {detector.modelname} is: { 100*malicious_events_proportion:.4f} %')
        
        events.append(malicious_class_score)

    benign_class_score = []    
    for i in range(events[0].shape[0]):
        reconnaissance_prob = events[0][i]
        infection_prob = events[1][i]
        attack_prob = events[2][i]
        
        not_reconnaissance_prob = 1-events[0][i]
        not_infection_prob = 1-events[1][i]
        not_attack_prob = 1-events[2][i]
        
        benign_prob = not_reconnaissance_prob * not_infection_prob * not_attack_prob
        benign_class_score.append(benign_prob)    
 
    benign_events_proportion = np.sum(np.array(benign_class_score)>0.5)/len(benign_class_score)
    print(f'proportion of events estimated to be benign is: { benign_events_proportion}')
    events.append(benign_class_score)

       
    events=np.array(events).squeeze()

    events = np.transpose(events)

    events = [softmax(e) for e in events]
    events=np.array(events)

    return events 