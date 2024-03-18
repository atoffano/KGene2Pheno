import matplotlib.pyplot as plt
import numpy as np
import os

logfolder = '/media/chris/Projects_Space/People/ChristopheH/KG_training_logs/'

logfiles = [logfolder + f for f in os.listdir(logfolder) if os.path.isfile(os.path.join(logfolder, f))]

#batch256 = (0.93341463804245, 0.937005162239075, 0.92594712972641, 0.934109210968018, 0.942490339279175, 0.91936606168747, 0.940536260604858, 0.938044428825378, 0.942703425884247, 0.937364757061005)
#batch512 = (0.934360861778259, 0.940148770809174, 0.934914350509644, 0.929935216903687, 0.937228620052338, 0.932532548904419, 0.929817080497742, 0.937147378921509, 0.930270314216614, 0.941583037376404)
#batch1024 = (0.919590830802918, 0.921469748020172, 0.922738313674927, 0.921929657459259, 0.912085771560669, 0.916210830211639, 0.941396117210388, 0.914083242416382, 0.940843641757965, 0.934471905231476)
#batch2048 = (0.874998450279236, 0.863339364528656, 0.841649651527405, 0.873595893383026, 0.843623459339142, 0.848901391029358, 0.860749483108521, 0.817505180835724, 0.84239661693573, 0.823643684387207)
#batch3072 = (0.76633882522583, 0.727367162704468, 0.768322706222534, 0.804896235466003, 0.777098834514618, 0.778181135654449, 0.739316523075104, 0.747185826301575, 0.756267368793488, 0.791497766971588)
#batch4096 = (0.76564085483551, 0.71735155582428, 0.771580338478088, 0.704336166381836, 0.729005634784699, 0.755512833595276, 0.695951819419861, 0.73934018611908, 0.739422142505646, 0.712694704532623)

for element in logfiles:
    if not element.endswith('.md'):
        logfiles.remove(element)

#dictionnary to store training stats in {batch_size : { epochs : [scores] }}
stats_dic = {}

for file in logfiles:
    if file.endswith('.md'):
        with open(file, 'r') as f:
            for line in f:
                if "Hit@1 :" in line:
                    score = line.split('Hit@1 :')[-1].strip()
                elif "n_epochs :" in line:
                    epochs = line.split('n_epochs :')[-1].strip()
                elif "batch_size :" in line:
                    batch_size = line.split('batch_size :')[-1].strip()
            if batch_size in stats_dic.keys():
                if epochs in stats_dic[batch_size].keys():
                    stats_dic[batch_size][epochs] = (stats_dic[batch_size][epochs], score)
                else:
                    stats_dic[batch_size][epochs] = score
            else:
                stats_dic[batch_size] = {epochs : score}

for key, value in stats_dic.items():
    print(key, value)

def figure(dds):
    #batches = ('0', '256', '512', '1024', '2048', '3072', '4096') # 0 is added for display and name alignment to data
    #epochs = '60, 60, 60, 115, 150, 210'
    #Data = (batch256, batch512, batch1024, batch2048, batch3072, batch4096)

    batches = []
    epochs = []
    scores = []
    for key1, value1 in dds.items():
        batches.append(key1)
        score = []
        for key2, value2 in value1.items():
            epochs.append(key2)
            score.append(value2)
        scores.append(score)
    
    print(scores)
    print(type(scores))
    # plot
    plt.boxplot(e for e in scores)

    plt.axis((0,len(batches)+1,0,1))
    plt.xlabel('batch size')
    plt.xticks(np.arange(len(batches)), labels=(batches))
    plt.ylabel('Hit@1')
    #plt.text(0.5,0.05,'epochs : {}'.format(epochs))
    plt.title('Hit@1 by batch size with ComplEx embedding')
    plt.show()

figure(stats_dic)