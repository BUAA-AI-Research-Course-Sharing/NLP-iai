import json
import numpy as np
import random
from seqeval.metrics import classification_report

'''
读取train.json和dev.json文件,返回字典格式
'''
dev_file = open("./dataset/dev.json", 'r', encoding='utf-8')
dev_set = []
for line in dev_file.readlines():
    dic = json.loads(line)
    dev_set.append(dic)
train_file = open("./dataset/train.json", 'r', encoding='utf-8')
train_set = []
for line in train_file.readlines():
    dic = json.loads(line)
    train_set.append(dic)




'''
labels_dic和label_start是为了方便BIO标注而创建的字典.
'''
print(len(dev_set),len(train_set))
id2label={0:'O',1:'B-ADDRESS', 2:'I-ADDRESS', 3:'B-BOOK', 4:'I-BOOK',
            5:'B-COMPANY',6:'I-COMPANY',7:'B-GAME', 8:'I-GAME',
            9: 'B-GOVERNMENT', 10:'I-GOVERNMENT', 11:'B-MOVIE', 12:'I-MOVIE',
            13:'B-NAME',14: 'I-NAME', 15:'B-ORGANIZATION', 16:'I-ORGANIZATION',
            17:'B-POSITION',18:'I-POSITION', 19:'B-SCENE',20: 'I-SCENE'}

label_start={'address':1,'book':3,'company':5,'game':7,'government':9,
             'movie':11,'name':13,'organization':15,'position':17,'scene':19}

#将id2label的key和value两两交换
label2id= dict(zip(id2label.values(), id2label.keys()))

'''
数据预处理:
对数据进行BIO标注并构建列表word2id.
'''
bio_train_tag_lists=[]
word_lists=[]
id2word=[]
for i in range(10748):  #10748是数据集的数据长度
    bio_tag=[]#用'O'初始化bio_tag
    for k in range(len(train_set[i]['text'])):
        bio_tag.append('O')
        if train_set[i]['text'][k] not in id2word:
            id2word.append(train_set[i]['text'][k])
    word_lists.append(train_set[i]['text'])
    for j in range(len(train_set[i]['label'])):
        entity_label=list(train_set[i]['label'].keys())[j]
        for k in range(len(list(train_set[i]['label'][entity_label].keys()))):
            entity=list(train_set[i]['label'][entity_label].keys())[k]
            entity_range=train_set[i]['label'][entity_label][entity][0]
            bio_tag[entity_range[0]]=id2label[label_start[entity_label]]
            for o in range(entity_range[0]+1 , entity_range[1]+1):
                bio_tag[o] = id2label[label_start[entity_label]+1]
    bio_train_tag_lists.append(bio_tag)

bio_dev_tag_lists=[]
dev_word_lists=[]
for i in range(1343):  #10748是数据集的数据长度
    bio_tag=[]#用'O'初始化bio_tag
    for k in range(len(dev_set[i]['text'])):
        bio_tag.append('O')
        if dev_set[i]['text'][k] not in id2word:
            id2word.append(dev_set[i]['text'][k])
    dev_word_lists.append(dev_set[i]['text'])
    for j in range(len(dev_set[i]['label'])):
        entity_label=list(dev_set[i]['label'].keys())[j]
        #print("changdu:",train_set[i]['label'][entity_label].keys())
        for k in range(len(list(dev_set[i]['label'][entity_label].keys()))):
            entity=list(dev_set[i]['label'][entity_label].keys())[k]
            entity_range=dev_set[i]['label'][entity_label][entity][0]
            bio_tag[entity_range[0]]=id2label[label_start[entity_label]]
            for o in range(entity_range[0]+1 , entity_range[1]+1):
                bio_tag[o] = id2label[label_start[entity_label]+1]
    bio_dev_tag_lists.append(bio_tag)
word2id={}
for i,j in zip(range(len(id2word)),id2word):
    word2id[j]=i
#print(word2id)
#print(word_lists)
'''
data=open("./data.txt",'w+')
for i in range(10748):
    print(bio_train_tag_lists[i],'\n', file=data)
data.close()
'''
### 完成数据标注后开始初始化

class HMM(object):
    def __init__(self, N, M):
        """Args:
            N: 状态数，这里对应存在的标注的种类
            M: 观测数，这里对应有多少不同的字
        """
        self.N = N
        self.M = M
        # 状态转移概率矩阵 A[i][j]表示从i状态转移到j状态的概率
        self.A = np.zeros([N, N])
        # 观测概率矩阵, B[i][j]表示i状态下生成j观测的概率
        self.B = np.zeros([N, M])
        # 初始状态概率  Pi[i]表示初始时刻为状态i的概率
        self.Pi = np.zeros(N)

    #创建并训练HMM的模型参数Pi(初始状态概率矩阵),A(状态转移矩阵),B(发射矩阵)
    def train(self,word_lists,tag2id,bio_train_tag_lists,word2id):

        for bio_tag in bio_train_tag_lists:
            for i in range(len(bio_tag)-1):
                self.A[tag2id[bio_tag[i]]][tag2id[bio_tag[i+1]]] += 1
        self.A[self.A == 0] = 1e-10
        self.A=self.A/self.A.sum(axis=1, keepdims=True)
        #print(self.A)
        # 估计发射矩阵
        for bio_tag, word_list in zip(bio_train_tag_lists, word_lists):
            for tag, word in zip(bio_tag, word_list):
                tag_id = tag2id[tag]
                word_id = word2id[word]
                self.B[tag_id][word_id] += 1
        self.B[self.B == 0.] = 1e-10
        self.B = self.B / self.B.sum(axis=1, keepdims=True)

        # 估计初始状态概率
        for bio_tag in bio_train_tag_lists:
            init_tagid = tag2id[bio_tag[0]]
            self.Pi[init_tagid] += 1
        self.Pi[self.Pi == 0.] = 1e-10
        self.Pi = self.Pi / self.Pi.sum()
        return self.A,self.B,self.Pi

    def Viterbi_predict(self,word_lists,tag2id,id2tag,word2id):

        for word_list in word_lists:
            # 初始化
            A = np.log(self.A)
            B = np.log(self.B)
            Pi = np.log(self.Pi)

            vtb=np.zeros([self.N,len(word_list)])
            back_pointer=np.zeros([self.N,len(word_list)])
            BT=B.T
            if word_list[0] not in id2word:
                # 如果字不再字典里，则假设状态的概率分布是均匀的
                bt = np.log(np.ones(self.N) / self.N)
            else:
                bt = BT[word2id[word_list[0]]]
            vtb[:, 0] = Pi + bt
            back_pointer[:,0]=-1

            # 开始递推
            # viterbi[tag_id, step]
            # = max(viterbi[:, step-1]* self.A.t()[tag_id]*BT[wordid])
            # 其中word是step时刻对应的字
            # 由上述递推公式求后续各步
            # bt代表每一步中的发射矩阵(当从不同状态下观测为word_id的概率)
            for step in range(1,len(word_list)):
                if word_list[step] not in id2word:
                    bt = np.log(np.ones(self.N) / self.N)
                    # 如果字不再字典里，则假设状态的概率分布是均匀的
                else:
                    bt = BT[word2id[word_list[step]]]
                for tag_id in range(len(tag2id)):
                    max_prob=np.max(vtb[:,step-1]+A[:,tag_id])
                    a=vtb[:,step-1]+A[:,tag_id]
                    max_id=np.argmax(a)
                    vtb[tag_id,step]=max_prob+bt[tag_id]
                    back_pointer[tag_id,step]=max_id
            last_best_prob=np.max(vtb[:,len(word_list)-1])
            last_best_id=np.argmax(vtb[:,len(word_list)-1])
            #print(last_best_prob,last_best_id)


            #回溯求最优路径
            best_id=int(last_best_id)

            best_path_id=[last_best_id]
            for back_step in range(len(word_list)-1,0,-1):
                s=back_pointer[best_id][back_step]
                s=int(s)
                best_path_id.append(s)
                best_id=s

            best_path_id.reverse()
            best_path_tag=[]
            for i in range(len(best_path_id)):
                best_path_tag.append(id2tag[best_path_id[i]])

            #print(best_path_tag,file=f)
            bio_tag_predict.append(best_path_tag)




if __name__== "__main__":
    bio_tag_predict = []
    HMM_model = HMM(21, len(word2id))
    A, B, Pi = HMM_model.train(word_lists, label2id, bio_train_tag_lists, word2id)
    HMM_model.Viterbi_predict(dev_word_lists,label2id,id2label,word2id)
    print(classification_report(bio_dev_tag_lists,bio_tag_predict))
