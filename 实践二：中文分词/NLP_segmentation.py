from tqdm import tqdm
import numpy as np


num_dic=['0','1','2','3','4','5','6','7','8','9','零','一','二','三','四','五','六','七','八','九','十','○','.']
num_dic=set(num_dic)
def is_date(piece):
    if piece[-1] == '年' or piece[-1] == '月' or piece[-1] == '日' or piece[-1] =='时' or piece[-1] =='分' :
        for i in iter(piece[:len(piece)-1]):
            if i not in num_dic:
                return 0
    else: return 0

    return 1


def evaluate(seg_results,seg_groundtruths):
    cnt_right=0
    cnt_seg_result=0
    cnt_seg_groundtruth=0
    for seg_result,seg_groundtruth in zip(seg_results,seg_groundtruths):
        set_result=set(seg_result)
        set_seg_groundtruth=set(seg_groundtruth.split())
        intersection=[x  for x in seg_result if x in seg_groundtruth.split()]  #交集
        cnt_right+=len(intersection)
        cnt_seg_groundtruth+=len(seg_groundtruth.split())
        cnt_seg_result+=len(seg_result)
        #print(cnt_right,cnt_seg_groundtruth,cnt_seg_result)

    P=round((cnt_right/cnt_seg_result)*100,6)
    R=round((cnt_right/cnt_seg_groundtruth)*100,6)
    F_1=round((2*P*R/(P+R)),6)
    return P,R,F_1
class MinSeg(object):
    '''
    最小分词法（最短路径法）

    算法思想：顺序构造有向无环图（邻接表），对句子进行遍历\\
              点从0开始到最后，对所有词典内的词设置一条有向边
              最后输出分次数最少的切分结果
    '''

    def __init__(self,texts,dic_words):
        self.texts=texts
        self.dic=dic_words
    def minseg_func(self):
        minseg_result_final=[]
        piece=None
        for text in tqdm(texts):
            adj_table=np.zeros((len(text)+1,len(text)+1))
            queue=[]
            visited=[0 for i in range(len(text)+1)]
            for i in range(len(text)):
                for j in range(i+1,len(text)+1):
                    piece=text[i:j]


                    if piece in self.dic or is_date(piece):
                        adj_table[i][j]=1
                    if i+1==j:
                        adj_table[i][j]=1
            #print(adj_table)
            #print(adj_table.shape)
            #找最短路径
            d=[len(text) for i in range(len(text)+1)]
            d[0]=0
            pre=[i-1 for i in range(len(text)+1)]
            #print(pre)
            s=0
            #print(len(text))
            while s != len(text)-1:
                for k in range(len(text)+1):
                    if adj_table[s][k]==1 and visited[s]==0:
                        if d[k]>d[s]+1:
                            pre[k]=s
                            d[k]=d[s]+1

                        #print(k)
                        queue.append(k)
                visited[s]=1
                #print(queue)
                s = queue.pop(0)
            result=[]
            s=len(text)
            while s!=0:
                #print(s)
                result.append([pre[s],s])
                s=pre[s]
            result.reverse()
            #print(result)
            minseg_result=[]
            for i in result:
                minseg_result.append(text[i[0]:i[1]])
            minseg_result_final.append(minseg_result)
        return  minseg_result_final

class BiDirectctionMatchingMethod(object):
    """
    双向最大匹配法

    算法思想:
        1. 如果正反向分词结果词数不同，则取分词数量较少的那个
        2. 如果分词结果词数相同：
            2.1 分词结果相同，说明没有歧义，可返回任意一个
            2.2 分词结果不同，返回其中单字较少的那个

    Attribute:
        window_size: 机器词典最长词条字符数
        dic: 机器词典
        text: 需要匹配的字符串(文本)
    """
    def __init__(self, texts,dic_words,seg_groundtruths):

        self.dic = dic_words
        self.texts=texts
        self.seg_groundtturhs=seg_groundtruths
        self.window_size=23#最长词典词长度

    def FMM_cut(self):
        """
        正向最大匹配法的方法
        算法思想:
        1. 从左向右取待切分汉语句的m个字符作为匹配字符, m为机器词典中最长词条的字符数
        2. 查找机器词典并进行匹配，若匹配成功, 则将这个匹配字段作为一个词切分出来。
           若匹配不成功, 则将这个匹配字段的最后一个字去掉, 剩下的字符串作为新的匹配字段,
           进行再次匹配, 重复以上过程, 直到切分出所有词为止。
        :return FMM_result: 正向最大匹配法匹配结果
        """
        FMM_result_all = []
        for text in self.texts:
            FMM_result = []
            FMM_text_length = len(text)
            FMM_piece = None
            FMM_result=[]
            FMM_index=0
            while FMM_index < FMM_text_length:
                for j in range(min(len(text),self.window_size) + FMM_index , FMM_index, -1):
                    # 每一轮循环从新的字符串的"索引位置(起始位置) + 机器词典中最长的词条字符数"位置开始匹配字符
                    # 如果这一轮循环匹配失败，则将要匹配的字符数进行-1操作，进行新一轮的匹配
                    # 最后一轮匹配为一个字符匹配
                    FMM_piece = text[FMM_index: j]
                    if FMM_piece in self.dic or is_date(FMM_piece):
                        # 如果这串字符在机器词典中，那么移动索引至匹配了的字符串的最后一个字符的下标处(将匹配了的字符串移出这个线性表)
                        FMM_index = j - 1
                        break
                # 将索引移动到下一轮匹配的开始字符位置，即如果匹配成功，将之前成功匹配的字符移除线性表
                # 如果匹配失败，则是将第一个字符移除线性表
                FMM_index +=1
                FMM_result.append(FMM_piece)
            FMM_result_all.append(FMM_result)
        return FMM_result_all
    def RMM_cut(self):
        """
逆向最大匹配法
RMM的算法思想:
1.
先将文档进行倒排处理(reverse)，生成逆序文档，然后根据逆序词典，对逆序文档用正向最大匹配法处理
2.
从左向右取待切分汉语句的m个字符作为匹配字符, m为机器词典中最长词条的字符数
3.
查找机器词典并进行匹配，若匹配成功, 则将这个匹配字段作为一个词切分出来。
若匹配不成功, 则将这个匹配字段的最后一个字去掉, 剩下的字符串作为新的匹配字段,
进行再次匹配, 重复以上过程, 直到切分出所有词为止。
该应用的算法思想:
没有使用reverse处理，而是直接从后向前匹配，只是匹配的结果进行了reverse处理
(因为匹配的结果第一个是"起源"，最后一个是"研究")
        :return RMM_result: 逆向最大匹配法匹配结果
        """
        RMM_result_all = []
        for text in self.texts:
            RMM_result = []
            RMM_index = len(text)
            RMM_piece = None
            while RMM_index > 0:
                # RMM的循环
                for size in range(max(0,RMM_index - self.window_size), RMM_index):
                    # 匹配最后的3个字符串，如果匹配就进行下一轮while循环，否则字符数-1，进行下一轮for循环
                    RMM_piece = text[size: RMM_index]
                    if RMM_piece in self.dic or is_date(RMM_piece):
                        # 如果这串字符在机器词典中，那么移动索引至成功匹配的第一个字符的下标处(将匹配了的字符串移出这个线性表)
                        RMM_index = size + 1
                        break
                # 将索引移动到下一轮匹配的开始字符位置，即如果匹配成功，将之前成功匹配的字符移除线性表
                # 如果匹配失败，则是将最后一个字符移除线性表
                RMM_index -= 1
                RMM_result.append(RMM_piece)

            RMM_result.reverse()
            RMM_result_all.append(RMM_result)
        return RMM_result_all

    def get_best_matching_result(self,FMM_result_all,RMM_result_all):
        """
        :param FMM_result_all: 正向最大匹配法的分词结果
        :param RMM_result_all: 逆向最大匹配法的分词结果
        :return:
            1.词数不同返回词数较少的那个
            2.词典结果相同，返回任意一个(MM_result)
            3.词数相同但是词典结果不同，返回单字最少的那个
        """
        MM_result_final=[]
        for MM_result , RMM_result in zip(FMM_result_all,RMM_result_all):
            if len(MM_result) != len(RMM_result):
                # 如果两个结果词数不同，返回词数较少的那个
                MM_result_final.append(MM_result if (len(MM_result) < len(RMM_result)) else RMM_result)
            else:
                if MM_result == RMM_result:
                    # 因为RMM的结果是取反了的，所以可以直接匹配
                    # 词典结果相同，返回任意一个
                    MM_result_final.append(MM_result)
                else:
                    # 词数相同但是词典结果不同，返回单字最少的那个
                    MM_word_1 = 0
                    RMM_word_1 = 0
                    for word in MM_result:
                        # 判断正向匹配结果中单字出现的词数
                        if len(word) == 1:
                            MM_word_1 += 1
                    for word in RMM_result:
                        # 判断逆向匹配结果中单字出现的词数
                        if len(word) == 1:
                            RMM_word_1 += 1
                    MM_result_final.append(MM_result if (MM_word_1 < RMM_word_1) else RMM_result)
        return MM_result_final


if __name__ == '__main__':

    # 数据集读取
    f_train=open("./词典/pku_training_words.utf8","r",encoding='utf-8')
    dic_words=f_train.read().splitlines()
    dic_words=frozenset(dic_words)
    #print(dic_words)
    f_texts = open("./待分词文件/corpus.txt", "r", encoding='utf-8')
    texts = f_texts.read().splitlines()
    #print(texts)
    f_seg_groundtruths = open("./分词对比文件/gold.txt", "r", encoding='utf-8')
    seg_groundtruths = f_seg_groundtruths.read().splitlines()
    #print(seg_groundtruths)
    f_stop_words = open("./停用词/stop_word.txt", "r", encoding='utf-8')
    stop_words = f_stop_words.read().splitlines()
    #print(stop_words)
    #数据处理
    minseg=MinSeg(texts,dic_words)
    minseg_results=minseg.minseg_func()
    P,R,F_1=evaluate(minseg_results,seg_groundtruths)
    #print(minseg_results)
    #print(seg_groundtruths)
    print('最少分词法:','Precision:',str(P)+'%','Recall ratio:',str(R)+'%','F1-Measure',str(F_1)+'%')
    BIM = BiDirectctionMatchingMethod(texts, dic_words, seg_groundtruths)
    FMM_result_all = BIM.FMM_cut()
    RMM_result_all = BIM.RMM_cut()
    MM_result_final = BIM.get_best_matching_result(FMM_result_all, RMM_result_all)
    P, R, F_1 = evaluate(MM_result_final, seg_groundtruths)
    print('双向匹配法:','Precision:',str(P)+'%','Recall ratio:',str(R)+'%','F1-Measure',str(F_1)+'%')


    # FMM=open('FMM_result.txt','w')
    # print(FMM_result_all, file=FMM)
    # FMM.close()
    # print('finish fmm')
    # RMM = open('RMM_result.txt', 'w')
    # print(RMM_result_all,file=RMM)
    # RMM.close()
    # print('finish rmm')
    # MM = open('MM_result_final.txt', 'w')
    # print(MM_result_final, file=MM)
    # MM.close()
