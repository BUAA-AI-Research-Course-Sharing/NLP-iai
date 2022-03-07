import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import NLP_segmentation as seg

stopwords = [line.strip() for line in open('./停用词/stop_word.txt', 'r', encoding='utf-8')]
#print(stopwords)
with open('./中国共产党第十九届中央委员会第六次全体会议公报.txt', 'r', encoding='utf-8') as f:
    txt = f.readlines()
    #print(txt)
    # 将jieba分词换成你所实现的分词算法
    #words = jieba.cut(txt)
    f_train = open("./词典/pku_training_words.utf8", "r", encoding='utf-8')
    dic_words = f_train.read().splitlines()
    dic_words = frozenset(dic_words)

    BIM = seg.BiDirectctionMatchingMethod(txt, dic_words, None)
    FMM_result_all = BIM.FMM_cut()
    RMM_result_all = BIM.RMM_cut()
    seg_result = BIM.get_best_matching_result(FMM_result_all, RMM_result_all)
    sentences = ""
    for words in seg_result:
        for word in words:
            if word in stopwords:
                continue
            sentences += str(word) + ' '
    # 生成词云就这一步
    wordcloud = WordCloud(background_color='white',
                          font_path="C:\Windows\Fonts\STXINGKA.TTF",
                          width=2000,
                          height=2000, ).generate(sentences)
    # 输出词云图片，自行学习matplotlib.pyplot如何使用
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.savefig("中国共产党第十九届中央委员会第六次全体会议公报_修改停用词表后")
    plt.show()