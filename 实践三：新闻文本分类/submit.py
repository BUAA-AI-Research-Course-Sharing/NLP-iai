import pandas as pd
import numpy as np
import fasttext

class Submit():
    def __init__(self,texts):
        self.model=fasttext.load_model("zty_0_fasttext.bin")
        test_df = pd.read_csv('train_set.csv', sep='\t', nrows=15000)
        test_df['test_text'] = test_df['text'].astype(str)
        # self.texts= pd.read_csv(filepath_or_buffer='test_set.csv', sep=',')["text"].values
        self.texts=texts

        print(len(self.texts))
        self.testDataFile = 'test_set.csv'

    def list_2_submission_csv(self,labels):
        np.array(labels)
        print(labels)
        save = pd.DataFrame(labels, columns=['label'])
        save.to_csv('./submission.csv', index=False, header=True)


    def test(self):
        labels=[]
        print(self.texts[0])
        for text in self.texts:
            text=str(text)
            label=self.model.predict(text)
            #print(label.astype)
            labels.append(label)
        print(len(labels))
        # labels=self.model.test(self.testDataFile)
        return self.list_2_submission_csv(labels)

if __name__ == '__main__':

    texts = []
    f = open("test_set.txt")
    text = f.readline()
    while text:
        text = f.readline()
        texts.append(text)
    submit = Submit(texts)
    submit.test()


