from sklearn.naive_bayes import MultinomialNB
import codecs
from sklearn.feature_extraction.text import TfidfVectorizer
test_headers = []
test_text = []
with codecs.open('C:\\Users\\green\\workspace\\Bayes_News\\src\\news_test.txt', 'r', encoding='utf-8') as input_file:
    for line in input_file:
        columns =list(map(lambda x: x.strip(), line.strip().split('\t')))
        test_headers.append(columns[0])
        if len(columns) == 1:
            test_text.append('')
        else:
            test_text.append(columns[1])
train_headers = []
train_text = []
labels = []
with codecs.open('C:\\Users\\green\\workspace\\Bayes_News\\src\\news_train.txt', 'r', encoding='utf-8') as input_file:
    for line in input_file:
        columns = list(map(lambda x: x.strip(), line.strip().split('\t')))
        labels.append(columns[0])
        train_headers.append(columns[1])
        if len(columns) == 2:
            train_text.append('')
        else:
            train_text.append(columns[2])
exp_list = []
for i in range(len(train_headers)):
    exp_list.append(train_headers[i] + train_text[i])
for i in range(len(test_text)):
    exp_list.append(test_headers[i] + test_text[i])
vectorizer = TfidfVectorizer(min_df = 10)
matrix_exp = vectorizer.fit_transform(exp_list)
clf = MultinomialNB()
clf.fit(matrix_exp[:60000],labels)
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
preds = clf.predict(matrix_exp[-15000:])
helpful_str = ''
for i in range(len(preds)):
    helpful_str = helpful_str + '' + preds[i] + '\n'
def write_answer(auc):
    with open("news_output.txt", "w") as fout:
        fout.write(auc)
write_answer(helpful_str)