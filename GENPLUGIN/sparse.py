from rank_bm25 import BM25Okapi
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
import string
import numpy as np
import json
from concurrent.futures import ThreadPoolExecutor
import json

# 文档预处理函数
def preprocess(text):
    # 移除标点符号，转小写，分词
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return word_tokenize(text)

# 计算每个用户查询的BM25得分
def process_query(query, bm25):
    tokenized_query = preprocess(query)
    doc_scores = bm25.get_scores(tokenized_query)
    top_indices = np.flip(doc_scores.argsort()[-1001:]).tolist()
    top_scores = doc_scores[top_indices].tolist()
    return list(zip(top_indices, top_scores))

# 批量处理并行
def process_queries_in_parallel(queries, bm25, num_workers=2000):
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # 并行处理每个用户的查询
        result = list(executor.map(lambda q: process_query(q, bm25), queries))
    return result


dataset = 'Beauty'
json_file = f'./data/{dataset}/{dataset}.item.json'
inter_file = f'./data/{dataset}/{dataset}.inter.json'
with open(json_file, 'r') as f:
    item = json.load(f)
with open(inter_file, 'r') as f:
    inter = json.load(f)
mode = 'train'
train_users = []
train_rag_user = []
val_rag_user = []
test_rag_user = []
train_index = 0
for uid, items in inter.items():
    train_items = items[:-2]
    
    for i in range(1, len(train_items)):
        text = ''
        history = items[:i]
        if len(history) > 20:
            history = history[-20:]
        for h in history:
            text += item[str(h)]['title'] + ' ' 
        train_users.append(text)
        train_index += 1
    train_rag_history = items[-20:-3]
    train_rag_text = ''
    for h in train_rag_history:
        train_rag_text += item[str(h)]['title'] + ' ' 
    train_rag_user.append(train_rag_text)
    val_rag_history = items[-20:-2]
    val_rag_text = ''
    for h in val_rag_history:
        val_rag_text += item[str(h)]['title'] + ' ' 
    val_rag_user.append(val_rag_text)
    test_rag_history = items[-20:-1]
    test_rag_text = ''
    for h in test_rag_history:
        test_rag_text += item[str(h)]['title'] + ' ' 
    test_rag_user.append(test_rag_text)
print(len(train_rag_user))
print(len(train_users))
print(len(val_rag_user))
print(len(test_rag_user))
save_file = f'./rag_need/{dataset}/train/user_train.json'
with open(save_file, 'w') as f:
    json.dump(train_users, f)
save_file = f'./rag_need/{dataset}/train/rag_user_train.json'
with open(save_file, 'w') as f:
    json.dump(train_rag_user, f)
save_file = f'./rag_need/{dataset}/val/rag_user_val.json'
with open(save_file, 'w') as f:
    json.dump(val_rag_user, f)
save_file = f'./rag_need/{dataset}/test/rag_user_test.json'

with open(save_file, 'w') as f:
    json.dump(test_rag_user, f)
    
    
# 文档集合
train_rag_user_file = f'./rag_need/{dataset}/train/rag_user_train.json'
train_user_file = f'./rag_need/{dataset}/train/user_train.json'

with open(train_rag_user_file, 'r') as f:
    train_rag_user = json.load(f)
with open(train_user_file, 'r') as f:
    train_user = json.load(f)

# 文本预处理
tokenized_corpus = [preprocess(doc) for doc in train_rag_user]

# 创建BM25模型
bm25 = BM25Okapi(tokenized_corpus)

# 并行处理查询
rag_user_index = process_queries_in_parallel(train_user, bm25)

# 保存训练数据的结果
save_file = f'./rag_need/{dataset}/train/rag_user_index.json'
with open(save_file, 'w') as f:
    json.dump(rag_user_index, f)

# 验证集处理
val_file = f'./rag_need/{dataset}/val/rag_user_val.json'
with open(val_file, 'r') as f:
    val_rag_user = json.load(f)

val_tokenized_corpus = [preprocess(doc) for doc in val_rag_user]
val_bm25 = BM25Okapi(val_tokenized_corpus)

# 并行处理验证集查询
rag_user_index = process_queries_in_parallel(val_rag_user, val_bm25)

# 保存验证集的结果
save_file = f'./rag_need/{dataset}/val/rag_user_index.json'
with open(save_file, 'w') as f:
    json.dump(rag_user_index, f)

# 测试集处理
test_file = f'./rag_need/{dataset}/test/rag_user_test.json'
with open(test_file, 'r') as f:
    test_rag_user = json.load(f)

test_tokenized_corpus = [preprocess(doc) for doc in test_rag_user]
test_bm25 = BM25Okapi(test_tokenized_corpus)

# 并行处理测试集查询
rag_user_index = process_queries_in_parallel(test_rag_user, test_bm25)

# 保存测试集的结果
save_file = f'./rag_need/{dataset}/test/rag_user_index.json'
with open(save_file, 'w') as f:
    json.dump(rag_user_index, f)
