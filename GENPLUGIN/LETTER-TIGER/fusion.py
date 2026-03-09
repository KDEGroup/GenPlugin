import numpy as np
import os
import argparse
import json
import tqdm
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Beauty', help='Beauty / Sports / Toys')
    parser.add_argument('--model_type', type=str, default='tiger', help='tiger / letter')
    return parser.parse_args()

def fusion_embedding(args):
    dataset = args.dataset
    model_type = args.model_type
    modes = ['test','val','train','aug_train']
    for mode in modes:
        
        folder_path = f'../LETTER-TIGER-RAR/{model_type}/{dataset}/{mode}/'

        
        file_list = os.listdir(folder_path)

        
        file_list_sorted = sorted(file_list, key=lambda x: int(x.split('.')[0]))

        
        user_emb_mean = []
        mask = []
        max_seq_length = 0  
        index = []
        text = []
        decoder = []
        
        for file in file_list_sorted:
            file_path = os.path.join(folder_path, file)
            
            
            if file.split('.')[1] == 'mean':
                temp_data = np.load(file_path)
                user_emb_mean.append(temp_data)
            
            elif file.split('.')[1] == 'text':
                temp_data = np.load(file_path)
                text.append(temp_data)
            elif file.split('.')[1] == 'decoder':
                temp_data = np.load(file_path)
                decoder.append(temp_data)
        
        user_emb_mean = np.concatenate(user_emb_mean, axis=0)
        
        text = np.concatenate(text, axis=0)
        decoder = np.concatenate(decoder, axis=0)
        
        np.save(f'../LETTER-TIGER-RAR/rag_need/{model_type}/{dataset}/{mode}/user_emb_mean.npy', user_emb_mean)
        
        np.save(f'../LETTER-TIGER-RAR/rag_need/{model_type}/{dataset}/{mode}/text_user_emb_mean.npy', text)
        np.save(f'../LETTER-TIGER-RAR/rag_need/{model_type}/{dataset}/{mode}/decoder_item_emb_mean.npy', decoder)
        print(mode, user_emb_mean.shape, decoder.shape, text.shape)
def rerank(args):
    dataset = args.dataset
    model_type = args.model_type
    modes = ['train','val','test']
    top_k = 50
    for mode in modes:
        sparse_index =  json.load(open(f"../LETTER-TIGER-RAR/rag_need/{model_type}/{dataset}/{mode}/rag_user_index.json"))
        cf_index = json.load(open(f"../LETTER-TIGER-RAR/rag_need/{model_type}/{dataset}/{mode}/item_retrival.json"))
        if mode == 'train':
            user_emb_mean_aug = np.load(f'../LETTER-TIGER-RAR/rag_need/{model_type}/{dataset}/aug_train/user_emb_mean.npy')
        user_emb_mean = np.load(f'../LETTER-TIGER-RAR/rag_need/{model_type}/{dataset}/{mode}/user_emb_mean.npy')
        reranked_index = []
        dict = {}
        print(len(sparse_index), len(cf_index), user_emb_mean.shape)
        for i in range(len(sparse_index)):
            sim_set = set()
            sparse = sparse_index[i][:50]
            cf = cf_index[i][:50]
            rerank = []

            
            for user in sparse:
                if user in cf and user not in sim_set:
                    rerank.append(user)
                    sim_set.add(user)

            
            remaining_candidates = [u for u in sparse + cf if u not in sim_set]
            remaining_candidates = np.array(remaining_candidates)
            if mode == 'train':
                
                current_emb = user_emb_mean_aug[i]
            else:
                current_emb = user_emb_mean[i]
            candidate_embs = user_emb_mean[remaining_candidates]
            remaining_candidates = remaining_candidates.tolist()
            
            norm_current = np.linalg.norm(current_emb) + 1e-10
            norm_candidates = np.linalg.norm(candidate_embs, axis=1) + 1e-10
            similarity = np.dot(candidate_embs, current_emb) / (norm_candidates * norm_current)

            
            sorted_idx = np.argsort(-similarity)  

            for idx in sorted_idx:
                candidate_user = remaining_candidates[idx]
                if candidate_user not in sim_set:
                    rerank.append(candidate_user)
                    sim_set.add(candidate_user)
                if len(rerank) >= top_k:
                    break

            
            if len(rerank) < top_k:
                all_candidates = sparse + cf
                for u in all_candidates:
                    if u not in sim_set:
                        rerank.append(u)
                        sim_set.add(u)
                    if len(rerank) >= top_k:
                        break

            reranked_index.append(rerank)
        dict['user_index'] = reranked_index
        
        save_path = f"../LETTER-TIGER-RAR/rag_need/{model_type}/{dataset}/{mode}/reranked_user_index.json"
        with open(save_path, 'w') as f:
            json.dump(dict['user_index'], f)

        print(f"Finished processing mode: {mode}, saved to {save_path}")



if __name__ == '__main__':
    args = parse_args()
    fusion_embedding(args)
    rerank(args)