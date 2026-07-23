import os
import numpy as np

from evaluator.map import MAP
from evaluator.fmeasure import FMeasure
from evaluator.ndcg import NDCG
from classes.data import get_all_data, get_rank_triples, get_topk_triples

def evaluation(dataset, k, model_name):
    ndcg_class = NDCG()
    fmeasure = FMeasure()
    m = MAP()

    if dataset.ds_name == "dbpedia":
        IN_SUMM = os.path.join(os.getcwd(), f'{model_name}/dbpedia')
        start = [0, 140]
        end   = [100, 165]
    elif dataset.ds_name == "lmdb":
        IN_SUMM = os.path.join(os.getcwd(), f'{model_name}/lmdb')
        start = [100, 165]
        end   = [140, 175]
    elif dataset.ds_name == "faces":
        IN_SUMM = os.path.join(os.getcwd(), f'{model_name}/faces')
        start = [0, 25]
        end   = [25, 50]

    all_ndcg_scores = []
    all_fscore = []
    all_penalized_fscore = []
    all_map_scores = []
    all_pred_lengths = []
    complete_count = 0
    empty_count = 0
    total_ndcg=0
    total_fscore=0
    total_map_score=0
    for i in range(start[0], end[0]):
        t = i+1
        print(t)
        gold_list_top, triples_dict, triple_tuples = get_all_data(dataset.db_path, t, k, dataset.file_n)
        #print("triples_dict", triples_dict)
        #rank_triples, encoded_rank_triples = get_rank_triples(IN_SUMM, t, k, triples_dict)
        topk_triples, encoded_topk_triples = get_topk_triples(IN_SUMM, t, k, triples_dict)
        print(f"#####{topk_triples}###########")
        #ndcg_score = ndcg_class.get_score(gold_list_top, encoded_rank_triples)
        ndcg_score = 0
        f_score = fmeasure.get_score(encoded_topk_triples, gold_list_top)
        penalized_f_score = fmeasure.get_penalized_score(encoded_topk_triples, gold_list_top, k)
        #map_score = m.get_map(encoded_rank_triples, gold_list_top)
        map_score = 0
        pred_len = len(encoded_topk_triples)
        all_pred_lengths.append(pred_len)
        if pred_len == k:
            complete_count += 1
        if pred_len == 0:
            empty_count += 1
        total_ndcg += ndcg_score
        all_ndcg_scores.append(ndcg_score)

        total_fscore += f_score
        all_fscore.append(f_score)
        all_penalized_fscore.append(penalized_f_score)

        all_map_scores.append(map_score)

    for i in range(start[1], end[1]):
        t = i+1
        print(t)
        gold_list_top, triples_dict, triple_tuples = get_all_data(dataset.db_path, t, k, dataset.file_n)
        #print("triples_dict", triples_dict)
        #rank_triples, encoded_rank_triples = get_rank_triples(IN_SUMM, t, k, triples_dict)
        topk_triples, encoded_topk_triples = get_topk_triples(IN_SUMM, t, k, triples_dict)
        print(f"#####{topk_triples}###########")
        #ndcg_score = ndcg_class.get_score(gold_list_top, encoded_rank_triples)
        ndcg_score = 0
        f_score = fmeasure.get_score(encoded_topk_triples, gold_list_top)
        penalized_f_score = fmeasure.get_penalized_score(encoded_topk_triples, gold_list_top, k)
        #map_score = m.get_map(encoded_rank_triples, gold_list_top)
        map_score = 0
        pred_len = len(encoded_topk_triples)
        all_pred_lengths.append(pred_len)
        if pred_len == k:
            complete_count += 1
        if pred_len == 0:
            empty_count += 1
        total_ndcg += ndcg_score
        all_ndcg_scores.append(ndcg_score)
        total_fscore += f_score
        all_fscore.append(f_score)
        all_penalized_fscore.append(penalized_f_score)
        all_map_scores.append(map_score)

    completion_rate = complete_count / len(all_fscore) if all_fscore else 0
    empty_rate = empty_count / len(all_fscore) if all_fscore else 0
    avg_pred_len = np.average(all_pred_lengths) if all_pred_lengths else 0
    f_avg = np.average(all_fscore) if all_fscore else 0
    penalized_f_avg = np.average(all_penalized_fscore) if all_penalized_fscore else 0
    ndcg_avg = np.average(all_ndcg_scores) if all_ndcg_scores else 0
    map_avg = np.average(all_map_scores) if all_map_scores else 0
    print("{}@top{}: F-Measure={}, penalized_F-Measure={}, NDCG={}, MAP={}, completion_rate={}, empty_rate={}, avg_pred_len={}".format(
        dataset.ds_name, k, f_avg, penalized_f_avg, ndcg_avg, map_avg, completion_rate, empty_rate, avg_pred_len
    ))
    return {
        "dataset": dataset.ds_name,
        "k": k,
        "f_measure": f_avg,
        "penalized_f_measure": penalized_f_avg,
        "ndcg": ndcg_avg,
        "map": map_avg,
        "completion_rate": completion_rate,
        "empty_rate": empty_rate,
        "avg_pred_len": avg_pred_len,
        "num_entities": len(all_fscore),
    }