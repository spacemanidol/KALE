"""
This module computes evaluation metrics for MSMARCO dataset on the ranking task.
Command line:
python msmarco_eval_ranking.py <path_to_reference_file> <path_to_candidate_file>

Creation Date : 06/12/2018
Last Modified : 11/24/2022
Authors : Daniel Campos <dacamp@microsoft.com>, Rutger van Haasteren <ruvanh@microsoft.com>
"""
import re
import sys
import statistics
import json

from collections import Counter

MaxMRRRank = 200

def load_reference_from_stream(f):
    """Load Reference reference relevant passages
    Args:f (stream): stream to load.
    Returns:qids_to_relevant_passageids (dict): dictionary mapping from query_id (int) to relevant passages (list of ints). 
    """
    qids_to_relevant_passageids = {}
    for l in f:
        try:
            l = re.split('[\t\s]', l.strip())
            qid = int(l[0])
            if qid in qids_to_relevant_passageids:
                pass
            else:
                qids_to_relevant_passageids[qid] = []
            qids_to_relevant_passageids[qid].append(int(l[2]))
        except:
            raise IOError('\"%s\" is not valid format' % l)
    return qids_to_relevant_passageids

def load_reference(path_to_reference):
    """Load Reference reference relevant passages
    Args:path_to_reference (str): path to a file to load.
    Returns:qids_to_relevant_passageids (dict): dictionary mapping from query_id (int) to relevant passages (list of ints). 
    """
    with open(path_to_reference,'r') as f:
        qids_to_relevant_passageids = load_reference_from_stream(f)
    return qids_to_relevant_passageids

def load_candidate_from_stream(f):
    """Load candidate data from a stream.
    Args:f (stream): stream to load.
    Returns:qid_to_ranked_candidate_passages (dict): dictionary mapping from query_id (int) to a list of 1000 passage ids(int) ranked by relevance and importance
    """
    qid_to_ranked_candidate_passages = {}
    for l in f:
        try:
            l = l.strip().split('\t')
            qid = int(l[0])
            pid = int(l[1])
            rank = int(l[2])
            if qid in qid_to_ranked_candidate_passages:
                pass    
            else:
                # By default, all PIDs in the list of 1000 are 0. Only override those that are given
                tmp = [0] * 1000
                qid_to_ranked_candidate_passages[qid] = tmp
            qid_to_ranked_candidate_passages[qid][rank-1]=pid
        except:
            raise IOError('\"%s\" is not valid format' % l)
    return qid_to_ranked_candidate_passages
                
def load_candidate(path_to_candidate):
    """Load candidate data from a file.
    Args:path_to_candidate (str): path to file to load.
    Returns:qid_to_ranked_candidate_passages (dict): dictionary mapping from query_id (int) to a list of 1000 passage ids(int) ranked by relevance and importance
    """
    
    with open(path_to_candidate,'r') as f:
        qid_to_ranked_candidate_passages = load_candidate_from_stream(f)
    return qid_to_ranked_candidate_passages

def quality_checks_qids(qids_to_relevant_passageids, qids_to_ranked_candidate_passages):
    """Perform quality checks on the dictionaries

    Args:
    p_qids_to_relevant_passageids (dict): dictionary of query-passage mapping
        Dict as read in with load_reference or load_reference_from_stream
    p_qids_to_ranked_candidate_passages (dict): dictionary of query-passage candidates
    Returns:
        bool,str: Boolean whether allowed, message to be shown in case of a problem
    """
    message = ''
    allowed = True

    # Create sets of the QIDs for the submitted and reference queries
    candidate_set = set(qids_to_ranked_candidate_passages.keys())
    ref_set = set(qids_to_relevant_passageids.keys())

    # Check that we do not have multiple passages per query
    for qid in qids_to_ranked_candidate_passages:
        # Remove all zeros from the candidates
        duplicate_pids = set([item for item, count in Counter(qids_to_ranked_candidate_passages[qid]).items() if count > 1])

        if len(duplicate_pids-set([0])) > 0:
            message = "Cannot rank a passage multiple times for a single query. QID={qid}, PID={pid}".format(
                    qid=qid, pid=list(duplicate_pids)[0])
            allowed = False

    return allowed, message

def compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages,per_query_metric_filename):
    """Compute MRR metric
    Args:    
    p_qids_to_relevant_passageids (dict): dictionary of query-passage mapping
        Dict as read in with load_reference or load_reference_from_stream
    p_qids_to_ranked_candidate_passages (dict): dictionary of query-passage candidates
    Returns:
        dict: dictionary of metrics {'MRR': <MRR Score>}
    """
    all_scores = {}
    MRR = 0
    MRR_at_ten = 0
    recall_at_k = [0,0,0,0,0,0,0,0,0,0,0] #10,20,40,60,80,100,120,140,160,180,200
    qids_with_relevant_passages = 0
    ranking = []
    print(len(qids_to_ranked_candidate_passages))
    print(len(qids_to_relevant_passageids))
    with open(per_query_metric_filename,'w') as w:
        for qid in qids_to_ranked_candidate_passages:
            if qid in qids_to_relevant_passageids:
                ranking.append(0)
                target_pid = qids_to_relevant_passageids[qid]
                candidate_pid = qids_to_ranked_candidate_passages[qid]
                MRR_current = 0  
                MRR_at_ten_current = 0
                recall_at_k_current = [0,0,0,0,0,0,0,0,0,0,0]
                for i in range(0,MaxMRRRank):
                    if candidate_pid[i] in target_pid:
                        if i < 10:
                            MRR_at_ten += 1/(i + 1)
                            MRR_at_ten_current = 1/(i + 1)
                            recall_at_k[0] += 1
                            recall_at_k_current[0] = 1
                        if i < 20:
                            recall_at_k[1] += 1
                            recall_at_k_current[1] = 1
                        if i < 40:
                            recall_at_k[2] += 1
                            recall_at_k_current[2] = 1
                        if i < 60:
                            recall_at_k[3] += 1
                            recall_at_k_current[3] = 1
                        if i < 80:
                            recall_at_k[4] += 1
                            recall_at_k_current[4] = 1
                        if i < 100:
                            recall_at_k[5] += 1
                            recall_at_k_current[5] = 1
                        if i < 120:
                            recall_at_k[6] += 1
                            recall_at_k_current[6] = 1
                        if i < 140:
                            recall_at_k[7] += 1
                            recall_at_k_current[7] = 1
                        if i < 160:
                            recall_at_k[8] += 1
                            recall_at_k_current[8] = 1
                        if i < 180:
                            recall_at_k[9] += 1
                            recall_at_k_current[9] = 1
                        if i < 200:
                            recall_at_k[10] += 1
                            recall_at_k_current[10] = 1
                        MRR += 1/(i + 1)
                        MRR_current += 1/(i + 1)
                d = {'query_id':qid,'mrr@200':MRR_current,'mrr@10':MRR_at_ten_current,'recall@k':recall_at_k_current}
                w.write("{}\n".format(json.dumps(d)))
    num_qrels=len(qids_to_ranked_candidate_passages)
    MRR /= num_qrels
    MRR_at_ten /= num_qrels

    all_scores['MRR @200'] = MRR
    all_scores['MRR @10'] = MRR_at_ten
    all_scores['Recall @K'] = [recall_at_k[i]/num_qrels for i in range(len(recall_at_k))]
    
    all_scores['QueriesRanked'] = len(qids_to_ranked_candidate_passages)
    return all_scores
                
def compute_metrics_from_files(path_to_reference, path_to_candidate, per_query_metric_filename, perform_checks=True):
    """Compute MRR metric
    Args:    
    p_path_to_reference_file (str): path to reference file.
        Reference file should contain lines in the following format:
            QUERYID\tPASSAGEID
            Where PASSAGEID is a relevant passage for a query. Note QUERYID can repeat on different lines with different PASSAGEIDs
    p_path_to_candidate_file (str): path to candidate file.
        Candidate file sould contain lines in the following format:
            QUERYID\tPASSAGEID1\tRank
            If a user wishes to use the TREC format please run the script with a -t flag at the end. If this flag is used the expected format is 
            QUERYID\tITER\tDOCNO\tRANK\tSIM\tRUNID 
            Where the values are separated by tabs and ranked in order of relevance 
    Returns:
        dict: dictionary of metrics {'MRR': <MRR Score>}
    """
    
    qids_to_relevant_passageids = load_reference(path_to_reference)
    qids_to_ranked_candidate_passages = load_candidate(path_to_candidate)
    if perform_checks:
        allowed, message = quality_checks_qids(qids_to_relevant_passageids, qids_to_ranked_candidate_passages)
        if message != '': print(message)

    return compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages,per_query_metric_filename)

def main():
    """Command line:
    python msmarco_eval_ranking.py <path_to_reference_file> <path_to_candidate_file>
    """

    if len(sys.argv) == 4:
        path_to_reference = sys.argv[1]
        path_to_candidate = sys.argv[2]
        per_query_metric_filename = sys.argv[3]
        metrics = compute_metrics_from_files(path_to_reference, path_to_candidate,per_query_metric_filename)
        print('#####################')
        for metric in sorted(metrics):
            print('{}: {}'.format(metric, metrics[metric]))
        print('#####################')

    else:
        print('Usage: msmarco_eval_ranking.py <reference ranking> <candidate ranking> <per query score filename>')
        exit()
    
if __name__ == '__main__':
    main()

