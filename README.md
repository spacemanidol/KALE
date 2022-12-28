# KALE

for s in $(seq -f "%02g" 0 19); do python -m tevatron.driver.encode   --output_dir=temp   --cache_dir cache --model_name_or_path $model/passage_model   --fp16   --per_device_eval_batch_size 156   --dataset_name Tevatron/wikipedia-nq-corpus   --encoded_save_path $model/corpus_emb.$s.pkl   --encode_num_shard 20   --encode_shard_index $s; done
python -m tevatron.driver.encode --output_dir=temp --model_name_or_path $model/query_model --fp16 --per_device_eval_batch_size 156  --dataset_name Tevatron/wikipedia-nq/test  --encoded_save_path $model/query_emb.pkl  --encode_is_qry
python -m tevatron.faiss_retriever --query_reps $model/query_emb.pkl --passage_reps $model/'corpus_emb.*.pkl' --depth 200 --batch_size -1 --save_text --save_ranking_to $model/run.nq.test.txt
python -m tevatron.utils.format.convert_result_to_trec               --input $model/run.nq.test.txt               --output $model/run.nq.test.trec
python -m pyserini.eval.convert_trec_run_to_dpr_retrieval_run               --topics dpr-nq-test               --index wikipedia-dpr               --input $model/run.nq.test.trec               --output $model/run.nq.test.json
python -m pyserini.eval.evaluate_dpr_retrieval                 --retrieval $model/run.nq.test.json                 --topk 20 40 60 80 100 120 140 160 180 200
python -m pyserini.eval.evaluate_dpr_retrieval                 --retrieval $model/run.nq.test.json                 --topk 20 40 60 80 100 120 140 160 180 200 > $model/run.metrics


sh scripts/eval_squad.sh  models/squad/struct/1/ models/squad/bert-base/
sh scripts/eval_squad.sh  models/squad/struct/2/ models/squad/bert-base/
sh scripts/eval_squad.sh  models/squad/struct/3/ models/squad/bert-base/
sh scripts/eval_squad.sh  models/squad/struct/6/ models/squad/bert-base/
sh scripts/eval_squad.sh  models/squad/struct/9/ models/squad/bert-base/
sh scripts/eval_squad.sh  models/squad/struct/12/ models/squad/bert-base/

sh scripts/eval_squad.sh  models/squad/kale/1layer models/squad/bert-base/
sh scripts/eval_squad.sh  models/squad/kale/1layer-long models/squad/bert-base/
sh scripts/eval_squad.sh  models/squad/kale/1layer-short models/squad/bert-base/
sh scripts/eval_squad.sh  models/squad/kale/1layer-long-temp-10-bs4 models/squad/bert-base/

sh scripts/eval_squad.sh  models/squad/kale/2layer models/squad/bert-base/
sh scripts/eval_squad.sh  models/squad/kale/2layer-long models/squad/bert-base/
sh scripts/eval_squad.sh  models/squad/kale/2layer-short models/squad/bert-base/
sh scripts/eval_squad.sh  models/squad/kale/2layer-long-temp-10-bs4 models/squad/bert-base/

sh scripts/eval_squad.sh  models/squad/kale/3layer models/squad/bert-base/
sh scripts/eval_squad.sh  models/squad/kale/3layer-long models/squad/bert-base/
sh scripts/eval_squad.sh  models/squad/kale/3layer-short models/squad/bert-base/
sh scripts/eval_squad.sh  models/squad/kale/3layer-long-temp-10-bs4 models/squad/bert-base/

sh scripts/eval_squad.sh  models/squad/kale/6layer models/squad/bert-base/
sh scripts/eval_squad.sh  models/squad/kale/6layer-long models/squad/bert-base/
sh scripts/eval_squad.sh  models/squad/kale/6layer-short models/squad/bert-base/
sh scripts/eval_squad.sh  models/squad/kale/6layer-long-temp-10-bs4 models/squad/bert-base/

sh scripts/eval_squad.sh  models/squad/kale/9layer models/squad/bert-base/
sh scripts/eval_squad.sh  models/squad/kale/9layer-long models/squad/bert-base/
sh scripts/eval_squad.sh  models/squad/kale/9layer-short models/squad/bert-base/
sh scripts/eval_squad.sh  models/squad/kale/9layer-long-temp-10-bs4 models/squad/bert-base/