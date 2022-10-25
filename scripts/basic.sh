export CUDA_VISIBLE_DEVICES=0
cp -r $1/*.json $1/query_model/
cp -r $1/*.json $1/passage_model/
cp -r $1/*.txt $1/query_model/
cp -r $1/*.txt $1/passage_model/


for s in $(seq -f "%02g" 0 19)
do
python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_name_or_path $1/passage_model \
  --fp16 \
  --per_device_eval_batch_size 156 \
  --dataset_name Tevatron/wikipedia-nq-corpus \
  --encoded_save_path $1/corpus_emb.$s.pkl \
  --encode_num_shard 20 \
  --encode_shard_index $s
done


python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_name_or_path $1/query_model \
  --fp16 \
  --per_device_eval_batch_size 156 \
  --dataset_name Tevatron/wikipedia-nq/test \
  --encoded_save_path $1/query_emb.pkl \
  --encode_is_qry



python -m tevatron.faiss_retriever \
--query_reps $1/query_emb.pkl \
--passage_reps 'corpus_emb.*.pkl' \
--depth 200 \
--batch_size -1 \
--save_text \
--save_ranking_to $1/run.nq.test.txt


python -m tevatron.utils.format.convert_result_to_trec \
              --input $1/run.nq.test.txt \
              --output $1/run.nq.test.trec


python -m pyserini.eval.convert_trec_run_to_dpr_retrieval_run \
              --topics dpr-nq-test \
              --index wikipedia-dpr \
              --input $1/run.nq.test.trec \
              --output $1/run.nq.test.json

python -m pyserini.eval.evaluate_dpr_retrieval \
                --retrieval $1/run.nq.test.json \
                --topk 20 40 60 80 100 120 140 160 180 200

python -m pyserini.eval.evaluate_dpr_retrieval \
                --retrieval $1/run.nq.test.json \
                --topk 20 40 60 80 100 120 140 160 180 200 > $1/run.metrics