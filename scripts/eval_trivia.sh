python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_name_or_path $1 \
  --per_device_eval_batch_size 1 \
  --dataset_name Tevatron/wikipedia-trivia/test \
  --encoded_save_path $1/query_embed.pkl \
  --encode_is_qry \
  --use_onnx \
  --onnx_filepath $1/query-encoder.onnx

python -m tevatron.faiss_retriever \
--query_reps $1/query_embed.pkl \
--passage_reps $2/'corpus_emb.*.pkl' \
--depth 200 \
--batch_size -1 \
--save_text \
--save_ranking_to $1/run.txt


python -m tevatron.utils.format.convert_result_to_trec \
              --input $1/run.txt \
              --output $1/run.trec


python -m pyserini.eval.convert_trec_run_to_dpr_retrieval_run \
              --topics dpr-trivia-test \
              --index wikipedia-dpr \
              --input $1/run.trec \
              --output $1/run.json

python -m pyserini.eval.evaluate_dpr_retrieval \
                --retrieval $1/run.json \
                --topk 20 40 60 80 100 120 140 160 180 200 > $1/run.metrics


python -m pyserini.eval.evaluate_dpr_retrieval \
                --retrieval $1/run.json \
                --topk 20 40 60 80 100 120 140 160 180 200
