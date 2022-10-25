python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_name_or_path $1 \
  --fp16 \
  --per_device_eval_batch_size 1 \
  --dataset_name Tevatron/msmarco-passage/dev \
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



python -m tevatron.utils.format.convert_result_to_marco \
              --input $1/run.txt \
              --output $1/run.marco

python -m pyserini.eval.msmarco_passage_eval msmarco-passage-dev-subset $1/run.marco
python -m pyserini.eval.msmarco_passage_eval msmarco-passage-dev-subset $1/run.marco > $1/run.metrics
