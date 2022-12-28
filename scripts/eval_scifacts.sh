python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_name_or_path $1 \
  --fp16 \
  --per_device_eval_batch_size 1 \
  --dataset_name Tevatron/scifact/dev \
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

python -m pyserini.eval.trec_eval -c -mrecip_rank -mndcg_cut.10 -mrecall.10 datasets/scifact/dev_qrels.txt $1/run.trec > $1/run.metrics
python -m pyserini.eval.trec_eval -c -mrecall.20  datasets/scifact/dev_qrels.txt $1/run.trec >> $1/run.metrics
python -m pyserini.eval.trec_eval -c -mrecall.40   datasets/scifact/dev_qrels.txt $1/run.trec >> $1/run.metrics
python -m pyserini.eval.trec_eval -c -mrecall.60  datasets/scifact/dev_qrels.txt $1/run.trec >> $1/run.metrics
python -m pyserini.eval.trec_eval -c -mrecall.80  datasets/scifact/dev_qrels.txt $1/run.trec >> $1/run.metrics
python -m pyserini.eval.trec_eval -c -mrecall.100  datasets/scifact/dev_qrels.txt $1/run.trec >> $1/run.metrics
python -m pyserini.eval.trec_eval -c -mrecall.120  datasets/scifact/dev_qrels.txt $1/run.trec >> $1/run.metrics
python -m pyserini.eval.trec_eval -c -mrecall.140 datasets/scifact/dev_qrels.txt $1/run.trec >> $1/run.metrics
python -m pyserini.eval.trec_eval -c -mrecall.160 datasets/scifact/dev_qrels.txt $1/run.trec >> $1/run.metrics
python -m pyserini.eval.trec_eval -c -mrecall.180  datasets/scifact/dev_qrels.txt $1/run.trec >> $1/run.metrics
python -m pyserini.eval.trec_eval -c -mrecall.200  datasets/scifact/dev_qrels.txt $1/run.trec >> $1/run.metrics
cat $1/run.metrics