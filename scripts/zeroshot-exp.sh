sh scripts/zero-shot.sh models/nq/bert-base models/nq/struct/12-oneshot 12
sh scripts/zero-shot.sh models/nq/bert-base models/nq/struct/9-oneshot 9
sh scripts/zero-shot.sh models/nq/bert-base models/nq/struct/6-oneshot 6
sh scripts/zero-shot.sh models/nq/bert-base models/nq/struct/3-oneshot 3
sh scripts/zero-shot.sh models/nq/bert-base models/nq/struct/2-oneshot 2
sh scripts/zero-shot.sh models/nq/bert-base models/nq/struct/1-oneshot 1
sh scripts/zero-shot-quant.sh models/nq/bert-base models/nq/quantized/12-oneshot 12
sh scripts/zero-shot-quant.sh models/nq/bert-base models/nq/quantized/9-oneshot 9
sh scripts/zero-shot-quant.sh models/nq/bert-base models/nq/quantized/6-oneshot 6
sh scripts/zero-shot-quant.sh models/nq/bert-base models/nq/quantized/3-oneshot 3
sh scripts/zero-shot-quant.sh models/nq/bert-base models/nq/quantized/2-oneshot 2
sh scripts/zero-shot-quant.sh models/nq/bert-base models/nq/quantized/1-oneshot 1
sh scripts/eval_nq.sh  models/nq/struct/12-oneshot/ models/nq/bert-base/
sh scripts/eval_nq.sh  models/nq/struct/9-oneshot/ models/nq/bert-base/
sh scripts/eval_nq.sh  models/nq/struct/6-oneshot/ models/nq/bert-base/
sh scripts/eval_nq.sh  models/nq/struct/3-oneshot/ models/nq/bert-base/
sh scripts/eval_nq.sh  models/nq/struct/2-oneshot/ models/nq/bert-base/
sh scripts/eval_nq.sh  models/nq/struct/1-oneshot/ models/nq/bert-base/
sh scripts/eval_nq.sh  models/nq/quantized/12-oneshot/ models/nq/bert-base/
sh scripts/eval_nq.sh  models/nq/quantized/9-oneshot/ models/nq/bert-base/
sh scripts/eval_nq.sh  models/nq/quantized/6-oneshot/ models/nq/bert-base/
sh scripts/eval_nq.sh  models/nq/quantized/3-oneshot/ models/nq/bert-base/
sh scripts/eval_nq.sh  models/nq/quantized/2-oneshot/ models/nq/bert-base/
sh scripts/eval_nq.sh  models/nq/quantized/1-oneshot/ models/nq/bert-base/