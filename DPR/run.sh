echo "=====HotpotQA====="
echo "Train"
python3 main.py --dataset='HotpotQA' --do_train
echo "Eval"
python3 main.py --dataset='HotpotQA'

echo "=====IIRC====="
echo "Train"
python3 main.py --dataset='IIRC' --do_train
echo "Eval"
python3 main.py --dataset='IIRC'

echo "=====MuSiQue====="
echo "Train"
python3 main.py --dataset='MuSiQue' --do_train
echo "Eval"
python3 main.py --dataset='MuSiQue'

echo "=====2WikiMQA====="
echo "Train"
python3 main.py --dataset='2WikiMQA' --do_train
echo "Eval"
python3 main.py --dataset='2WikiMQA'


echo "=====Retrieve the Relevant Context====="
python3 main_eval.py --dataset='HotpotQA'
python3 main_eval.py --dataset='IIRC'
python3 main_eval.py --dataset='MuSiQue'
python3 main_eval.py --dataset='2WikiMQA'
