echo "======HotpotQA KG Construction======"
# echo "======TF-IDF======"
# for n_kw in 1 2 3 5 10 15 20 30 40 50 60 70 80 90 100
# do
#     echo 'TF-IDF' $n_kw
#     python3 main.py --kg='TF-IDF' --dataset='HotpotQA' --n_processes=16 --n_kw=$n_kw
#     # python3 count_nei_overlap.py --kg='TF-IDF' --dataset='HotpotQA' --n_kw=$n_kw
# done

# echo "======KNN======"
# for k_knn in 1 2 3 5 10 15 20 30 40 50 60 70 80 90 100
# do
#     echo 'KNN' $k_knn
#     python3 main.py --kg='KNN' --dataset='HotpotQA' --n_processes=64 --k_knn=$k_knn
#     # python3 count_nei_overlap.py --kg='KNN' --dataset='HotpotQA' --k_knn=$k_knn
# done

echo "======TAGME======"
for prior_prob in 0.8
do
    echo 'Prior Probability Threshold' $prior_prob
    python3 main.py --kg='TAGME' --dataset='HotpotQA' --n_processes=64 --prior_prob=$prior_prob
    # python3 count_nei_overlap.py --kg='TAGME' --dataset='HotpotQA' --n_processes=64 --prior_prob=$prior_prob
done


# echo "======MDR_KNN======"
# for k_knn in 1 2 3 5 10 15 20 30 40 50 60 70 80 90 100
# do
#     echo 'MDR-KNN' $k_knn
#     python3 main.py --kg='MDR-KNN' --dataset='HotpotQA' --n_processes=64 --k_knn=$k_knn
#     # python3 count_nei_overlap.py --kg='MDR-KNN' --dataset='HotpotQA' --n_processes=64 --k_knn=$k_knn
# done



