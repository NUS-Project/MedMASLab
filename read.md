## 这个代码里面的驱动都改成了Qwen7b模型
下面这个命令行是运行命令：
MDAgents(base_model是qwen2.5vl_7b)
评测pubmedqa
srun -p raise --gres=gpu:4 --quotatype=reserved apptainer exec --nv --bind /mnt:/mnt /mnt/petrelfs/panjiabao/huxiaobin/yunhang/apptainer/qyh_run_container.sif python /mnt/petrelfs/panjiabao/huxiaobin/yunhang/Code/visual-tool-lab-main/main.py --model MDAgents --dataset_name pubmedqa --batch_size 10 --num_workers 10 --judge_batch_size 10 --save_interval 100 --base_model Qwen2.5-VL-7B-Instruct

评测medbullets
srun -p raise --gres=gpu:4 --quotatype=reserved apptainer exec --nv --bind /mnt:/mnt /mnt/petrelfs/panjiabao/huxiaobin/yunhang/apptainer/qyh_run_container.sif python /mnt/petrelfs/panjiabao/huxiaobin/yunhang/Code/visual-tool-lab-main/main.py --model MDAgents --dataset_name medbullets --batch_size 10 --num_workers 10 --judge_batch_size 10 --save_interval 100 --base_model Qwen2.5-VL-7B-Instruct

评测MMLU
srun -p raise --gres=gpu:4 --quotatype=reserved apptainer exec --nv --bind /mnt:/mnt /mnt/petrelfs/panjiabao/huxiaobin/yunhang/apptainer/qyh_run_container.sif python /mnt/petrelfs/panjiabao/huxiaobin/yunhang/Code/visual-tool-lab-main/main.py --model MDAgents --dataset_name MMLU --batch_size 10 --num_workers 10 --judge_batch_size 10 --save_interval 100 --base_model Qwen2.5-VL-7B-Instruct


评测dxbench
srun -p raise --gres=gpu:4 --quotatype=reserved apptainer exec --nv --bind /mnt:/mnt /mnt/petrelfs/panjiabao/huxiaobin/yunhang/apptainer/qyh_run_container.sif python /mnt/petrelfs/panjiabao/huxiaobin/yunhang/Code/visual-tool-lab-main/main.py --model MDAgents --dataset_name dxbench --batch_size 10 --num_workers 10 --judge_batch_size 10 --save_interval 100 --base_model Qwen2.5-VL-7B-Instruct

评测MedCXR
srun -p raise --gres=gpu:4 --quotatype=reserved apptainer exec --nv --bind /mnt:/mnt /mnt/petrelfs/panjiabao/huxiaobin/yunhang/apptainer/qyh_run_container.sif python /mnt/petrelfs/panjiabao/huxiaobin/yunhang/Code/visual-tool-lab-main/main.py --model MDAgents --dataset_name MedCXR --batch_size 10 --num_workers 10 --judge_batch_size 10 --save_interval 100 --base_model Qwen2.5-VL-7B-Instruct


评测MedXpertQA_MM
srun -p raise --gres=gpu:4 --quotatype=reserved apptainer exec --nv --bind /mnt:/mnt /mnt/petrelfs/panjiabao/huxiaobin/yunhang/apptainer/qyh_run_container.sif python /mnt/petrelfs/panjiabao/huxiaobin/yunhang/Code/visual-tool-lab-main/main.py --model MDAgents --dataset_name MedXpertQA_MM --batch_size 10 --num_workers 10 --judge_batch_size 10 --save_interval 100 --base_model Qwen2.5-VL-7B-Instruct