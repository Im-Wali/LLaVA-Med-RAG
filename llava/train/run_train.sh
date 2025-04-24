export TORCH_DISTRIBUTED_DEBUG="OFF"
export TORCHDYNAMO_DISABLE="1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,roundup_power2_divisions:[32:256,64:128,256:64,>:32]"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export OMP_NUM_THREADS=2

python /workspace/LLaVA-Med-RAG/llava/train/jp-train.py \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --model_name_or_path jp1924/llava-med-v1.5-mistral-7b \
    --data_path /workspace/LLaVA-Med-RAG/data/mimic_cxr_annotaion_json/chat_test_MIMIC_CXR_all_gpt4extract_rulebased_v1_test.json \
    --image_folder /workspace/LLaVA-Med-RAG/data/mimic_cxr_annotaion_json \
    --output_dir /workspace/LLaVA-Med-RAG/checkpoints/llava-med-mistral-rag-v1.5-7b-mimic-8K-5epoch_orgin_test \
    --num_train_epochs 2 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --eval_strategy no \
    --eval_steps 1000 \
    --save_strategy steps \
    --save_steps 400000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.05 \
    --lr_scheduler_type cosine \
    --logging_strategy steps \
    --logging_steps 1 \
    --optim lomo \
    --bf16 True \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --dataloader_prefetch_factor 2 \
    --report_to none \
    --seed 42