### Problem 1
experienced a MemoryError during sdp training as smaller server is used: CUDA out of memory. Tried to allocate 3.47 GiB. GPU 0 has a total capacity of 31.36 GiB of which 2.36 GiB is free.

Solution: keep the same effective batch size. But reduce the batch size by half and increase gradient_accumulation from 8 to 16. The trade-offs is the increase in the total training time for sdp

### Problem 2
in joint run phase b, I noticed the running time is still slow and the GPU is not fully occupied (~60%). I add a batch=16 (~92%) instead of running it one by one to increase the training time.

### Problem 3
If experienced TypeError during DPO: DPOConfig.__init__() got an unexpected keyword argument 'overwrite_output_dir', fix the code by
```bash
grep -n "overwrite_output_dir" train_dpo_lora.py
sed -i '/overwrite_output_dir/d' train_dpo_lora.py
grep -n "max_prompt_length" train_dpo_lora.py
sed -i '/max_prompt_length/d' train_dpo_lora.py
python train_dpo_lora.py --config configs/_phase_a_pad_dpo_iter1.yml
```

### Problem 4
CUDA error during DPO:
OOM 不是因為 batch 大，是因為兩份完整 8B 模型同時在 GPU 上：

model (bf16): ~16GB
ref_model (bf16): ~16GB
LoRA + optimizer: ~2GB
DPO forward (chosen + rejected 一起跑，等於 2×4096 tokens 的 activations): ~10-15GB
加起來超過你 GPU 的 VRAM。
such try solution:
precompute_ref_log_probs=True and updated train_dpo_lora.py 是唯一不犧牲任何東西的解法——先用 ref_model 算完分數存起來，卸載，再用 training model 訓練。嚴格符合論文，不用量化，不降 batch size，不降 max_length。
```bash
sed -i 's/ref_model=None/ref_model=ref_model/' train_dpo_lora.py
sed -i '/remove_unused_columns/a\        precompute_ref_log_probs=True,' train_dpo_lora.py
python train_dpo_lora.py --config configs/_phase_a_pad_dpo_iter1.yml
```
### Problem 5
ValueError: The table can't have duplicated columns but columns ['ref_chosen_logps'] are duplicated.
solution: updated train_dpo_lora.py
其實有個更簡單乾淨的方案：根本不需要手動預算 ref logps。只要 ref_model=None + precompute_ref_log_probs=True，TRL 會用 policy model 本身（在 init 時等同 ref，因為還沒訓練）來算 ref logps，全程只有一個模型在 GPU 上。

Updated version ```train_joint.sh```
```text
Phase A data generation — 生成 PAD 的 DPO 配對資料 
Phase A DPO training — 用 DPO 訓練 PAD adapter 
Phase B data generation — 用新的 PAD 生成 SDP 的 on-policy SFT 資料
Phase B SDP SFT training — 用 SFT 訓練 SDP adapter
```

### Problem 6 
As the memory in the 系統盤 (/root, 30G) is not enough, the checkpoints are moved to the 數據盤 (/root/autodl-tmp). They can be accessed by:
```bash
# 看每個 checkpoint 的內容
echo "=== PAD SFT checkpoints ==="
ls /root/autodl-tmp/checkpoints/pad-sft-ckpt-400/
ls /root/autodl-tmp/checkpoints/pad-sft-ckpt-450/
ls /root/autodl-tmp/checkpoints/pad-sft-ckpt-459/

echo "=== SDP SFT checkpoints ==="
ls /root/autodl-tmp/checkpoints/sdp-sft-ckpt-250/
ls /root/autodl-tmp/checkpoints/sdp-sft-ckpt-300/
ls /root/autodl-tmp/checkpoints/sdp-sft-ckpt-306/

echo "=== PAD DPO checkpoint ==="
ls /root/autodl-tmp/checkpoints/pad-dpo-ckpt-151/
```