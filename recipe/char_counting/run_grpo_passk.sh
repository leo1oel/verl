set -x

export VLLM_USE_V1=1

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo_passk \
    data.train_files=./data/char_count/rl/train.parquet \
    data.val_files=./data/char_count/rl/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=128 \
    data.max_response_length=128 \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    actor_rollout_ref.model.path=/pasteur2/u/yiming/verl/experiments/char_count/models/sft/fsdp/global_step_140/huggingface \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=5000 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.compute_sum_pi_squared=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_example' \
    trainer.experiment_name='smol135m_grpo_passk' \
    trainer.default_local_dir=./experiments/char_count/models/grpo_passk \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=70 \
    trainer.test_freq=5 \
    trainer.total_epochs=5 \
    trainer.use_legacy_worker_impl=disable \
    custom_reward_function.path=./recipe/char_counting/reward_function.py \
    custom_reward_function.name=char_count_reward_function
