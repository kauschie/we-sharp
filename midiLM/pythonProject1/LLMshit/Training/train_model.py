import matplotlib.pyplot as plt
import os
from transformers import TrainingArguments, Trainer
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling
from datetime import datetime
import json
from transformers import TrainerCallback
from transformers import AdamW


def train_model(model, tokenized_data, tokenizer, output_path, data_file_name):
    # 确保日志存储目录存在
    os.makedirs(os.path.join(output_path, "logs"), exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f"log_{data_file_name.replace('.json', '')}_{timestamp}.json"
    log_file_path = os.path.join(output_path, "logs", log_file_name)

    # **转换 tokenized_data 为 Dataset**
    dataset = Dataset.from_list([{"input_ids": entry["input_ids"]} for entry in tokenized_data])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_metrics = []  # 存 loss 数据

    # ✅ **定义 LogCallback 并正确注册**
    class LogCallback(TrainerCallback):
        """
        自定义 Trainer 回调，用于记录 loss 并保存到日志文件。
        """

        def __init__(self, log_file_path):
            self.log_file_path = log_file_path
            self.training_metrics = []  # 存 loss 数据

        def on_log(self, args, state, control, logs=None, **kwargs):
            """
            记录 loss，并保存到日志文件。
            """
            if logs and "loss" in logs:
                self.training_metrics.append({"step": state.global_step, "loss": logs["loss"]})
                with open(self.log_file_path, "w", encoding="utf-8") as f:
                    json.dump(self.training_metrics, f, indent=2)

        def on_init_end(self, args, state, control, **kwargs):
            """初始化 Trainer 时调用"""
            print("✅ LogCallback 初始化完成！")

        def on_train_begin(self, args, state, control, **kwargs):
            """训练开始时调用"""
            print(f"🚀 训练开始，总步数: {state.max_steps}")

        def on_train_end(self, args, state, control, **kwargs):
            """训练结束时调用"""
            print(f"🎉 训练完成，总步数: {state.global_step}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 创建日志路径
    log_file_name = f"log_{data_file_name.replace('.json', '')}_{timestamp}.json"
    log_file_path = os.path.join(output_path, "logs", log_file_name)

    log_callback = LogCallback(log_file_path)  # ✅ 传入日志文件路径

    # **训练参数**
    training_args = TrainingArguments(
        output_dir=output_path,
        overwrite_output_dir=True,

        # **1️⃣ 训练策略**
        num_train_epochs=4,  # 🔹 **减少 epoch，避免过拟合**
        per_device_train_batch_size=16,  # 🔹 **显存足够，batch 16**
        gradient_accumulation_steps=4,  # 🔹 **全局 batch size = 16×4=64**

        # **2️⃣ 优化显存**
        save_steps=2000,  # 🔹 **减少 checkpoint 频率，降低 IO 影响**
        save_total_limit=2,  # 🔹 **最多保留 2 个 checkpoint**
        fp16=True,  # 🔹 **启用混合精度，降低显存占用**

        # **3️⃣ 学习率 & 训练优化**
        learning_rate=3e-4,  # 🔹 **更高的学习率（适应小数据）**
        # lr_scheduler_type="cosine",  # 🔹 **更平稳的衰减**
        warmup_steps=1000,  # 🔹 **减少 warmup，快速进入收敛**
        weight_decay=0.1,  # 🔹 **更高的正则化，提高泛化**

        # **4️⃣ 日志 & 监控**
        logging_dir=os.path.join(output_path, "logs"),
        logging_steps=100,  # 🔹 **减少日志频率**
        save_strategy="epoch",  # 🔹 **每个 epoch 存 checkpoint**
        report_to="none",
    )

    optimizer = AdamW(
        model.parameters(),
        lr=3e-4,  # 初始学习率
        weight_decay=0.1,  # 权重衰减（增强泛化）
        betas=(0.9, 0.95)  # 动量参数（适合 LLM）
    )

    # ✅ **将 log_callback 绑定到 Trainer**
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=(optimizer, None),
        callbacks=[log_callback],  # ✅ 这里注册回调
    )

    # **训练模型**
    trainer.train()

    # **保存模型和分词器**
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print(f"✅ 训练完成！模型保存至 {output_path}")
    print(f"📂 训练日志保存至 {log_file_path}")

    # ✅ **绘制 Loss 曲线**
    if training_metrics:
        steps = [entry["step"] for entry in training_metrics]
        losses = [entry["loss"] for entry in training_metrics]

        plt.figure(figsize=(8, 5))
        plt.plot(steps, losses, label="Training Loss", color="blue")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_path, "loss_curve.png"))  # 保存 loss 曲线
        plt.show()

    return log_file_path
