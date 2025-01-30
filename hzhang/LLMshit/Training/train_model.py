import os
from transformers import TrainingArguments, Trainer
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling
from datetime import datetime
import json
def train_model(model, tokenized_data, tokenizer, output_path, data_file_name):
    """
    使用 Trainer API 训练模型。

    参数:
        model: 需要训练的模型。
        tokenized_data: 已分词的数据，列表格式。
        tokenizer: 分词器。
        output_path: 模型和分词器保存路径。
    """
    # 创建数据集
    dataset = Dataset.from_list(tokenized_data)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # 不使用 Masked Language Modeling
    )

    # 确保输出路径存在
    os.makedirs(output_path, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f"{data_file_name.replace('.json', '')}_log_{timestamp}.json"
    log_file_path = os.path.join(output_path, log_file_name)
    training_metrics = []

    # 自定义回调函数记录日志
    class LogCallback:
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                training_metrics.append(logs)
                # 实时保存日志
                with open(log_file_path, "w", encoding="utf-8") as f:
                    json.dump(training_metrics, f, indent=2)
    # 定义训练参数
    training_args = TrainingArguments(
        output_dir="./model_output",
        overwrite_output_dir=True,
        num_train_epochs=10,  # 仍保持 10 个 epoch
        per_device_train_batch_size=16,  # 每设备的批量大小
        gradient_accumulation_steps=2,  # 梯度累积，等效于 batch_size = 32
        save_steps=1000,  # 每 1000 步保存一次检查点
        save_total_limit=3,  # 保存最近 3 个检查点
        logging_dir="./logs",
        logging_steps= 1,  # 每 100 步记录一次日志
        learning_rate=1.5e-5,  # 初始学习率降低
        lr_scheduler_type="cosine",  # 使用余弦衰减策略
        warmup_steps=1000,  # 延长 warmup 阶段
        weight_decay=0.01,  # 权重衰减
        fp16=True,  # 使用混合精度训练
        report_to="none"
    )

    # 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 训练模型
    trainer.train()

    # 保存模型和分词器
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print(f"训练完成！模型保存至 {os.path.join(output_path, 'trained_model')}")
    print(f"训练日志保存至 {log_file_path}")

    return log_file_path
