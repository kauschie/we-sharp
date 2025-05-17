import os
import json
from datetime import datetime
from transformers import TrainingArguments, Trainer
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling


def train_model_with_logging(model, tokenized_data, tokenizer, output_dir, data_file_name):
    """
    使用 Trainer 训练模型并记录日志。

    参数:
        model: 需要训练的模型。
        tokenized_data: 已分词的数据。
        tokenizer: 分词器。
        output_dir: 输出目录。
        data_file_name: 数据文件名，用于生成日志文件名。

    返回:
        log_file_path: 日志文件的路径。
    """
    from transformers import TrainingArguments, Trainer
    from datasets import Dataset
    from transformers import DataCollatorForLanguageModeling

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 创建数据集
    dataset = Dataset.from_list(tokenized_data)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 根据数据文件名和时间戳生成日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name = f"{data_file_name.replace('.json', '')}_log_{timestamp}.json"
    log_file_path = os.path.join(output_dir, log_file_name)
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
        output_dir=os.path.join(output_dir, "model_output"),
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=16,
        save_steps=1000,
        save_total_limit=3,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=100,
        learning_rate=2e-5,
        weight_decay=0.01,
        fp16=True,
        report_to="none",
    )

    # 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[LogCallback()],
    )

    # 开始训练
    trainer.train()

    # 保存模型和分词器
    model.save_pretrained(os.path.join(output_dir, "trained_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "trained_model"))

    print(f"训练完成！模型保存至 {os.path.join(output_dir, 'trained_model')}")
    print(f"训练日志保存至 {log_file_path}")

    return log_file_path