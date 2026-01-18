# OCR/utils/experiment_logger.py (or place directly at the top of train.py)
import os
import csv
import datetime
import yaml  # For recording hyperparameter details

LOG_FILE_NAME = "training_log.csv"
CONFIG_LOG_DIR = "experiment_configs"


def initialize_log_file(log_dir="checkpoints", hyperparameters=None, model_cfg_path=None, data_cfg_path=None):
    """
    Initialize log file, create and write header if it doesn't exist.
    Also save the current experiment configuration.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, LOG_FILE_NAME)


    current_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    current_config_log_dir = os.path.join(log_dir, CONFIG_LOG_DIR, current_time_str)
    os.makedirs(current_config_log_dir, exist_ok=True)

    experiment_id = current_time_str

    if model_cfg_path:
        try:
            with open(model_cfg_path, 'r') as f_model_cfg, \
                    open(os.path.join(current_config_log_dir, "model_config_snapshot.yaml"), 'w') as f_model_snapshot:
                model_config = yaml.safe_load(f_model_cfg)
                yaml.dump(model_config, f_model_snapshot, sort_keys=False)
        except Exception as e:
            print(f"Warning: Unable to save model config file snapshot: {e}")

    if data_cfg_path:
        try:
            with open(data_cfg_path, 'r') as f_data_cfg, \
                    open(os.path.join(current_config_log_dir, "data_config_snapshot.yaml"), 'w') as f_data_snapshot:
                data_config = yaml.safe_load(f_data_cfg)
                yaml.dump(data_config, f_data_snapshot, sort_keys=False)
        except Exception as e:
            print(f"Warning: Unable to save data config file snapshot: {e}")


    write_header = not os.path.exists(log_file_path)


    header = [
        "experiment_id", "timestamp", "epoch",
        "batch_size", "learning_rate", "weight_decay", "lambda_ctc", "model_type",

        "d_model", "encoder_layers", "decoder_layers", "feature_extractor",
        "train_loss_total", "train_loss_ctc", "train_loss_attention",
        "val_loss_total", "val_loss_ctc", "val_loss_attention",
        "val_cer", "val_sequence_acc"
    ]


    try:
        with open(log_file_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(header)
        print(f"Log file initialized/verified: {log_file_path}")
        return experiment_id, log_file_path
    except IOError:
        print(f"Error: Unable to write to log file {log_file_path}")
        return None, None

def log_epoch_data(log_file_path, experiment_id, epoch_data, hyperparameters):
    """
    Record one epoch's data to CSV file.
    epoch_data should be a dictionary containing all metrics.
    hyperparameters should be a dictionary containing key hyperparameters of the current run.
    """
    if not log_file_path:
        print("Error: Log file path not provided, cannot record data.")
        return

    try:

        row_data = {
            "experiment_id": experiment_id,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "epoch": epoch_data.get("epoch", -1),


            "batch_size": hyperparameters.get("batch_size", "N/A"),
            "learning_rate": hyperparameters.get("learning_rate", "N/A"),
            "weight_decay": hyperparameters.get("weight_decay", "N/A"),
            "lambda_ctc": hyperparameters.get("lambda_ctc", "N/A"),
            "model_type": hyperparameters.get("model_type", "N/A"),
            "d_model": hyperparameters.get("d_model", "N/A"),
            "encoder_layers": hyperparameters.get("encoder_num_layers", "N/A"),
            "decoder_layers": hyperparameters.get("decoder_num_layers", "N/A"),
            "feature_extractor": hyperparameters.get("feature_extractor_type", "N/A"),

            "train_loss_total": epoch_data.get("train_loss_total", float('nan')),
            "train_loss_ctc": epoch_data.get("train_loss_ctc", float('nan')),
            "train_loss_attention": epoch_data.get("train_loss_attention", float('nan')),

            "val_loss_total": epoch_data.get("val_loss_total", float('nan')),
            "val_loss_ctc": epoch_data.get("val_loss_ctc", float('nan')),
            "val_loss_attention": epoch_data.get("val_loss_attention", float('nan')),

            "val_cer": epoch_data.get("val_cer", float('nan')),
            "val_sequence_acc": epoch_data.get("val_sequence_acc", float('nan'))
        }


        header = [
            "experiment_id", "timestamp", "epoch",
            "batch_size", "learning_rate", "weight_decay", "lambda_ctc", "model_type",
            "d_model", "encoder_layers", "decoder_layers", "feature_extractor",
            "train_loss_total", "train_loss_ctc", "train_loss_attention",
            "val_loss_total", "val_loss_ctc", "val_loss_attention",
            "val_cer", "val_sequence_acc"
        ]

        with open(log_file_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            #
            writer.writerow({key: row_data.get(key, "") for key in header})

    except IOError:
        print(f"Error: Unable to write log to {log_file_path}")
    except Exception as e:
        print(f"Error occurred while recording epoch data: {e}")
