import modal
 
app = modal.App("my-gpu-job")
 
volume = modal.Volume.from_name("probe-checkpoints", create_if_missing=True)
 
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "transformers", "scipy", "matplotlib", "scikit-learn", "accelerate")
    .add_local_dir("../../src", remote_path="/root/src")
    .add_local_dir("../../data", remote_path="/root/data")
)
 
@app.function(gpu="A100-80GB", image=image, timeout=18000, volumes={"/root/checkpoints": volume})
def run_job():
    import sys
    import os
    os.makedirs("/root/checkpoints/reading_probe", exist_ok=True)
    os.makedirs("/root/checkpoints/controlling_probe", exist_ok=True)
    # Directories for plots and underlying CM data
    os.makedirs("/root/checkpoints/reading_probe/plots", exist_ok=True)
    os.makedirs("/root/checkpoints/reading_probe/cm_data", exist_ok=True)
    os.makedirs("/root/checkpoints/controlling_probe/plots", exist_ok=True)
    os.makedirs("/root/checkpoints/controlling_probe/cm_data", exist_ok=True)
    os.makedirs("/root/checkpoints/", exist_ok=True)
    sys.path.insert(0, "/root/src")
 
    from torch.utils.data import Dataset
    from torch.utils.data.dataloader import DataLoader
    import torch.nn.functional as F
    from losses import edl_mse_loss
 
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    from tqdm.auto import tqdm
 
    from dataset import TextDataset
 
    import matplotlib
    matplotlib.use("Agg")  # headless backend; safe for Modal containers
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
    from sklearn.model_selection import train_test_split
    from torch.utils.data import Subset
 
    from probes import ProbeClassification, ProbeClassificationMixScaler
    from train_test_utils import train, test
    import torch.nn as nn
 
    import time
    import numpy as np
    import pickle
 
    tic, toc = (time.time, time.time)
 
    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
 
    model = AutoModelForCausalLM.from_pretrained(
        "openai/gpt-oss-20b",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
 
    from probes import ProbeClassification, ProbeClassificationMixScaler
 
    class TrainerConfig:
        # optimization parameters
        learning_rate = 1e-3
        betas = (0.9, 0.95)
        weight_decay = 0.1  # only applied on matmul weights
 
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
 
    import os
    from torch.utils.data import Dataset
    from torch.utils.data.dataloader import DataLoader
    import torch.nn.functional as F
 
    import torch
    from tqdm.auto import tqdm
    from dataset import split_conversation, llama_v2_prompt, TextDataset
 
    from probes import LinearProbeClassification, LinearProbeClassificationMixScaler
    import sklearn.model_selection
    import random
 
    # ------------------------------------------------------------------
    # Helper: save confusion matrix figure + underlying data for a layer
    # ------------------------------------------------------------------
    def save_cm_artifacts(probe_kind, dict_name, layer_num, test_results,
                          label_to_id, train_acc, best_acc, final_acc):
        """
        probe_kind: "reading_probe" or "controlling_probe"
        test_results: tuple from `test(...)` with return_raw_outputs=True.
                      Convention used in the original script:
                          test_results[1] -> accuracy
                          test_results[2] -> predictions
                          test_results[3] -> ground truth labels
        """
        plots_dir = f"/root/checkpoints/{probe_kind}/plots"
        data_dir = f"/root/checkpoints/{probe_kind}/cm_data"
 
        label_list = list(label_to_id.keys())
        label_ids = list(label_to_id.values())
 
        y_pred = test_results[2]
        y_true = test_results[3]
 
        cm = confusion_matrix(y_true, y_pred, labels=label_ids)
 
        # --- Save the underlying data so the plot can be regenerated later ---
        cm_payload = {
            "probe_kind": probe_kind,
            "dict_name": dict_name,
            "layer_num": layer_num,
            "label_to_id": dict(label_to_id),
            "label_list": label_list,
            "label_ids": label_ids,
            "y_true": np.asarray(y_true),
            "y_pred": np.asarray(y_pred),
            "confusion_matrix": np.asarray(cm),
            "train_acc": float(train_acc) if train_acc is not None else None,
            "best_test_acc": float(best_acc) if best_acc is not None else None,
            "final_test_acc": float(final_acc) if final_acc is not None else None,
        }
        data_path = f"{data_dir}/{dict_name}_cm_layer_{layer_num}.pkl"
        with open(data_path, "wb") as f:
            pickle.dump(cm_payload, f)
 
        # --- Render and save the plot itself (PNG + PDF) ---
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(cm, display_labels=label_list)
        disp.plot(ax=ax, colorbar=True)
        ax.set_title(
            f"{probe_kind} | {dict_name} | layer {layer_num}\n"
            f"best_acc={best_acc:.4f}  final_acc={final_acc:.4f}"
        )
        fig.tight_layout()
 
        png_path = f"{plots_dir}/{dict_name}_cm_layer_{layer_num}.png"
        pdf_path = f"{plots_dir}/{dict_name}_cm_layer_{layer_num}.pdf"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        plt.close(fig)
 
        return png_path, data_path
 
    jump_socioeco = True
 
    new_prompt_format = True
    residual_stream = True
    uncertainty = False
    logistic = True
    augmented = False
    remove_last_ai_response = True
    include_inst = True
    one_hot = True
 
    label_to_id_age = {"child": 0, "adolescent": 1, "adult": 2, "older adult": 3}
    label_to_id_gender = {"male": 0, "female": 1}
    label_to_id_socioeconomic = {"low": 0, "middle": 1, "high": 2}
    label_to_id_neweducation = {"someschool": 0, "highschool": 1, "collegemore": 2}
    label_to_id_priority = {"platform": 0, "developer": 1, "user": 2}
 
    prompt_translator = {
        "_age_": "age",
        "_gender_": "gender",
        "_socioeco_": "socioeconomic status",
        "_education_": "education level",
        "_priority_": "priority level",
    }
 
    openai_dataset = {
        "_age_": "/root/data/dataset/openai_age_1/",
        "_gender_": "/root/data/dataset/openai_gender_1/",
        "_education_": "/root/data/dataset/openai_education_1/",
        "_socioeco_": "/root/data/dataset/openai_socioeconomic_1/",
        "_priority_": "/root/data/dataset/openai_priority_academic_dishonesty",
    }
 
    accuracy_dict = {}
 
    directories = ["/root/data/dataset/openai_priority_academic_dishonesty/"]
    label_idfs = ["_priority_"]
    label_to_ids = [label_to_id_priority]
 
    # ==================================================================
    # READING PROBE
    # ==================================================================
    for directory, label_idf, label_to_id in zip(directories, label_idfs, label_to_ids):
        if label_idf == "_education_":
            additional_dataset = []
        elif label_idf == "_priority_":
            additional_dataset = []
        else:
            additional_dataset = [directory[:-2] + "2/", openai_dataset[label_idf]]
        if label_idf == "_gender_":
            additional_dataset += [
                "/root/data/dataset/openai_gender_2/",
                "/root/data/dataset/openai_gender_3/",
                "/root/data/dataset/openai_gender_4",
            ]
        if label_idf == "_education_":
            additional_dataset += [
                "/root/data/dataset/openai_education_2",
                "/root/data/dataset/openai_education_3/",
            ]
        if label_idf == "_socioeco_":
            additional_dataset += [
                "/root/data/dataset/openai_socioeconomic_2/",
                "/root/data/dataset/openai_socioeconomic_3/",
            ]
        if label_idf == "_age_":
            additional_dataset += ["/root/data/dataset/openai_age_2/"]
        if label_idf == "_priority_":
            additional_dataset += [
                "/root/data/dataset/openai_priority_addiction_facilitation/",
                "/root/data/dataset/openai_priority_animal_harm/",
                "/root/data/dataset/openai_priority_autonomous_agent_scope_creep/",
            ]
 
        dataset = TextDataset(
            directory, tokenizer, model, label_idf=label_idf, label_to_id=label_to_id,
            convert_to_llama2_format=True, additional_datas=additional_dataset,
            new_format=new_prompt_format,
            residual_stream=residual_stream, if_augmented=augmented,
            remove_last_ai_response=remove_last_ai_response, include_inst=include_inst, k=1,
            one_hot=False, last_tok_pos=-1,
        )
        dict_name = label_idf.strip("_")
 
        train_idx, val_idx = sklearn.model_selection.train_test_split(
            list(range(len(dataset))),
            test_size=0.2, train_size=0.8,
            random_state=12345, shuffle=True,
            stratify=dataset.labels,
        )
 
        train_dataset = Subset(dataset, train_idx)
        test_dataset = Subset(dataset, val_idx)
 
        sampler = None
        train_loader = DataLoader(train_dataset, shuffle=True, sampler=sampler, pin_memory=True, batch_size=200, num_workers=1)
        test_loader = DataLoader(test_dataset, shuffle=False, pin_memory=True, batch_size=400, num_workers=1)
 
        if uncertainty:
            loss_func = edl_mse_loss
        else:
            loss_func = nn.BCELoss()
        torch_device = "cuda"
 
        seeds = list(range(42))
        seeds = seeds[:9]
        accuracy_dict[dict_name] = []
        accuracy_dict[dict_name + "_final"] = []
        accuracy_dict[dict_name + "_train"] = []
 
        accs = []
        final_accs = []
        train_accs = []
        for i in tqdm(range(0, 24)):
            trainer_config = TrainerConfig()
            d_model = model.config.hidden_size
            probe = LinearProbeClassification(
                probe_class=len(label_to_id.keys()), device="cuda", input_dim=d_model,
                logistic=logistic,
            )
            optimizer, scheduler = probe.configure_optimizers(trainer_config)
            best_acc = 0
            max_epoch = 50
            verbosity = False
            layer_num = i
            print("-" * 23 + f"Layer {layer_num}" + "-" * 23)
            for epoch in range(1, max_epoch + 1):
                if epoch == max_epoch:
                    verbosity = True
                if uncertainty:
                    train_results = train(
                        probe, torch_device, train_loader, optimizer,
                        epoch, loss_func=loss_func, verbose_interval=None,
                        verbose=verbosity, layer_num=layer_num,
                        return_raw_outputs=True, epoch_num=epoch,
                        num_classes=len(label_to_id.keys()),
                    )
                    test_results = test(
                        probe, torch_device, test_loader, loss_func=loss_func,
                        return_raw_outputs=True, verbose=verbosity, layer_num=layer_num,
                        scheduler=scheduler, epoch_num=epoch,
                        num_classes=len(label_to_id.keys()),
                    )
                else:
                    train_results = train(
                        probe, torch_device, train_loader, optimizer,
                        epoch, loss_func=loss_func, verbose_interval=None,
                        verbose=verbosity, layer_num=layer_num,
                        return_raw_outputs=True,
                        one_hot=one_hot, num_classes=len(label_to_id.keys()),
                    )
                    test_results = test(
                        probe, torch_device, test_loader, loss_func=loss_func,
                        return_raw_outputs=True, verbose=verbosity, layer_num=layer_num,
                        scheduler=scheduler,
                        one_hot=one_hot, num_classes=len(label_to_id.keys()),
                    )
 
                if test_results[1] > best_acc:
                    best_acc = test_results[1]
                    torch.save(probe.state_dict(), f"/root/checkpoints/reading_probe/{dict_name}_probe_at_layer_{layer_num}.pth")
                    volume.commit()
            torch.save(probe.state_dict(), f"/root/checkpoints/reading_probe/{dict_name}_probe_at_layer_{layer_num}_final.pth")
            volume.commit()
 
            accs.append(best_acc)
            final_accs.append(test_results[1])
            train_accs.append(train_results[1])
 
            # ---- Save plot + underlying CM data for this layer ----
            png_path, data_path = save_cm_artifacts(
                probe_kind="reading_probe",
                dict_name=dict_name,
                layer_num=layer_num,
                test_results=test_results,
                label_to_id=label_to_id,
                train_acc=train_results[1],
                best_acc=best_acc,
                final_acc=test_results[1],
            )
            print(f"  saved plot: {png_path}")
            print(f"  saved data: {data_path}")
            volume.commit()
 
            accuracy_dict[dict_name].append(accs)
            accuracy_dict[dict_name + "_final"].append(final_accs)
            accuracy_dict[dict_name + "_train"].append(train_accs)
 
            with open("/root/checkpoints/reading_probe_experiment.pkl", "wb") as outfile:
                pickle.dump(accuracy_dict, outfile)
            volume.commit()
        del dataset, train_dataset, test_dataset, train_loader, test_loader
        torch.cuda.empty_cache()
 
    # ==================================================================
    # CONTROLLING PROBE
    # ==================================================================
    accuracy_dict = {}
 
    directories = ["/root/data/dataset/openai_priority_academic_dishonesty/"]
    label_idfs = ["_priority_"]
    label_to_ids = [label_to_id_priority]
 
    for directory, label_idf, label_to_id in zip(directories, label_idfs, label_to_ids):
        if label_idf == "_education_":
            additional_dataset = []
        elif label_idf == "_priority_":
            additional_dataset = []
        else:
            additional_dataset = [directory[:-2] + "2/", openai_dataset[label_idf]]
        if label_idf == "_gender_":
            additional_dataset += [
                "/root/data/dataset/openai_gender_2/",
                "/root/data/dataset/openai_gender_3/",
                "/root/data/dataset/openai_gender_4",
            ]
        if label_idf == "_education_":
            additional_dataset += [
                "/root/data/dataset/openai_education_2",
                "/root/data/dataset/openai_education_3/",
            ]
        if label_idf == "_socioeco_":
            additional_dataset += [
                "/root/data/dataset/openai_socioeconomic_2/",
                "/root/data/dataset/openai_socioeconomic_3/",
            ]
        if label_idf == "_age_":
            additional_dataset += ["/root/data/dataset/openai_age_2/"]
        if label_idf == "_priority_":
            additional_dataset += [
                "/root/data/dataset/openai_priority_addiction_facilitation/",
                "/root/data/dataset/openai_priority_animal_harm/",
                "/root/data/dataset/openai_priority_autonomous_agent_scope_creep/",
            ]
 
        dataset = TextDataset(
            directory, tokenizer, model, label_idf=label_idf, label_to_id=label_to_id,
            convert_to_llama2_format=True, additional_datas=additional_dataset,
            new_format=new_prompt_format, control_probe=True,
            residual_stream=residual_stream, if_augmented=augmented,
            remove_last_ai_response=remove_last_ai_response, include_inst=include_inst, k=1,
            one_hot=False, last_tok_pos=-1,
        )
        dict_name = label_idf.strip("_")
 
        train_idx, val_idx = sklearn.model_selection.train_test_split(
            list(range(len(dataset))),
            test_size=0.2, train_size=0.8,
            random_state=12345, shuffle=True,
            stratify=dataset.labels,
        )
 
        train_dataset = Subset(dataset, train_idx)
        test_dataset = Subset(dataset, val_idx)
 
        sampler = None
        train_loader = DataLoader(train_dataset, shuffle=True, sampler=sampler, pin_memory=True, batch_size=200, num_workers=1)
        test_loader = DataLoader(test_dataset, shuffle=False, pin_memory=True, batch_size=400, num_workers=1)
 
        if uncertainty:
            loss_func = edl_mse_loss
        else:
            loss_func = nn.BCELoss()
        torch_device = "cuda"
 
        accuracy_dict[dict_name] = []
        accuracy_dict[dict_name + "_final"] = []
        accuracy_dict[dict_name + "_train"] = []
 
        accs = []
        final_accs = []
        train_accs = []
        for i in tqdm(range(0, 24)):
            trainer_config = TrainerConfig()
            d_model = model.config.hidden_size
            probe = LinearProbeClassification(
                probe_class=len(label_to_id.keys()), device="cuda", input_dim=d_model,
                logistic=logistic,
            )
            optimizer, scheduler = probe.configure_optimizers(trainer_config)
            best_acc = 0
            max_epoch = 50
            verbosity = False
            layer_num = i
            print("-" * 40 + f"Layer {layer_num}" + "-" * 40)
            for epoch in range(1, max_epoch + 1):
                if epoch == max_epoch:
                    verbosity = True
                if uncertainty:
                    train_results = train(
                        probe, torch_device, train_loader, optimizer,
                        epoch, loss_func=loss_func, verbose_interval=None,
                        verbose=verbosity, layer_num=layer_num,
                        return_raw_outputs=True, epoch_num=epoch,
                        num_classes=len(label_to_id.keys()),
                    )
                    test_results = test(
                        probe, torch_device, test_loader, loss_func=loss_func,
                        return_raw_outputs=True, verbose=verbosity, layer_num=layer_num,
                        scheduler=scheduler, epoch_num=epoch,
                        num_classes=len(label_to_id.keys()),
                    )
                else:
                    train_results = train(
                        probe, torch_device, train_loader, optimizer,
                        epoch, loss_func=loss_func, verbose_interval=None,
                        verbose=verbosity, layer_num=layer_num,
                        return_raw_outputs=True,
                        one_hot=one_hot, num_classes=len(label_to_id.keys()),
                    )
                    test_results = test(
                        probe, torch_device, test_loader, loss_func=loss_func,
                        return_raw_outputs=True, verbose=verbosity, layer_num=layer_num,
                        scheduler=scheduler,
                        one_hot=one_hot, num_classes=len(label_to_id.keys()),
                    )
 
                if test_results[1] > best_acc:
                    best_acc = test_results[1]
                    torch.save(probe.state_dict(), f"/root/checkpoints/controlling_probe/{dict_name}_probe_at_layer_{layer_num}.pth")
                    volume.commit()
            torch.save(probe.state_dict(), f"/root/checkpoints/controlling_probe/{dict_name}_probe_at_layer_{layer_num}_final.pth")
            volume.commit()
 
            accs.append(best_acc)
            final_accs.append(test_results[1])
            train_accs.append(train_results[1])
 
            # ---- Save plot + underlying CM data for this layer ----
            png_path, data_path = save_cm_artifacts(
                probe_kind="controlling_probe",
                dict_name=dict_name,
                layer_num=layer_num,
                test_results=test_results,
                label_to_id=label_to_id,
                train_acc=train_results[1],
                best_acc=best_acc,
                final_acc=test_results[1],
            )
            print(f"  saved plot: {png_path}")
            print(f"  saved data: {data_path}")
            volume.commit()
 
            accuracy_dict[dict_name].append(accs)
            accuracy_dict[dict_name + "_final"].append(final_accs)
            accuracy_dict[dict_name + "_train"].append(train_accs)
 
            with open("/root/checkpoints/controlling_probe_experiment.pkl", "wb") as outfile:
                pickle.dump(accuracy_dict, outfile)
            volume.commit()
        del dataset, train_dataset, test_dataset, train_loader, test_loader
        torch.cuda.empty_cache()
    volume.commit()
 
 
@app.function(volumes={"/root/checkpoints": volume})
def collect_checkpoints():
    """Read all files from the volume and return as a dict of {path: bytes}."""
    import os
    volume.reload()  # make sure we see the latest committed state
    files = {}
    for root, _, filenames in os.walk("/root/checkpoints"):
        for fname in filenames:
            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, "/root/checkpoints")
            with open(full_path, "rb") as f:
                files[rel_path] = f.read()
    return files
 
 
@app.local_entrypoint()
def main():
    run_job.remote()
    files = collect_checkpoints.remote()
    import os
    local_dir = "./downloaded_checkpoints"
    for rel_path, data in files.items():
        out_path = os.path.join(local_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(data)
        print(f"Wrote {out_path} ({len(data)} bytes)")
 