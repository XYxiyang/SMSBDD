import os
import subprocess

def execute_command(command):
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command '{e.cmd}' failed with return code {e.returncode}")
        exit(e.returncode)

def main(data_id, outdir, top_k=3, dup_num=30):
    ref_list = [4, 6, 12, 19, 21, 22, 42, 44, 49, 53, 54, 59, 70, 72, 73, 74, 75, 76, 79, 80, 82, 83, 84, 88, 97, 98]
    
    # opt prior
    prior_mode = "beta_prior"
    if data_id in ref_list:
        prior_mode = "ref_prior"

    padded_data_id = f"{data_id:03d}"
    output_dir = os.path.join(outdir, f"sampling_{padded_data_id}")
    best_res_dir = os.path.join(output_dir, "best_record")

    if os.path.exists(best_res_dir):
        subprocess.run(f"rm -rf {best_res_dir}", shell=True)

    os.makedirs(best_res_dir)

    for dup_id in range(dup_num + 1):
        padded_dup_id = f"{dup_id:03d}"
        if dup_id == 0:
            ckpt_path = "logs_diffusion_full/pretrained_cond_decompdiff/decompdiff.pt"
        else:
            ckpt_path = "logs_diffusion_full/pretrained_cond_decompdiff/cond_decompdiff.pt" if os.path.exists(
                os.path.join(best_res_dir, "best_mol_arms.pt")) else "logs_diffusion_full/pretrained_cond_decompdiff/decompdiff.pt"
        
        print(f"Checkpoint path: {ckpt_path}")

        batch_size = 20
        while batch_size >= 1:
            try:
                execute_command(f"python scripts/sample_diffusion_decomp_compose.py "
                                f"configs/sampling_10_ret.yml --ckpt_path {ckpt_path} "
                                f"--outdir {output_dir} -i {data_id} "
                                f"--reference_arm_path {best_res_dir}/best_mol_arms.pt "
                                f"--dup_id {dup_id} --batch_size {batch_size} --prior_mode {prior_mode}")
                print(f"Successfully executed with batch_size: {batch_size}")
                break
            except subprocess.CalledProcessError:
                print(f"Failed with batch_size: {batch_size}. Reducing batch size.")
                batch_size //= 2
                if batch_size == 1:
                    print("Program failed with batch_size 1. Exiting.")
                    exit(1)

        sample_dir = os.path.join(output_dir, f"sampling_10_ret-{prior_mode}-{padded_data_id}-{padded_dup_id}")
        print(f"Sample directory: {sample_dir}")

        execute_command(f"python scripts/evaluate_mol_in_place_compose.py {sample_dir} "
                        f"--data_id {data_id} --best_res_dir {best_res_dir} "
                        f"--top_k {top_k} --protein_root data/test_set")
        
        # log best dir
        execute_command(f"cp {os.path.join(best_res_dir, 'best_mol_arms.pt')} {os.path.join(sample_dir, 'eval')}")

if __name__ == "__main__":
    # Example usage:
    data_id = 1  # 替换为实际的参数
    outdir = "/path/to/output"  # 替换为实际的输出目录
    main(data_id, outdir)
