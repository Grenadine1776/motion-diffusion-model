import os
import numpy as np
import torch
import sys # For sys.path manipulation if absolutely needed, though ideally avoided.

# Assuming MDM utilities are in sibling directories like 'utils', 'data_loaders'
# This structure relies on 'MDM/' being in sys.path, which main.py should handle.
from utils.fixseed import fixseed
from utils.parser_util import generate_args # To get default args if needed.
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils import dist_util
from utils.sampler_util import ClassifierFreeSampleModel # AutoRegressiveSampler might be needed if that feature is used.
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
# paramUtil and plot_script are for visualization, not directly needed for core generation for API
# from data_loaders.humanml.utils import paramUtil
from data_loaders.tensors import collate


MDM_INITIALIZED_ARGS = None

def initialize_mdm(args):
    """
    Initializes the MDM model, diffusion, and dataset.
    Called once on server startup.
    """
    global MDM_INITIALIZED_ARGS
    MDM_INITIALIZED_ARGS = args # Store for use in generation if needed for fps etc.

    print("--- Initializing MDM for API ---")
    fixseed(args.seed) # Initial seed, can be overridden per request for generation
    dist_util.setup_dist(args.device)

    # --- Load Dataset ---
    print("Loading dataset for MDM...")
    # n_frames for load_dataset: This is data.fixed_length.
    # It's often linked to motion_length * fps. For init, use a representative or max value.
    # The actual n_frames for generation will be calculated per request.
    # max_frames is the max sequence length the dataset loader can handle.
    # Let's use args.max_frames if available, or a common default like 196.
    max_frames_for_loader = getattr(args, 'max_frames', 196)
    # n_frames_for_loader: This is data.fixed_length which get_dataset_loader sets.
    # We can use a typical motion length for this initial setup.
    # E.g. if typical motion_length is 6s and fps is 20, then 120.
    # This might not be strictly necessary if data.fixed_length is reset/unused by parts we need.
    # For HumanML3D, num_frames (max_frames_for_loader) is key for HumanML3D class init.
    n_frames_for_loader = getattr(args, 'motion_length', 6.0) * getattr(args, 'fps', 20) # fps depends on dataset
    if args.dataset == 'kit':
        n_frames_for_loader = getattr(args, 'motion_length', 6.0) * 12.5


    data = get_dataset_loader(name=args.dataset,
                              batch_size=args.batch_size, # Will be 1 for API use mostly
                              num_frames=max_frames_for_loader,
                              # For t2m, hml_mode is 'text_only' if not using prefix.
                              # If API might support prefix later, 'train' might be needed.
                              hml_mode='text_only',
                              fixed_len=True, #MDM code sets data.fixed_length = n_frames. get_data.py uses fixed_len to set self.fixed_len for humanml dataset
                              )
    print(f"Dataset loaded. Sample HML mode: {data.dataset.hml_mode if hasattr(data, 'dataset') and hasattr(data.dataset, 'hml_mode') else 'N/A'}")


    # --- Create Model and Diffusion ---
    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)

    # --- Load Checkpoints ---
    print(f"Loading checkpoints from [{args.model_path}]...")
    load_saved_model(model, args.model_path, use_avg=getattr(args, 'use_ema', True)) # use_ema often True

    # --- Setup Model for Inference ---
    # Wrap with ClassifierFreeSampleModel if guidance is used.
    # The default guidance_param in generate_args is 2.5, so this wrapper is likely always needed.
    # The actual scale is passed in model_kwargs['y']['scale'] per request.
    if float(args.guidance_param) != 1.0: # Check guidance_param from args used for init
        print(f"Wrapping model with ClassifierFreeSampleModel (init guidance_param: {args.guidance_param}).")
        model = ClassifierFreeSampleModel(model)
    else:
        print("Not wrapping model with ClassifierFreeSampleModel (init guidance_param is 1.0).")

    model.to(dist_util.dev())
    model.eval()  # Disable random masking etc.

    print("--- MDM Initialization Complete ---")
    return model, diffusion, data


def generate_motion_from_loaded_state(
    model, diffusion, data, # Pre-loaded components
    text_prompt: str,
    motion_length_seconds: float,
    guidance_param: float,
    seed: int,
    mdm_init_args # The args object used during initialize_mdm
):
    """
    Generates motion using the pre-loaded model, diffusion, and data.
    """
    print(f"--- Generating motion for prompt: '{text_prompt}' ---")
    fixseed(seed) # Set seed for this specific generation

    # --- Prepare Generation Parameters ---
    # fps needs to be determined from the dataset specified in mdm_init_args
    fps = 20  # Default for humanml
    if mdm_init_args.dataset == 'kit':
        fps = 12.5
    
    max_frames = 196 # Default for humanml/kit
    if mdm_init_args.dataset not in ['kit', 'humanml']: # e.g. amass
        max_frames = 60
    
    n_frames = min(max_frames, int(motion_length_seconds * fps))
    batch_size = 1 # We generate one sample at a time

    print(f"Parameters: motion_length={motion_length_seconds}s, fps={fps}, n_frames={n_frames}, guidance={guidance_param}, seed={seed}")

    # --- Prepare Model Keyword Arguments (model_kwargs) ---
    texts = [text_prompt]
    collate_args = [{'inp': torch.zeros(n_frames), 'tokens': None, 'lengths': n_frames, 'text': texts[0]}]
    _, model_kwargs = collate(collate_args)

    # Move model_kwargs tensors to the correct device
    for k, v in model_kwargs['y'].items():
        if torch.is_tensor(v):
            model_kwargs['y'][k] = v.to(dist_util.dev())
    
    # Add guidance scale to batch
    if guidance_param != 1.0:
        model_kwargs['y']['scale'] = torch.ones(batch_size, device=dist_util.dev()) * guidance_param

    # Encode text prompt
    # model.encode_text is available on the (potentially wrapped) model
    print("Encoding text prompt...")
    with torch.no_grad(): # Ensure no gradients are computed for text encoding
        model_kwargs['y']['text_embed'] = model.encode_text(model_kwargs['y']['text'])
    print("Text prompt encoded.")

    # --- Perform Sampling ---
    sample_fn = diffusion.p_sample_loop # Assuming standard sampling, not autoregressive for API

    motion_shape = (batch_size, model.njoints, model.nfeats, n_frames)

    print(f"Starting sampling with shape: {motion_shape}...")
    sample_time_start = torch.cuda.Event(enable_timing=True)
    sample_time_end = torch.cuda.Event(enable_timing=True)
    sample_time_start.record()

    sample = sample_fn(
        model,
        motion_shape,
        clip_denoised=False,
        model_kwargs=model_kwargs,
        skip_timesteps=0,
        init_image=None, # Not used for text-to-motion
        progress=False,  # Disable progress bar for server
        dump_steps=None,
        noise=None,
        const_noise=False,
    )
    sample_time_end.record()
    torch.cuda.synchronize()
    el_time = sample_time_start.elapsed_time(sample_time_end)
    print(f"Sampling finished (took {el_time / 1000.0:.2f}s).")


    # --- Post-process: Recover XYZ positions ---
    # This part uses `data.dataset.t2m_dataset.inv_transform` and `recover_from_ric`
    # which requires the `data` object passed in.
    if model.data_rep == 'hml_vec':
        print("Recovering from RIC representation...")
        # n_joints_rec = 22 if sample.shape[1] == 263 else 21 # model.njoints should be correct
        n_joints_rec = model.njoints 
        # The sample from p_sample_loop is [bs, njoints, nfeats, nframes]
        # inv_transform expects [bs, nframes, nfeats, njoints] ? No, it expects [bs, features, 1, nframes]
        # Let's check inv_transform input: (bs, d, 1, points) where d is feature_dim (e.g. 263 or 251)
        # And sample is (bs, njoints, nfeats_per_joint, nframes)
        # This needs to be reshaped/permuted to (bs, total_features, 1, nframes)
        # For hml_vec, sample is [bs, 263, 1, nframes] if model.njoints already reflects the full vector dim
        # The original generate.py does: sample.cpu().permute(0, 2, 3, 1) -> [bs, 1, nframes, features]
        # then inv_transform -> [bs, nframes, njoints, nfeats_ric]
        # then recover_from_ric expects [bs, nframes, njoints, 3]
        # then permute back.
        # Let's align with MDM's generate.py:
        # sample is [bs, model.njoints, model.nfeats, n_frames] but for hml_vec, model.njoints is the feature dim (263) and model.nfeats is 1.
        # So sample is [bs, 263, 1, n_frames]
        
        # Original: sample = data.dataset.t2m_dataset.inv_transform(sample.cpu().permute(0, 2, 3, 1)).float()
        # Permute sample from [bs, feat_dim, 1, nframes] to [bs, 1, nframes, feat_dim]
        sample_for_inv = sample.cpu().permute(0, 2, 3, 1) 
        ric_data = data.dataset.t2m_dataset.inv_transform(sample_for_inv).float() # ric_data is [bs, 1, nframes, feat_dim]
        
        # Reshape ric_data to be 3D [bs, nframes, feat_dim] for recover_from_ric
        # Original shape example: [1, 1, 120, 263]
        # Target shape for recover_from_ric's 3D path: [1, 120, 263]
        current_bs = ric_data.shape[0]
        current_nframes = ric_data.shape[2] # n_frames variable from earlier scope
        current_feat_dim = ric_data.shape[3]
        ric_data_for_recovery = ric_data.reshape(current_bs, current_nframes, current_feat_dim)
        print(f"Reshaped ric_data from {ric_data.shape} to {ric_data_for_recovery.shape} for recover_from_ric")

        # recover_from_ric expects joints to be 22 for HumanML3D or 21 for KIT
        # These are hardcoded in original generate.py based on sample.shape[1] after inv_transform,
        # or rather, fixed n_joints based on dataset before inv_transform.
        # For HumanML3D, it's 22 joints for RIC.
        num_joints_for_ric = 22 if mdm_init_args.dataset == 'humanml' else 21
        
        sample_xyz_ric = recover_from_ric(ric_data_for_recovery, num_joints_for_ric) # sample_xyz_ric is [bs, nframes, njoints_actual, 3]
        # Reshape and permute to final format [bs, njoints_actual, 3, nframes]
        # Original: sample = sample.view(-1, *sample.shape[2:]).permute(0, 2, 3, 1)
        # This permutes from [bs, nframes, njoints, 3] to [bs, njoints, 3, nframes]
        sample = sample_xyz_ric.permute(0, 2, 3, 1)
        print("Recovery from RIC finished.")


    # --- Post-process: Convert to XYZ (model.rot2xyz) ---
    # This is applied if data_rep is not 'xyz' already, or if it was 'hml_vec' (already handled)
    # rot2xyz_pose_rep = 'xyz' if model.data_rep in ['xyz', 'hml_vec'] else model.data_rep
    # The `sample` is now in a representation that rot2xyz can take, e.g. [bs, njoints, 3, nframes] for positions.
    # If original data_rep was rotations, rot2xyz converts them.
    # If it was hml_vec, it's already xyz positions.
    # If it was xyz, no conversion needed by rot2xyz.
    
    # Let's simplify: if model.data_rep was not 'hml_vec', we might need rot2xyz.
    # If it was 'hml_vec', sample is already [bs, njoints, 3, nframes] in world coords from recover_from_ric
    # If model.data_rep is 'xyz', sample is already [bs, njoints, 3, nframes] in model's joint definition
    # If model.data_rep involves rotations, rot2xyz is needed.

    if model.data_rep not in ['hml_vec', 'xyz']: # e.g. 'rot6d', 'rotvec'
        print(f"Converting to XYZ from representation: {model.data_rep} using model.rot2xyz...")
        # rot2xyz needs a mask of True values for the sequence length
        rot2xyz_mask = torch.ones(batch_size, n_frames, device=dist_util.dev()).bool()
        sample = model.rot2xyz(x=sample, mask=rot2xyz_mask, pose_rep=model.data_rep, glob=True, translation=True,
                               jointstype='smpl', # or 'kit' - depends on model.dataset
                               vertstrans=False, # Usually false for just motion
                               betas=None, beta=0, glob_rot=None,
                               get_rotations_back=False)
        print("Conversion to XYZ finished.")
    elif model.data_rep == 'xyz' and not getattr(model, 'is_absolute_xyz', False):
        # some xyz models might be relative and need to be made absolute,
        # but usually generate.py doesn't have specific logic for this beyond recover_from_ric for hml_vec
        print("Data representation is XYZ. Assuming absolute or handled by model.")


    # Ensure sample is on CPU and NumPy format for returning
    # Sample is expected to be [bs, njoints, 3, nframes] (an XYZ position format)
    # For a single sample (bs=1), squeeze the batch dimension.
    motion_data_np = sample.squeeze(0).cpu().numpy() # Shape: [njoints, 3, nframes]

    print(f"--- Motion generation complete. Output shape: {motion_data_np.shape} ---")
    return motion_data_np 