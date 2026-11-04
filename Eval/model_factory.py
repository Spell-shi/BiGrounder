"""Model factory: dispatch --model_type to the corresponding model wrapper.

Used by eval_screenspot_pro.py. The GPU is already pinned by the caller
(torch.cuda.set_device in the worker initializer), so `device` is accepted
for interface compatibility.
"""


def build_model(args, device='cuda:0'):
    model_type = args.model_type
    model_name_or_path = args.model_name_or_path

    if model_type == "BiGrounder":
        # Instantiate the BiGrounder model wrapper.
        # Note: load_model does not take a device argument; the GPU is already
        # pinned by the worker initializer, so only the path is passed.
        from models.bigrounder import BiGrounderModel

        model = BiGrounderModel()
        if args.model_name_or_path:
            model.load_model(model_name_or_path=model_name_or_path)
        else:
            model.load_model()

    else:
        raise ValueError(f"Unsupported model type {model_type}.")

    model.set_generation_config(temperature=0, max_new_tokens=256)
    return model
