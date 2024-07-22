from models.segment_anything.build_sam import sam_model_registry
from models.segment_anything_memsam.build_memsam import memsam_model_registry

def get_model(modelname="SAM", args=None):
    if modelname == "SAM":
        pass
    if modelname == "SAMUS":
        pass
    elif modelname == "MemSAM":
        model = memsam_model_registry['vit_b'](args=args, checkpoint=args.sam_ckpt)
    else:
        raise RuntimeError("Could not find the model:", modelname)
    return model
