from hyper.models import HYPER, TransductiveHCNet
from hyper_text.text_models import TextHYPER, TextTransductiveHCNet


def build_model(cfg):
    use_grel = cfg.model.get("use_grel", True)   # آیا G-rel/RelHCNet استفاده شود
    use_text = cfg.model.get("use_text", True)   # آیا فیوژن متنی فعال شود

    if use_grel and use_text:
        return TextHYPER(cfg.model.relation_model, cfg.model.entity_model)
    if use_grel and not use_text:
        return HYPER(cfg.model.relation_model, cfg.model.entity_model)
    if (not use_grel) and use_text:
        return TextTransductiveHCNet(cfg.model.entity_model, cfg.model.num_relations)
    return TransductiveHCNet(cfg.model.entity_model, cfg.model.num_relations)