from optimum.intel.openvino import OVDiffusionPipeline
from pipeline import Flex2Pipeline


class OVFlex2Pipeline(OVDiffusionPipeline, Flex2Pipeline):
    main_input_name = "prompt"
    export_feature = "text-to-image"
    auto_model_class = Flex2Pipeline
