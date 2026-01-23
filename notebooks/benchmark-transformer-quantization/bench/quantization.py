from transformers import BitsAndBytesConfig

class QuantizationConfigFactory:
    """
    Returns valid quantization configurations for OpenVINO/NNCF.
    """
    @staticmethod
    def get_int4_config():
        """
        Returns NNCF Weight Compression config for 4-bit.
        Sym=True and GroupSize=128 are standard for CPU accuracy/speed balance.
        """
        return {
            "load_in_4bit": True,
            "quantization_config": {
                "bits": 4, 
                "sym": True, 
                "group_size": 128,
                "ratio": 1.0,
            }
        }
        

    @staticmethod
    def get_int8_config():
        """
        Returns NNCF Weight Compression config for 8-bit.
        """
        return {
            "load_in_8bit": True,
        }