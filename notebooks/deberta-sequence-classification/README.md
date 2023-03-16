# Deberta Sequence Classification with OpenVINO

NLI involves determining the correlation between two given texts. Specifically, the model inputs a premise and a hypothesis, and outputs one of the following classes:


| Entailment |
| ------- | 
|Entailment, which indicates that the hypothesis is true.         |


| Contradiction | 
| ------- | 
| Contradiction, which indicates that the hypothesis is false.        |

| Neutral |
| ------- | 
| Neutral, which indicates that there is no relation between the hypothesis and the premise.        |

In this notebook, OpenVINO is utilized to carry out NLI tasks. Specifically, Hugging Face's [microsoft/deberta-base-mnli model](https://huggingface.co/microsoft/deberta-base-mnli), which is based on the Transformer architecture, will be employed. To make use of the Hugging Face model, it is necessary to first convert it to the ONNX format via the torch.onnx.export function, followed by a conversion to the OpenVINO IR format.  Upon analyzing the hypothesis presented in the text, the model is capable of predicting one of several possible categories: Entailment, Contradiction or Neutral.


```
User Input:  I love you. I like you.

probability:  [0.00127007 0.24879578 0.74993414]
Label:  ENTAILMENT 

User Input:  That dog is cute. Today is a nice day.

probability:  [0.01947436 0.92858535 0.05194035]
Label:  NEUTRAL 

User Input:  I hate you. I think what you said makes some sense.
probability:  [0.7899069  0.20804428 0.00204879]
Label:  CONTRADICTION 
```

