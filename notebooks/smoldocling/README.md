# Document conversion with SmolDocling and OpenVINO

<div style="display: flex; align-items: center;">
<img src="https://github.com/user-attachments/assets/52a86ee7-065c-4ca5-b0b3-2458a79a6c9d" alt="SmolDocling" style="width: 200px; height: auto; margin-right: 20px;">
<div>
        <p>SmolDocling is a multimodal Image-Text-to-Text model designed for efficient document conversion. It retains Docling's most popular features while ensuring full compatibility with <a href=https://github.com/docling-project/docling> Docling </a> through seamless support for <strong>DoclingDocuments</strong> (The JSON that can be exported using Docling follows this schema. The DoclingDocument type also models the metadata that may be attached to the converted document.)
        </p>
</div>
</div>
🚀 Model Features:  
- 🏷️ **DocTags for Efficient Tokenization** – Introduces DocTags an efficient and minimal representation for documents that is fully compatible with **DoclingDocuments**.  
- 🔍 **OCR (Optical Character Recognition)** – Extracts text accurately from images.  
- 📐 **Layout and Localization** – Preserves document structure and document element **bounding boxes**.  
- 💻 **Code Recognition** – Detects and formats code blocks including identation.  
- 🔢 **Formula Recognition** – Identifies and processes mathematical expressions.  
- 📊 **Chart Recognition** – Extracts and interprets chart data.  
- 📑 **Table Recognition** – Supports column and row headers for structured table extraction.  
- 🖼️ **Figure Classification** – Differentiates figures and graphical elements.  
- 📝 **Caption Correspondence** – Links captions to relevant images and figures.  
- 📜 **List Grouping** – Organizes and structures list elements correctly.  
- 📄 **Full-Page Conversion** – Processes entire pages for comprehensive document conversion including all page elements (code, equations, tables, charts etc.) 
- 🔲 **OCR with Bounding Boxes** – OCR regions using a bounding box.
- 📂 **General Document Processing** – Trained for both scientific and non-scientific documents.  
- 🔄 **Seamless Docling Integration** – Import into **Docling** and export in multiple formats.

More details about model can be found in the [model card](https://huggingface.co/ds4sd/SmolDocling-256M-preview) and [paper](https://huggingface.co/papers/2503.11576).

In this tutorial we consider how to convert and run SmolDocling model using OpenVINO [Optimum Intel](https://github.com/huggingface/optimum-intel) integration. Additionally, we demonstrate how to apply model optimization techniques like weights compression using [NNCF](https://github.com/openvinotoolkit/nncf).




<img referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=5b5a4db0-7875-4bfb-bdbd-01698b5b1a77&file=notebooks/smoldocling/README.md" />
