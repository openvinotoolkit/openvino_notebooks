from typing import Callable, Literal
import gradio as gr

chinese_examples = [
    ["英特尔®酷睿™ Ultra处理器可以降低多少功耗？"],
    ["相比英特尔之前的移动处理器产品，英特尔®酷睿™ Ultra处理器的AI推理性能提升了多少？"],
    ["英特尔博锐® Enterprise系统提供哪些功能？"],
]

english_examples = [
    ["How much power consumption can Intel® Core™ Ultra Processors help save?"],
    ["Compared to Intel’s previous mobile processor, what is the advantage of Intel® Core™ Ultra Processors for Artificial Intelligence?"],
    ["What can Intel vPro® Enterprise systems offer?"],
]


def clear_files():
    return "Vector Store is Not ready"


def handle_user_message(message, history):
    """
    callback function for updating user messages in interface on submit button click

    Params:
      message: current message
      history: conversation history
    Returns:
      None
    """
    # Append the user's message to the conversation history
    return "", history + [[message, ""]]


def make_demo(
    load_doc_fn: Callable,
    run_fn: Callable,
    stop_fn: Callable,
    update_retriever_fn: Callable,
    model_name: str,
    language: Literal["English", "Chinese"] = "English",
):
    examples = chinese_examples if (language == "Chinese") else english_examples

    if language == "English":
        text_example_path = "text_example_en.pdf"
    else:
        text_example_path = "text_example_cn.pdf"

    with gr.Blocks(
        theme=gr.themes.Soft(),
        css=".disclaimer {font-variant-caps: all-small-caps;}",
    ) as demo:
        gr.Markdown("""<h1><center>QA over Document</center></h1>""")
        gr.Markdown(f"""<center>Powered by OpenVINO and {model_name} </center>""")
        with gr.Row():
            with gr.Column(scale=1):
                docs = gr.File(
                    label="Step 1: Load a PDF file",
                    value=text_example_path,
                    file_types=[
                        ".pdf",
                    ],
                )
                load_docs = gr.Button("Step 2: Build Vector Store", variant="primary")
                db_argument = gr.Accordion("Vector Store Configuration", open=False)
                with db_argument:
                    spliter = gr.Dropdown(
                        ["SentenceSplitter", "RecursiveCharacter"],
                        value="SentenceSplitter",
                        label="Text Spliter",
                        info="Method used to splite the documents",
                        multiselect=False,
                    )

                    chunk_size = gr.Slider(
                        label="Chunk size",
                        value=200,
                        minimum=50,
                        maximum=2000,
                        step=50,
                        interactive=True,
                        info="Size of sentence chunk",
                    )

                    chunk_overlap = gr.Slider(
                        label="Chunk overlap",
                        value=20,
                        minimum=0,
                        maximum=400,
                        step=10,
                        interactive=True,
                        info=("Overlap between 2 chunks"),
                    )

                vector_store_status = gr.Textbox(
                    label="Vector Store Status",
                    value="Vector Store is Ready",
                    interactive=False,
                )
                do_rag = gr.Checkbox(
                    value=True,
                    label="RAG is ON",
                    interactive=True,
                    info="Whether to do RAG for generation",
                )
                with gr.Accordion("Generation Configuration", open=False):
                    with gr.Row():
                        with gr.Column():
                            with gr.Row():
                                temperature = gr.Slider(
                                    label="Temperature",
                                    value=0.1,
                                    minimum=0.0,
                                    maximum=1.0,
                                    step=0.1,
                                    interactive=True,
                                    info="Higher values produce more diverse outputs",
                                )
                        with gr.Column():
                            with gr.Row():
                                top_p = gr.Slider(
                                    label="Top-p (nucleus sampling)",
                                    value=1.0,
                                    minimum=0.0,
                                    maximum=1,
                                    step=0.01,
                                    interactive=True,
                                    info=(
                                        "Sample from the smallest possible set of tokens whose cumulative probability "
                                        "exceeds top_p. Set to 1 to disable and sample from all tokens."
                                    ),
                                )
                        with gr.Column():
                            with gr.Row():
                                top_k = gr.Slider(
                                    label="Top-k",
                                    value=50,
                                    minimum=0.0,
                                    maximum=200,
                                    step=1,
                                    interactive=True,
                                    info="Sample from a shortlist of top-k tokens — 0 to disable and sample from all tokens.",
                                )
                        with gr.Column():
                            with gr.Row():
                                repetition_penalty = gr.Slider(
                                    label="Repetition Penalty",
                                    value=1.1,
                                    minimum=1.0,
                                    maximum=2.0,
                                    step=0.1,
                                    interactive=True,
                                    info="Penalize repetition — 1.0 to disable.",
                                )
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    height=600,
                    label="Step 3: Input Query",
                )
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            msg = gr.Textbox(
                                label="QA Message Box",
                                placeholder="Chat Message Box",
                                show_label=False,
                                container=False,
                            )
                    with gr.Column():
                        with gr.Row():
                            submit = gr.Button("Submit", variant="primary")
                            stop = gr.Button("Stop")
                            clear = gr.Button("Clear")
                gr.Examples(examples, inputs=msg, label="Click on any example and press the 'Submit' button")
                retriever_argument = gr.Accordion("Retriever Configuration", open=True)
                with retriever_argument:
                    with gr.Row():
                        with gr.Row():
                            do_rerank = gr.Checkbox(
                                value=True,
                                label="Rerank searching result",
                                interactive=True,
                            )
                        with gr.Row():
                            vector_rerank_top_n = gr.Slider(
                                1,
                                10,
                                value=2,
                                step=1,
                                label="Rerank top n",
                                info="Number of rerank results",
                                interactive=True,
                            )
                        with gr.Row():
                            vector_search_top_k = gr.Slider(
                                1,
                                50,
                                value=10,
                                step=1,
                                label="Search top k",
                                info="Search top k must >= Rerank top n",
                                interactive=True,
                            )
        docs.clear(clear_files, outputs=[vector_store_status], queue=False)
        load_docs.click(
            fn=load_doc_fn,
            inputs=[docs, spliter, chunk_size, chunk_overlap, vector_search_top_k, vector_rerank_top_n, do_rerank],
            outputs=[vector_store_status],
            queue=False,
        )
        submit_event = msg.submit(handle_user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
            run_fn,
            [chatbot, temperature, top_p, top_k, repetition_penalty, do_rag],
            chatbot,
            queue=True,
        )
        submit_click_event = submit.click(handle_user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
            run_fn,
            [chatbot, temperature, top_p, top_k, repetition_penalty, do_rag],
            chatbot,
            queue=True,
        )
        stop.click(
            fn=stop_fn,
            inputs=None,
            outputs=None,
            cancels=[submit_event, submit_click_event],
            queue=False,
        )
        clear.click(lambda: None, None, chatbot, queue=False)
        vector_search_top_k.release(
            update_retriever_fn,
            [vector_search_top_k, vector_rerank_top_n, do_rerank],
        )
        vector_rerank_top_n.release(
            update_retriever_fn,
            [vector_search_top_k, vector_rerank_top_n, do_rerank],
        )
        do_rerank.change(
            update_retriever_fn,
            [vector_search_top_k, vector_rerank_top_n, do_rerank],
        )
    return demo
