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
                    label="Step 1: Load text files",
                    value=[text_example_path],
                    file_count="multiple",
                    file_types=[
                        ".csv",
                        ".doc",
                        ".docx",
                        ".enex",
                        ".epub",
                        ".html",
                        ".md",
                        ".odt",
                        ".pdf",
                        ".ppt",
                        ".pptx",
                        ".txt",
                    ],
                )
                load_docs = gr.Button("Step 2: Build Vector Store", variant="primary")
                db_argument = gr.Accordion("Vector Store Configuration", open=False)
                with db_argument:
                    spliter = gr.Dropdown(
                        ["Character", "RecursiveCharacter", "Markdown", "Chinese"],
                        value="RecursiveCharacter",
                        label="Text Spliter",
                        info="Method used to splite the documents",
                        multiselect=False,
                    )

                    chunk_size = gr.Slider(
                        label="Chunk size",
                        value=400,
                        minimum=50,
                        maximum=2000,
                        step=50,
                        interactive=True,
                        info="Size of sentence chunk",
                    )

                    chunk_overlap = gr.Slider(
                        label="Chunk overlap",
                        value=50,
                        minimum=0,
                        maximum=400,
                        step=10,
                        interactive=True,
                        info=("Overlap between 2 chunks"),
                    )

                langchain_status = gr.Textbox(
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
                    height=800,
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
                            stop = gr.Button("Stop", visible=stop_fn is not None)
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
                            hide_context = gr.Checkbox(
                                value=True,
                                label="Hide searching result in prompt",
                                interactive=True,
                            )
                        with gr.Row():
                            search_method = gr.Dropdown(
                                ["similarity_score_threshold", "similarity", "mmr"],
                                value="similarity_score_threshold",
                                label="Searching Method",
                                info="Method used to search vector store",
                                multiselect=False,
                                interactive=True,
                            )
                        with gr.Row():
                            score_threshold = gr.Slider(
                                0.01,
                                0.99,
                                value=0.5,
                                step=0.01,
                                label="Similarity Threshold",
                                info="Only working for 'similarity score threshold' method",
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
        docs.clear(clear_files, outputs=[langchain_status], queue=False)
        load_docs.click(
            fn=load_doc_fn,
            inputs=[docs, spliter, chunk_size, chunk_overlap, vector_search_top_k, vector_rerank_top_n, do_rerank, search_method, score_threshold],
            outputs=[langchain_status],
            queue=False,
        )
        submit_event = msg.submit(handle_user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
            run_fn,
            [chatbot, temperature, top_p, top_k, repetition_penalty, hide_context, do_rag],
            chatbot,
            queue=True,
        )
        submit_click_event = submit.click(handle_user_message, [msg, chatbot], [msg, chatbot], queue=False).then(
            run_fn,
            [chatbot, temperature, top_p, top_k, repetition_penalty, hide_context, do_rag],
            chatbot,
            queue=True,
        )
        if stop_fn is not None:
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
            [vector_search_top_k, vector_rerank_top_n, do_rerank, search_method, score_threshold],
            outputs=[langchain_status],
        )
        vector_rerank_top_n.release(
            update_retriever_fn,
            inputs=[vector_search_top_k, vector_rerank_top_n, do_rerank, search_method, score_threshold],
            outputs=[langchain_status],
        )
        do_rerank.change(
            update_retriever_fn,
            inputs=[vector_search_top_k, vector_rerank_top_n, do_rerank, search_method, score_threshold],
            outputs=[langchain_status],
        )
        search_method.change(
            update_retriever_fn,
            inputs=[vector_search_top_k, vector_rerank_top_n, do_rerank, search_method, score_threshold],
            outputs=[langchain_status],
        )
        score_threshold.change(
            update_retriever_fn,
            inputs=[vector_search_top_k, vector_rerank_top_n, do_rerank, search_method, score_threshold],
            outputs=[langchain_status],
        )
    return demo
