import os
import torch
import gradio as gr
import langid
import se_extractor

supported_languages = ['zh', 'en']

def build_predict(output_dir, tone_color_converter, en_tts_model, zh_tts_model, en_source_default_se, en_source_style_se, zh_source_se):
    def predict(prompt, style, audio_file_pth, agree):
        return predict_impl(prompt, style, audio_file_pth, agree, output_dir, tone_color_converter, en_tts_model, zh_tts_model, en_source_default_se, en_source_style_se, zh_source_se)
    return predict

def predict_impl(prompt, style, audio_file_pth, agree, output_dir, tone_color_converter, en_tts_model, zh_tts_model, en_source_default_se, en_source_style_se, zh_source_se):
    text_hint = ''
    if agree == False:
        text_hint += '[ERROR] Please accept the Terms & Condition!\n'
        gr.Warning("Please accept the Terms & Condition!")
        return (
            text_hint,
            None,
            None,
        )

    language_predicted = langid.classify(prompt)[0].strip()  
    print(f"Detected language:{language_predicted}")

    if language_predicted not in supported_languages:
        text_hint += f"[ERROR] The detected language {language_predicted} for your input text is not in our Supported Languages: {supported_languages}\n"
        gr.Warning(
            f"The detected language {language_predicted} for your input text is not in our Supported Languages: {supported_languages}"
        )

        return (
            text_hint,
            None,
        )
    
    if language_predicted == "zh":
        tts_model = zh_tts_model
        source_se = zh_source_se
        language = 'Chinese'
        if style not in ['default']:
            text_hint += f"[ERROR] The style {style} is not supported for Chinese, which should be in ['default']\n"
            gr.Warning(f"The style {style} is not supported for Chinese, which should be in ['default']")
            return (
                text_hint,
                None,
            )

    else:
        tts_model = en_tts_model
        if style == 'default':
            source_se = en_source_default_se
        else:
            source_se = en_source_style_se
        language = 'English'
        supported_styles = ['default', 'whispering', 'shouting', 'excited', 'cheerful', 'terrified', 'angry', 'sad', 'friendly']
        if style not in supported_styles:
            text_hint += f"[ERROR] The style {style} is not supported for English, which should be in {*supported_styles,}\n"
            gr.Warning(f"The style {style} is not supported for English, which should be in {*supported_styles,}")
            return (
                text_hint,
                None,
            )

    speaker_wav = audio_file_pth

    if len(prompt) < 2:
        text_hint += f"[ERROR] Please give a longer prompt text \n"
        gr.Warning("Please give a longer prompt text")
        return (
            text_hint,
            None,
        )
    if len(prompt) > 200:
        text_hint += f"[ERROR] Text length limited to 200 characters for this demo, please try shorter text. You can clone our open-source repo and try for your usage \n"
        gr.Warning(
            "Text length limited to 200 characters for this demo, please try shorter text. You can clone our open-source repo for your usage"
        )
        return (
            text_hint,
            None,
        )
    
    # note diffusion_conditioning not used on hifigan (default mode), it will be empty but need to pass it to model.inference
    try:
        target_se, audio_name = se_extractor.get_se(speaker_wav, tone_color_converter, target_dir='processed', vad=True)
    except Exception as e:
        text_hint += f"[ERROR] Get target tone color error {str(e)} \n"
        gr.Warning(
            "[ERROR] Get target tone color error {str(e)} \n"
        )
        return (
            text_hint,
            None,
        )

    src_path = f'{output_dir}/tmp.wav'
    tts_model.tts(prompt, src_path, speaker=style, language=language)

    save_path = f'{output_dir}/output.wav'
    encode_message = "@MyShell"
    tone_color_converter.convert(
        audio_src_path=src_path, 
        src_se=source_se, 
        tgt_se=target_se, 
        output_path=save_path,
        message=encode_message)

    text_hint += f'''Get response successfully \n'''

    return (
        text_hint,
        save_path,
    )

description = """
    # OpenVoice accelerated by OpenVINO:
    
    a versatile instant voice cloning approach that requires only a short audio clip from the reference speaker to replicate their voice and generate speech in multiple languages. OpenVoice enables granular control over voice styles, including emotion, accent, rhythm, pauses, and intonation, in addition to replicating the tone color of the reference speaker. OpenVoice also achieves zero-shot cross-lingual voice cloning for languages not included in the massive-speaker training set.
"""

content = """
<div>
<strong>If the generated voice does not sound like the reference voice, please refer to <a href='https://github.com/myshell-ai/OpenVoice/blob/main/docs/QA.md'>this QnA</a>.</strong> <strong>For multi-lingual & cross-lingual examples, please refer to <a href='https://github.com/myshell-ai/OpenVoice/blob/main/demo_part2.ipynb'>this jupyter notebook</a>.</strong>
This online demo mainly supports <strong>English</strong>. The <em>default</em> style also supports <strong>Chinese</strong>. But OpenVoice can adapt to any other language as long as a base speaker is provided.
</div>
"""
wrapped_markdown_content = f"<div style='border: 1px solid #000; padding: 10px;'>{content}</div>"


examples = [
    [
        "今天天气真好，我们一起出去吃饭吧。",
        'default',
        "OpenVoice/resources/demo_speaker1.mp3",
        True,
    ],[
        "This audio is generated by open voice with a half-performance model.",
        'whispering',
        "OpenVoice/resources/demo_speaker2.mp3",
        True,
    ],
    [
        "He hoped there would be stew for dinner, turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick, peppered, flour-fattened sauce.",
        'sad',
        "OpenVoice/resources/demo_speaker0.mp3",
        True,
    ],
]

def get_demo(output_dir, tone_color_converter, en_tts_model, zh_tts_model, en_source_default_se, en_source_style_se, zh_source_se):
    with gr.Blocks(analytics_enabled=False) as demo:

        with gr.Row():
            gr.Markdown(description)
        with gr.Row():
            gr.HTML(wrapped_markdown_content)

        with gr.Row():
            with gr.Column():
                input_text_gr = gr.Textbox(
                    label="Text Prompt",
                    info="One or two sentences at a time is better. Up to 200 text characters.",
                    value="He hoped there would be stew for dinner, turnips and carrots and bruised potatoes and fat mutton pieces to be ladled out in thick, peppered, flour-fattened sauce.",
                )
                style_gr = gr.Dropdown(
                    label="Style",
                    info="Select a style of output audio for the synthesised speech. (Chinese only support 'default' now)",
                    choices=['default', 'whispering', 'cheerful', 'terrified', 'angry', 'sad', 'friendly'],
                    max_choices=1,
                    value="default",
                )
                ref_gr = gr.Audio(
                    label="Reference Audio",
                    # info="Click on the ✎ button to upload your own target speaker audio",
                    type="filepath",
                    value="OpenVoice/resources/demo_speaker2.mp3",
                )
                tos_gr = gr.Checkbox(
                    label="Agree",
                    value=False,
                    info="I agree to the terms of the cc-by-nc-4.0 license-: https://github.com/myshell-ai/OpenVoice/blob/main/LICENSE",
                )

                tts_button = gr.Button("Send", elem_id="send-btn", visible=True)


            with gr.Column():
                out_text_gr = gr.Text(label="Info")
                audio_gr = gr.Audio(label="Synthesised Audio", autoplay=True)
                # ref_audio_gr = gr.Audio(label="Reference Audio Used")
                predict = build_predict(
                    output_dir, 
                    tone_color_converter, 
                    en_tts_model, 
                    zh_tts_model, 
                    en_source_default_se, 
                    en_source_style_se, 
                    zh_source_se
                )

                gr.Examples(examples,
                            label="Examples",
                            inputs=[input_text_gr, style_gr, ref_gr, tos_gr],
                            outputs=[out_text_gr, audio_gr],
                            fn=predict,
                            cache_examples=False,)
                tts_button.click(predict, [input_text_gr, style_gr, ref_gr, tos_gr], outputs=[out_text_gr, audio_gr])
    return demo
