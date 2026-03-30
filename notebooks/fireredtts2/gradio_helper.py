import re
import gradio as gr
from typing import Literal, List

examples = [
    [
        "English",
        "FireRedTTS2/examples/chat_prompt/en/S1.flac",
        "[S1]I think we should just talk about what happened and move on because there's going to be other jousts and Sir Saif isn't done yet. It's not, he's not, it's not done yet.",
        "FireRedTTS2/examples/chat_prompt/en/S2.flac",
        "[S2]You know, maybe sorry, maybe maybe I pushed, maybe I pushed too hard. I was really excited. I didn't mean to make you snap.",
        "[S1]It's alright, we'll take a breath and plan the next pass together.[S2]Yeah, thanks. We'll get it right this time.[S1]Let's review our signals tonight so we're in sync on the field tomorrow.",
    ],
    [
        "中文",
        "FireRedTTS2/examples/chat_prompt/zh/S1.flac",
        "[S1]啊，可能说更适合美国市场应该是什么样子。那这这个可能说当然如果说有有机会能亲身的去考察去了解一下，那当然是有更好的帮助。",
        "FireRedTTS2/examples/chat_prompt/zh/S2.flac",
        "[S2]比如具体一点的，他觉得最大的一个跟他预想的不一样的是在什么地方。",
        "[S1]那可能说对对，没有去过美国来说去去看到美国线下。巴斯曼也好，沃尔玛也好，他们线下不管说，因为深圳出去的还是电子周边的会表达，会发现哇对这个价格真的是很高呀。都是卖三十五美金、四十美金，甚至一个手机壳，就是二十五美金开。[S2]对，没错，我每次都觉得不不可思议。我什么人会买三五十美金的手机壳？但是其实在在那个target啊，就塔吉特这种超级市场，大家都是这样的，定价也很多人买。",
    ],
]


def initiate_model(ov_model):
    global model
    model = ov_model


# ================================================
#                   Gradio
# ================================================

# i18n
_i18n_key2lang_dict = dict(
    # Title markdown
    title_md_desc=dict(
        en="FireRedTTS-2 🔥 Dialogue Generation",
        zh="FireRedTTS-2 🔥 对话生成",
    ),
    # Voice mode radio
    voice_mode_label=dict(
        en="Voice Mode",
        zh="音色模式",
    ),
    voice_model_choice1=dict(
        en="Voice Clone",
        zh="音色克隆",
    ),
    voice_model_choice2=dict(
        en="Random Voice",
        zh="随机音色",
    ),
    # Speaker1 Prompt
    spk1_prompt_audio_label=dict(
        en="Speaker 1 Prompt Audio",
        zh="说话人 1 参考语音",
    ),
    spk1_prompt_text_label=dict(
        en="Speaker 1 Prompt Text",
        zh="说话人 1 参考文本",
    ),
    spk1_prompt_text_placeholder=dict(
        en="[S1] text of speaker 1 prompt audio.",
        zh="[S1] 说话人 1 参考文本",
    ),
    # Speaker2 Prompt
    spk2_prompt_audio_label=dict(
        en="Speaker 2 Prompt Audio",
        zh="说话人 2 参考语音",
    ),
    spk2_prompt_text_label=dict(
        en="Speaker 2 Prompt Text",
        zh="说话人 2 参考文本",
    ),
    spk2_prompt_text_placeholder=dict(
        en="[S2] text of speaker 2 prompt audio.",
        zh="[S2] 说话人 2 参考文本",
    ),
    # Dialogue input textbox
    dialogue_text_input_label=dict(
        en="Dialogue Text Input",
        zh="对话文本输入",
    ),
    dialogue_text_input_placeholder=dict(
        en="[S1]text[S2]text[S1]text...",
        zh="[S1]文本[S2]文本[S1]文本...",
    ),
    # Generate button
    generate_btn_label=dict(
        en="Generate Audio",
        zh="合成",
    ),
    # Generated audio
    generated_audio_label=dict(
        en="Generated Dialogue Audio",
        zh="合成的对话音频",
    ),
    # Warining1: invalid text for prompt
    warn_invalid_spk1_prompt_text=dict(
        en='Invalid speaker 1 prompt text, should strictly follow: "[S1]xxx"',
        zh='说话人 1 参考文本不合规，格式："[S1]xxx"',
    ),
    warn_invalid_spk2_prompt_text=dict(
        en='Invalid speaker 2 prompt text, should strictly follow: "[S2]xxx"',
        zh='说话人 2 参考文本不合规，格式："[S2]xxx"',
    ),
    # Warining2: invalid text for dialogue input
    warn_invalid_dialogue_text=dict(
        en='Invalid dialogue input text, should strictly follow: "[S1]xxx[S2]xxx..."',
        zh='对话文本输入不合规，格式："[S1]xxx[S2]xxx..."',
    ),
    # Warining3: incomplete prompt info
    warn_incomplete_prompt=dict(
        en="Please provide prompt audio and text for both speaker 1 and speaker 2",
        zh="请提供说话人 1 与说话人 2 的参考语音与参考文本",
    ),
)

global_lang: Literal["zh", "en"] = "en"


def i18n(key):
    global global_lang
    return _i18n_key2lang_dict[key][global_lang]


def check_monologue_text(text: str, prefix: str = None) -> bool:
    text = text.strip()
    # Check speaker tags
    if prefix is not None and (not text.startswith(prefix)):
        return False
    # Remove prefix
    if prefix is not None:
        text = text.removeprefix(prefix)
    text = text.strip()
    # If empty?
    if len(text) == 0:
        return False
    return True


def check_dialogue_text(text_list: List[str]) -> bool:
    if len(text_list) == 0:
        return False
    for text in text_list:
        if not (
            check_monologue_text(text, "[S1]") or check_monologue_text(text, "[S2]") or check_monologue_text(text, "[S3]") or check_monologue_text(text, "[S4]")
        ):
            return False
    return True


def dialogue_synthesis_function(
    target_text: str,
    voice_mode: Literal[0, 1] = 0,  # 0 means voice clone
    spk1_prompt_text: str | None = "",
    spk1_prompt_audio: str | None = None,
    spk2_prompt_text: str | None = "",
    spk2_prompt_audio: str | None = None,
):
    # Voice clone mode, check prompt info
    if voice_mode == 0:
        prompt_has_value = [
            spk1_prompt_text != "",
            spk1_prompt_audio is not None,
            spk2_prompt_text != "",
            spk2_prompt_audio is not None,
        ]
        if not all(prompt_has_value):
            gr.Warning(message=i18n("warn_incomplete_prompt"))
            return None
        if not check_monologue_text(spk1_prompt_text, "[S1]"):
            gr.Warning(message=i18n("warn_invalid_spk1_prompt_text"))
            return None
        if not check_monologue_text(spk2_prompt_text, "[S2]"):
            gr.Warning(message=i18n("warn_invalid_spk2_prompt_text"))
            return None
    # Check dialogue text
    target_text_list: List[str] = re.findall(r"(\[S[0-9]\][^\[\]]*)", target_text)
    target_text_list = [text.strip() for text in target_text_list]
    if not check_dialogue_text(target_text_list):
        gr.Warning(message=i18n("warn_invalid_dialogue_text"))
        return None

    # Go synthesis
    progress_bar = gr.Progress(track_tqdm=True)
    prompt_wav_list = None if voice_mode != 0 else [spk1_prompt_audio, spk2_prompt_audio]
    prompt_text_list = None if voice_mode != 0 else [spk1_prompt_text, spk2_prompt_text]
    target_audio = model.generate_dialogue(
        text_list=target_text_list,
        prompt_wav_list=prompt_wav_list,
        prompt_text_list=prompt_text_list,
        temperature=0.9,
        topk=30,
    )
    return (24000, target_audio.squeeze(0).numpy())


# UI rendering
def render_interface() -> gr.Blocks:
    with gr.Blocks(title="FireRedTTS-2", theme=gr.themes.Default()) as page:
        # ======================== UI ========================
        # A large title
        title_desc = gr.Markdown(value="# {}".format(i18n("title_md_desc")))
        with gr.Row():
            lang_choice = gr.Radio(
                choices=["中文", "English"],
                value="中文",
                label="Display Language/显示语言",
                type="index",
                interactive=True,
            )
            voice_mode_choice = gr.Radio(
                choices=[i18n("voice_model_choice1"), i18n("voice_model_choice2")],
                value=i18n("voice_model_choice1"),
                label=i18n("voice_mode_label"),
                type="index",
                interactive=True,
            )
        with gr.Row():
            # ==== Speaker1 Prompt ====
            with gr.Column(scale=1):
                with gr.Group(visible=True) as spk1_prompt_group:
                    spk1_prompt_audio = gr.Audio(
                        label=i18n("spk1_prompt_audio_label"),
                        type="filepath",
                        editable=False,
                        interactive=True,
                    )  # Audio component returns tmp audio path
                    spk1_prompt_text = gr.Textbox(
                        label=i18n("spk1_prompt_text_label"),
                        placeholder=i18n("spk1_prompt_text_placeholder"),
                        lines=3,
                    )
            # ==== Speaker2 Prompt ====
            with gr.Column(scale=1):
                with gr.Group(visible=True) as spk2_prompt_group:
                    spk2_prompt_audio = gr.Audio(
                        label=i18n("spk2_prompt_audio_label"),
                        type="filepath",
                        editable=False,
                        interactive=True,
                    )
                    spk2_prompt_text = gr.Textbox(
                        label=i18n("spk2_prompt_text_label"),
                        placeholder=i18n("spk2_prompt_text_placeholder"),
                        lines=3,
                    )
            # ==== Text input ====
            with gr.Column(scale=2):
                dialogue_text_input = gr.Textbox(
                    label=i18n("dialogue_text_input_label"),
                    placeholder=i18n("dialogue_text_input_placeholder"),
                    lines=18,
                )
        # Generate button
        generate_btn = gr.Button(value=i18n("generate_btn_label"), variant="primary", size="lg")
        # Long output audio
        generate_audio = gr.Audio(
            label=i18n("generated_audio_label"),
            interactive=False,
        )
        gr.Examples(
            examples=examples,
            inputs=[lang_choice, spk1_prompt_audio, spk1_prompt_text, spk2_prompt_audio, spk2_prompt_text, dialogue_text_input],
        )

        # ======================== Action ========================
        # Language action
        def _change_component_language(lang):
            global global_lang
            global_lang = ["zh", "en"][lang]
            return [
                # title_desc
                gr.update(value="# {}".format(i18n("title_md_desc"))),
                # voice_mode_choice
                gr.update(
                    choices=[i18n("voice_model_choice1"), i18n("voice_model_choice2")],
                    value=i18n("voice_model_choice1"),
                    label=i18n("voice_mode_label"),
                ),
                # spk1_prompt_{audio,text}
                gr.update(label=i18n("spk1_prompt_audio_label")),
                gr.update(
                    label=i18n("spk1_prompt_text_label"),
                    placeholder=i18n("spk1_prompt_text_placeholder"),
                ),
                # spk2_prompt_{audio,text}
                gr.update(label=i18n("spk2_prompt_audio_label")),
                gr.update(
                    label=i18n("spk2_prompt_text_label"),
                    placeholder=i18n("spk2_prompt_text_placeholder"),
                ),
                # dialogue_text_input
                gr.update(
                    label=i18n("dialogue_text_input_label"),
                    placeholder=i18n("dialogue_text_input_placeholder"),
                ),
                # generate_btn
                gr.update(value=i18n("generate_btn_label")),
                # generate_audio
                gr.update(label=i18n("generated_audio_label")),
            ]

        lang_choice.change(
            fn=_change_component_language,
            inputs=[lang_choice],
            outputs=[
                title_desc,
                voice_mode_choice,
                spk1_prompt_audio,
                spk1_prompt_text,
                spk2_prompt_audio,
                spk2_prompt_text,
                dialogue_text_input,
                generate_btn,
                generate_audio,
            ],
        )

        # Voice clone mode action
        def _change_prompt_input_visibility(voice_mode):
            enable = voice_mode == 0
            return [gr.update(visible=enable), gr.update(visible=enable)]

        voice_mode_choice.change(
            fn=_change_prompt_input_visibility,
            inputs=[voice_mode_choice],
            outputs=[spk1_prompt_group, spk2_prompt_group],
        )
        generate_btn.click(
            fn=dialogue_synthesis_function,
            inputs=[
                dialogue_text_input,
                voice_mode_choice,
                spk1_prompt_text,
                spk1_prompt_audio,
                spk2_prompt_text,
                spk2_prompt_audio,
            ],
            outputs=[generate_audio],
        )
    return page


def make_demo(model):
    initiate_model(model)
    # UI
    page = render_interface()
    return page
