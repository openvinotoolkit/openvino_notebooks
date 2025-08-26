import gradio as gr
import librosa
import copy
from acestep.ui.components import GENRE_PRESETS, TAG_DEFAULT, update_tags_from_preset, create_output_ui
import ipywidgets as widgets


LYRIC_DEFAULT = """[verse]
Neon lights they flicker bright
City hums in dead of night
Rhythms pulse through concrete veins
Lost in echoes of refrains
"""


def create_text2music_ui(gr, ov_pipeline, sample_data_func=None, lora_path=None):
    with gr.Row():
        with gr.Column():
            with gr.Row(equal_height=True):
                audio_duration = gr.Slider(
                    0,
                    90.0,
                    step=0.01,
                    value=10,
                    label="Audio Duration",
                    interactive=True,
                    info="The length of the generated audio in sec. Longer duration takes more time.",
                    scale=9,
                )

            with gr.Row(equal_height=True):
                lora_enable = gr.Checkbox(label="Enable LoRA", value=False, info="Enable generation with Rap LoRA.", elem_id="lora_checkbox")

            def toggle_lora_in_pipeline(is_checked):
                if is_checked:
                    ov_pipeline.load_lora(lora_path)
                else:
                    ov_pipeline.load_lora("none")

            lora_enable.change(fn=toggle_lora_in_pipeline, inputs=[lora_enable])

            with gr.Row(equal_height=True):
                audio2audio_enable = gr.Checkbox(
                    label="Enable Audio2Audio", value=False, info="Enable Audio-to-Audio generation using a reference audio.", elem_id="audio2audio_checkbox"
                )

            ref_audio_input = gr.Audio(
                type="filepath", label="Reference Audio (for Audio2Audio task)", visible=False, elem_id="ref_audio_input", show_download_button=True
            )
            ref_audio_strength = gr.Slider(
                label="Refer audio strength",
                minimum=0.0,
                maximum=1.0,
                step=0.01,
                value=0.5,
                elem_id="ref_audio_strength",
                visible=False,
                interactive=True,
            )

            def toggle_ref_audio_visibility(is_checked):
                return (
                    gr.update(visible=is_checked, elem_id="ref_audio_input"),
                    gr.update(visible=is_checked, elem_id="ref_audio_strength"),
                )

            audio2audio_enable.change(
                fn=toggle_ref_audio_visibility,
                inputs=[audio2audio_enable],
                outputs=[ref_audio_input, ref_audio_strength],
            )

            with gr.Column(scale=2):
                with gr.Group():
                    gr.Markdown(
                        """<center>Support tags, descriptions, and scene. Use commas to separate different tags.<br>Tags and lyrics examples are from AI music generation community.</center>"""
                    )
                    with gr.Row():
                        genre_preset = gr.Dropdown(
                            choices=["Custom"] + list(GENRE_PRESETS.keys()),
                            value="Custom",
                            label="Preset",
                            scale=1,
                        )
                        prompt = gr.Textbox(
                            lines=1,
                            label="Tags",
                            max_lines=4,
                            value=TAG_DEFAULT,
                            scale=9,
                        )

            # Add the change event for the preset dropdown
            genre_preset.change(fn=update_tags_from_preset, inputs=[genre_preset], outputs=[prompt])
            with gr.Group():
                gr.Markdown(
                    """<center>Support lyric structure tags like [verse], [chorus], and [bridge] to separate different parts of the lyrics.<br>Use [instrumental] or [inst] to generate instrumental music. Not support genre structure tag in lyrics</center>"""
                )
                lyrics = gr.Textbox(
                    lines=9,
                    label="Lyrics",
                    max_lines=13,
                    value=LYRIC_DEFAULT,
                )

            with gr.Accordion("Basic Settings", open=False):
                infer_step = gr.Slider(
                    minimum=1,
                    maximum=200,
                    step=1,
                    value=20,
                    label="Infer Steps",
                    interactive=True,
                )
                guidance_scale = gr.Slider(
                    minimum=0.0,
                    maximum=30.0,
                    step=0.1,
                    value=15.0,
                    label="Guidance Scale",
                    interactive=True,
                    info="When guidance_scale_lyric > 1 and guidance_scale_text > 1, the guidance scale will not be applied.",
                )
                guidance_scale_text = gr.Slider(
                    minimum=0.0,
                    maximum=10.0,
                    step=0.1,
                    value=0.0,
                    label="Guidance Scale Text",
                    interactive=True,
                    info="Guidance scale for text condition. It can only apply to cfg. set guidance_scale_text=5.0, guidance_scale_lyric=1.5 for start",
                )
                guidance_scale_lyric = gr.Slider(
                    minimum=0.0,
                    maximum=10.0,
                    step=0.1,
                    value=0.0,
                    label="Guidance Scale Lyric",
                    interactive=True,
                )

                manual_seeds = gr.Textbox(
                    label="manual seeds (default None)",
                    placeholder="1,2,3,4",
                    value=None,
                    info="Seed for the generation",
                )

            with gr.Accordion("Advanced Settings", open=False):
                scheduler_type = gr.Radio(
                    ["euler", "heun"],
                    value="euler",
                    label="Scheduler Type",
                    elem_id="scheduler_type",
                    info="Scheduler type for the generation. euler is recommended. heun will take more time.",
                )
                cfg_type = gr.Radio(
                    ["cfg", "apg", "cfg_star"],
                    value="apg",
                    label="CFG Type",
                    elem_id="cfg_type",
                    info="CFG type for the generation. apg is recommended. cfg and cfg_star are almost the same.",
                )
                use_erg_lyric = gr.Checkbox(
                    label="use ERG for lyric",
                    value=True,
                    info="Use Entropy Rectifying Guidance for lyric. It will multiple a temperature to the attention to make a weaker lyric condition and make better diversity.",
                )
                use_erg_diffusion = gr.Checkbox(
                    label="use ERG for diffusion",
                    value=True,
                    info="The same but apply to diffusion model's attention.",
                )

                omega_scale = gr.Slider(
                    minimum=-100.0,
                    maximum=100.0,
                    step=0.1,
                    value=10.0,
                    label="Granularity Scale",
                    interactive=True,
                    info="Granularity scale for the generation. Higher values can reduce artifacts",
                )

                guidance_interval = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.5,
                    label="Guidance Interval",
                    interactive=True,
                    info="Guidance interval for the generation. 0.5 means only apply guidance in the middle steps (0.25 * infer_steps to 0.75 * infer_steps)",
                )
                guidance_interval_decay = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.0,
                    label="Guidance Interval Decay",
                    interactive=True,
                    info="Guidance interval decay for the generation. Guidance scale will decay from guidance_scale to min_guidance_scale in the interval. 0.0 means no decay.",
                )
                min_guidance_scale = gr.Slider(
                    minimum=0.0,
                    maximum=200.0,
                    step=0.1,
                    value=3.0,
                    label="Min Guidance Scale",
                    interactive=True,
                    info="Min guidance scale for guidance interval decay's end scale",
                )
                oss_steps = gr.Textbox(
                    label="OSS Steps",
                    placeholder="16, 29, 52, 96, 129, 158, 172, 183, 189, 200",
                    value=None,
                    info="Optimal Steps for the generation.",
                )

            text2music_bnt = gr.Button("Generate", variant="primary")

        with gr.Column():
            outputs, input_params_json = create_output_ui()
            with gr.Tab("retake"):
                retake_variance = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.2, label="variance")
                retake_seeds = gr.Textbox(label="retake seeds (default None)", placeholder="", value=None)
                retake_bnt = gr.Button("Retake", variant="primary")
                retake_outputs, retake_input_params_json = create_output_ui("Retake")

                def retake_process_func(json_data, retake_variance, retake_seeds):
                    return ov_pipeline(
                        audio_duration=json_data["audio_duration"],
                        prompt=json_data["prompt"],
                        lyrics=json_data["lyrics"],
                        infer_step=json_data["infer_step"],
                        guidance_scale=json_data["guidance_scale"],
                        scheduler_type=json_data["scheduler_type"],
                        cfg_type=json_data["cfg_type"],
                        omega_scale=json_data["omega_scale"],
                        manual_seeds=json_data["actual_seeds"],
                        guidance_interval=json_data["guidance_interval"],
                        guidance_interval_decay=json_data["guidance_interval_decay"],
                        min_guidance_scale=json_data["min_guidance_scale"],
                        use_erg_tag=False,
                        use_erg_lyric=json_data["use_erg_lyric"],
                        use_erg_diffusion=json_data["use_erg_diffusion"],
                        oss_steps=", ".join(map(str, json_data["oss_steps"])),
                        guidance_scale_text=json_data["guidance_scale_text"] if "guidance_scale_text" in json_data else 0.0,
                        guidance_scale_lyric=json_data["guidance_scale_lyric"] if "guidance_scale_lyric" in json_data else 0.0,
                        retake_seeds=retake_seeds,
                        retake_variance=retake_variance,
                        task="retake",
                    )

                retake_bnt.click(
                    fn=retake_process_func,
                    inputs=[
                        input_params_json,
                        retake_variance,
                        retake_seeds,
                    ],
                    outputs=retake_outputs + [retake_input_params_json],
                )
            with gr.Tab("repainting"):
                retake_variance = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.2, label="variance")
                retake_seeds = gr.Textbox(label="repaint seeds (default None)", placeholder="", value=None)
                repaint_start = gr.Slider(
                    minimum=0.0,
                    maximum=90.0,
                    step=0.01,
                    value=0.0,
                    label="Repaint Start Time",
                    interactive=True,
                )
                repaint_end = gr.Slider(
                    minimum=0.0,
                    maximum=90.0,
                    step=0.01,
                    value=10.0,
                    label="Repaint End Time",
                    interactive=True,
                )
                repaint_source = gr.Radio(
                    ["text2music", "last_repaint", "upload"],
                    value="text2music",
                    label="Repaint Source",
                    elem_id="repaint_source",
                )

                repaint_source_audio_upload = gr.Audio(
                    label="Upload Audio",
                    type="filepath",
                    visible=False,
                    elem_id="repaint_source_audio_upload",
                    show_download_button=True,
                )
                repaint_source.change(
                    fn=lambda x: gr.update(visible=x == "upload", elem_id="repaint_source_audio_upload"),
                    inputs=[repaint_source],
                    outputs=[repaint_source_audio_upload],
                )

                repaint_bnt = gr.Button("Repaint", variant="primary")
                repaint_outputs, repaint_input_params_json = create_output_ui("Repaint")

                def repaint_process_func(
                    text2music_json_data,
                    repaint_json_data,
                    retake_variance,
                    retake_seeds,
                    repaint_start,
                    repaint_end,
                    repaint_source,
                    repaint_source_audio_upload,
                    prompt,
                    lyrics,
                    infer_step,
                    guidance_scale,
                    scheduler_type,
                    cfg_type,
                    omega_scale,
                    manual_seeds,
                    guidance_interval,
                    guidance_interval_decay,
                    min_guidance_scale,
                    use_erg_lyric,
                    use_erg_diffusion,
                    oss_steps,
                    guidance_scale_text,
                    guidance_scale_lyric,
                ):
                    if repaint_source == "upload":
                        src_audio_path = repaint_source_audio_upload
                        audio_duration = librosa.get_duration(filename=src_audio_path)
                        json_data = {"audio_duration": audio_duration}
                    elif repaint_source == "text2music":
                        json_data = text2music_json_data
                        src_audio_path = json_data["audio_path"]
                    elif repaint_source == "last_repaint":
                        json_data = repaint_json_data
                        src_audio_path = json_data["audio_path"]

                    return ov_pipeline(
                        audio_duration=json_data["audio_duration"],
                        prompt=prompt,
                        lyrics=lyrics,
                        infer_step=infer_step,
                        guidance_scale=guidance_scale,
                        scheduler_type=scheduler_type,
                        cfg_type=cfg_type,
                        omega_scale=omega_scale,
                        manual_seeds=manual_seeds,
                        guidance_interval=guidance_interval,
                        guidance_interval_decay=guidance_interval_decay,
                        min_guidance_scale=min_guidance_scale,
                        use_erg_tag=False,
                        use_erg_lyric=use_erg_lyric,
                        use_erg_diffusion=use_erg_diffusion,
                        oss_steps=oss_steps,
                        guidance_scale_text=guidance_scale_text,
                        guidance_scale_lyric=guidance_scale_lyric,
                        retake_seeds=retake_seeds,
                        retake_variance=retake_variance,
                        task="repaint",
                        repaint_start=repaint_start,
                        repaint_end=repaint_end,
                        src_audio_path=src_audio_path,
                    )

                repaint_bnt.click(
                    fn=repaint_process_func,
                    inputs=[
                        input_params_json,
                        repaint_input_params_json,
                        retake_variance,
                        retake_seeds,
                        repaint_start,
                        repaint_end,
                        repaint_source,
                        repaint_source_audio_upload,
                        prompt,
                        lyrics,
                        infer_step,
                        guidance_scale,
                        scheduler_type,
                        cfg_type,
                        omega_scale,
                        manual_seeds,
                        guidance_interval,
                        guidance_interval_decay,
                        min_guidance_scale,
                        use_erg_lyric,
                        use_erg_diffusion,
                        oss_steps,
                        guidance_scale_text,
                        guidance_scale_lyric,
                    ],
                    outputs=repaint_outputs + [repaint_input_params_json],
                )
            with gr.Tab("edit"):
                edit_prompt = gr.Textbox(lines=2, label="Edit Tags", max_lines=4)
                edit_lyrics = gr.Textbox(lines=9, label="Edit Lyrics", max_lines=13)
                retake_seeds = gr.Textbox(label="edit seeds (default None)", placeholder="", value=None)

                edit_type = gr.Radio(
                    ["only_lyrics", "remix"],
                    value="only_lyrics",
                    label="Edit Type",
                    elem_id="edit_type",
                    info="`only_lyrics` will keep the whole song the same except lyrics difference. Make your diffrence smaller, e.g. one lyrc line change.\nremix can change the song melody and genre",
                )
                edit_n_min = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.6,
                    label="edit_n_min",
                    interactive=True,
                )
                edit_n_max = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    value=1.0,
                    label="edit_n_max",
                    interactive=True,
                )

                def edit_type_change_func(edit_type):
                    if edit_type == "only_lyrics":
                        n_min = 0.6
                        n_max = 1.0
                    elif edit_type == "remix":
                        n_min = 0.2
                        n_max = 0.4
                    return n_min, n_max

                edit_type.change(
                    edit_type_change_func,
                    inputs=[edit_type],
                    outputs=[edit_n_min, edit_n_max],
                )

                edit_source = gr.Radio(
                    ["text2music", "last_edit", "upload"],
                    value="text2music",
                    label="Edit Source",
                    elem_id="edit_source",
                )
                edit_source_audio_upload = gr.Audio(
                    label="Upload Audio",
                    type="filepath",
                    visible=False,
                    elem_id="edit_source_audio_upload",
                    show_download_button=True,
                )
                edit_source.change(
                    fn=lambda x: gr.update(visible=x == "upload", elem_id="edit_source_audio_upload"),
                    inputs=[edit_source],
                    outputs=[edit_source_audio_upload],
                )

                edit_bnt = gr.Button("Edit", variant="primary")
                edit_outputs, edit_input_params_json = create_output_ui("Edit")

                def edit_process_func(
                    text2music_json_data,
                    edit_input_params_json,
                    edit_source,
                    edit_source_audio_upload,
                    prompt,
                    lyrics,
                    edit_prompt,
                    edit_lyrics,
                    edit_n_min,
                    edit_n_max,
                    infer_step,
                    guidance_scale,
                    scheduler_type,
                    cfg_type,
                    omega_scale,
                    manual_seeds,
                    guidance_interval,
                    guidance_interval_decay,
                    min_guidance_scale,
                    use_erg_lyric,
                    use_erg_diffusion,
                    oss_steps,
                    guidance_scale_text,
                    guidance_scale_lyric,
                    retake_seeds,
                ):
                    if edit_source == "upload":
                        src_audio_path = edit_source_audio_upload
                        audio_duration = librosa.get_duration(filename=src_audio_path)
                        json_data = {"audio_duration": audio_duration}
                    elif edit_source == "text2music":
                        json_data = text2music_json_data
                        src_audio_path = json_data["audio_path"]
                    elif edit_source == "last_edit":
                        json_data = edit_input_params_json
                        src_audio_path = json_data["audio_path"]

                    if not edit_prompt:
                        edit_prompt = prompt
                    if not edit_lyrics:
                        edit_lyrics = lyrics

                    return ov_pipeline(
                        audio_duration=json_data["audio_duration"],
                        prompt=prompt,
                        lyrics=lyrics,
                        infer_step=infer_step,
                        guidance_scale=guidance_scale,
                        scheduler_type=scheduler_type,
                        cfg_type=cfg_type,
                        omega_scale=omega_scale,
                        manual_seeds=manual_seeds,
                        guidance_interval=guidance_interval,
                        guidance_interval_decay=guidance_interval_decay,
                        min_guidance_scale=min_guidance_scale,
                        use_erg_tag=False,
                        use_erg_lyric=use_erg_lyric,
                        use_erg_diffusion=use_erg_diffusion,
                        oss_steps=oss_steps,
                        guidance_scale_text=guidance_scale_text,
                        guidance_scale_lyric=guidance_scale_lyric,
                        task="edit",
                        src_audio_path=src_audio_path,
                        edit_target_prompt=edit_prompt,
                        edit_target_lyrics=edit_lyrics,
                        edit_n_min=edit_n_min,
                        edit_n_max=edit_n_max,
                        retake_seeds=retake_seeds,
                    )

                edit_bnt.click(
                    fn=edit_process_func,
                    inputs=[
                        input_params_json,
                        edit_input_params_json,
                        edit_source,
                        edit_source_audio_upload,
                        prompt,
                        lyrics,
                        edit_prompt,
                        edit_lyrics,
                        edit_n_min,
                        edit_n_max,
                        infer_step,
                        guidance_scale,
                        scheduler_type,
                        cfg_type,
                        omega_scale,
                        manual_seeds,
                        guidance_interval,
                        guidance_interval_decay,
                        min_guidance_scale,
                        use_erg_lyric,
                        use_erg_diffusion,
                        oss_steps,
                        guidance_scale_text,
                        guidance_scale_lyric,
                        retake_seeds,
                    ],
                    outputs=edit_outputs + [edit_input_params_json],
                )
            with gr.Tab("extend"):
                extend_seeds = gr.Textbox(label="extend seeds (default None)", placeholder="", value=None)
                left_extend_length = gr.Slider(
                    minimum=0.0,
                    maximum=20.0,
                    step=0.01,
                    value=0.0,
                    label="Left Extend Length",
                    interactive=True,
                )
                right_extend_length = gr.Slider(
                    minimum=0.0,
                    maximum=20.0,
                    step=0.01,
                    value=5.0,
                    label="Right Extend Length",
                    interactive=True,
                )
                extend_source = gr.Radio(
                    ["text2music", "last_extend", "upload"],
                    value="text2music",
                    label="Extend Source",
                    elem_id="extend_source",
                )

                extend_source_audio_upload = gr.Audio(
                    label="Upload Audio",
                    type="filepath",
                    visible=False,
                    elem_id="extend_source_audio_upload",
                    show_download_button=True,
                )
                extend_source.change(
                    fn=lambda x: gr.update(visible=x == "upload", elem_id="extend_source_audio_upload"),
                    inputs=[extend_source],
                    outputs=[extend_source_audio_upload],
                )

                extend_bnt = gr.Button("Extend", variant="primary")
                extend_outputs, extend_input_params_json = create_output_ui("Extend")

                def extend_process_func(
                    text2music_json_data,
                    extend_input_params_json,
                    extend_seeds,
                    left_extend_length,
                    right_extend_length,
                    extend_source,
                    extend_source_audio_upload,
                    prompt,
                    lyrics,
                    infer_step,
                    guidance_scale,
                    scheduler_type,
                    cfg_type,
                    omega_scale,
                    manual_seeds,
                    guidance_interval,
                    guidance_interval_decay,
                    min_guidance_scale,
                    use_erg_lyric,
                    use_erg_diffusion,
                    oss_steps,
                    guidance_scale_text,
                    guidance_scale_lyric,
                ):
                    if extend_source == "upload":
                        src_audio_path = extend_source_audio_upload
                        # get audio duration
                        audio_duration = librosa.get_duration(filename=src_audio_path)
                        json_data = {"audio_duration": audio_duration}
                    elif extend_source == "text2music":
                        json_data = text2music_json_data
                        src_audio_path = json_data["audio_path"]
                    elif extend_source == "last_extend":
                        json_data = extend_input_params_json
                        src_audio_path = json_data["audio_path"]

                    repaint_start = -left_extend_length
                    repaint_end = json_data["audio_duration"] + right_extend_length
                    return ov_pipeline(
                        audio_duration=json_data["audio_duration"],
                        prompt=prompt,
                        lyrics=lyrics,
                        infer_step=infer_step,
                        guidance_scale=guidance_scale,
                        scheduler_type=scheduler_type,
                        cfg_type=cfg_type,
                        omega_scale=omega_scale,
                        manual_seeds=manual_seeds,
                        guidance_interval=guidance_interval,
                        guidance_interval_decay=guidance_interval_decay,
                        min_guidance_scale=min_guidance_scale,
                        use_erg_tag=False,
                        use_erg_lyric=use_erg_lyric,
                        use_erg_diffusion=use_erg_diffusion,
                        oss_steps=oss_steps,
                        guidance_scale_text=guidance_scale_text,
                        guidance_scale_lyric=guidance_scale_lyric,
                        retake_seeds=extend_seeds,
                        retake_variance=1.0,
                        task="extend",
                        repaint_start=repaint_start,
                        repaint_end=repaint_end,
                        src_audio_path=src_audio_path,
                    )

                extend_bnt.click(
                    fn=extend_process_func,
                    inputs=[
                        input_params_json,
                        extend_input_params_json,
                        extend_seeds,
                        left_extend_length,
                        right_extend_length,
                        extend_source,
                        extend_source_audio_upload,
                        prompt,
                        lyrics,
                        infer_step,
                        guidance_scale,
                        scheduler_type,
                        cfg_type,
                        omega_scale,
                        manual_seeds,
                        guidance_interval,
                        guidance_interval_decay,
                        min_guidance_scale,
                        use_erg_lyric,
                        use_erg_diffusion,
                        oss_steps,
                        guidance_scale_text,
                        guidance_scale_lyric,
                    ],
                    outputs=extend_outputs + [extend_input_params_json],
                )

    def ov_pipeline_wrap(
        audio_duration,
        prompt,
        lyrics,
        infer_step,
        guidance_scale,
        scheduler_type,
        cfg_type,
        omega_scale,
        manual_seeds,
        guidance_interval,
        guidance_interval_decay,
        min_guidance_scale,
        use_erg_lyric,
        use_erg_diffusion,
        oss_steps,
        guidance_scale_text,
        guidance_scale_lyric,
        audio2audio_enable,
        ref_audio_strength,
        ref_audio_input,
    ):
        return ov_pipeline(
            audio_duration=audio_duration,
            prompt=prompt,
            lyrics=lyrics,
            infer_step=infer_step,
            guidance_scale=guidance_scale,
            scheduler_type=scheduler_type,
            cfg_type=cfg_type,
            omega_scale=omega_scale,
            manual_seeds=manual_seeds,
            guidance_interval=guidance_interval,
            guidance_interval_decay=guidance_interval_decay,
            min_guidance_scale=min_guidance_scale,
            use_erg_tag=False,
            use_erg_lyric=use_erg_lyric,
            use_erg_diffusion=use_erg_diffusion,
            oss_steps=oss_steps,
            guidance_scale_text=guidance_scale_text,
            guidance_scale_lyric=guidance_scale_lyric,
            audio2audio_enable=audio2audio_enable,
            ref_audio_strength=ref_audio_strength,
            ref_audio_input=ref_audio_input,
        )

    text2music_bnt.click(
        fn=ov_pipeline_wrap,
        inputs=[
            audio_duration,
            prompt,
            lyrics,
            infer_step,
            guidance_scale,
            scheduler_type,
            cfg_type,
            omega_scale,
            manual_seeds,
            guidance_interval,
            guidance_interval_decay,
            min_guidance_scale,
            use_erg_lyric,
            use_erg_diffusion,
            oss_steps,
            guidance_scale_text,
            guidance_scale_lyric,
            audio2audio_enable,
            ref_audio_strength,
            ref_audio_input,
        ],
        outputs=outputs + [input_params_json],
    )


def make_demo(pipeline, data_sampler):
    with gr.Blocks(
        title="ACE-Step Model with OpenVINO DEMO",
    ) as demo:
        gr.Markdown(
            """
            <h1 style="text-align: center;">Music generation with ACE-Step model and OpenVINO</h1>
        """
        )
        with gr.Tab("text2music"):
            create_text2music_ui(gr=gr, ov_pipeline=pipeline, sample_data_func=data_sampler.sample)
    return demo


def update_audio_widget():
    options = ["Retake", "Repainting", "Edit", "Extend"]
    return widgets.ToggleButtons(options=options, description="Choose next operation with audio:", disabled=False, button_style="info", value="Repainting")


def setup_update_audio_widgets(update_audio_widget):
    if update_audio_widget.value == "Retake":
        task = "retake"
        w = widgets.FloatSlider(value=0.2, min=0.0, max=1.0, step=0.01, description="Variance")
        vbox = widgets.VBox([w])
    elif update_audio_widget.value == "Repainting":
        task = "repaint"
        w1 = widgets.FloatSlider(value=0.2, min=0.0, max=1.0, step=0.01, description="Variance")
        w2 = widgets.FloatSlider(value=0, min=0, max=15, step=0.01, description="Repaint Start")
        w3 = widgets.FloatSlider(value=4, min=0, max=15, step=0.01, description="Repaint End")
        vbox = widgets.VBox([w1, w2, w3])
    elif update_audio_widget.value == "Edit":
        task = "edit"
        w1 = widgets.Textarea(value="", description="Edit lyric")
        w2 = widgets.Textarea(value="classical, orchestral, strings, piano, 60 bpm, elegant, emotive, timeless, instrumental", description="Edit tags")
        w3 = widgets.FloatSlider(value=0.2, min=0, max=1, step=0.01, description="edit_n_min")
        w4 = widgets.FloatSlider(value=0.4, min=0, max=1, step=0.01, description="edit_n_max")
        vbox = widgets.VBox([w1, w2, w3, w4])
    elif update_audio_widget.value == "Extend":
        task = "extend"
        w1 = widgets.FloatSlider(value=0.0, min=0, max=10, step=0.01, description="Left Extend Length")
        w2 = widgets.FloatSlider(value=5.0, min=0, max=10, step=0.01, description="Right Extend Length")
        vbox = widgets.VBox([w1, w2])

    return vbox, task


def get_inputs_base_on_setup_widget(base_inputs, source_audio_path, setup_widgets, task):
    extra_inputs = copy.deepcopy(base_inputs)
    extra_inputs.update({"task": task, "src_audio_path": source_audio_path})
    if task == "retake":
        extra_inputs["retake_variance"] = setup_widgets.children[0].value
        del extra_inputs["src_audio_path"]
    elif task == "repaint":
        extra_inputs["retake_variance"] = setup_widgets.children[0].value
        extra_inputs["repaint_start"] = setup_widgets.children[1].value
        extra_inputs["repaint_end"] = setup_widgets.children[2].value
    elif task == "edit":
        extra_inputs["edit_target_lyrics"] = setup_widgets.children[0].value if setup_widgets.children[0].value else extra_inputs["lyrics"]
        extra_inputs["edit_target_prompt"] = setup_widgets.children[1].value if setup_widgets.children[1].value else extra_inputs["prompt"]
        extra_inputs["edit_n_min"] = setup_widgets.children[2].value
        extra_inputs["edit_n_max"] = setup_widgets.children[3].value
    elif task == "extend":
        extra_inputs["repaint_start"] = -setup_widgets.children[0].value
        extra_inputs["repaint_end"] = base_inputs["audio_duration"] + setup_widgets.children[1].value

    return extra_inputs


def get_model_compression_format_widgets():
    return widgets.Dropdown(
        options=["FP16", "INT8", "INT4"],
        value="FP16",
        description="Model format:",
    )
