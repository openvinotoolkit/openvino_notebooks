from typing import Callable
import gradio as gr


def make_demo(fn: Callable):
    demo = gr.Interface(
        fn=fn,
        inputs=[
            gr.Textbox(label="Prompt"),
            gr.Textbox(label="Negative prompt"),
            gr.Slider(32, 1280, value=702, label="Width", step=32),
            gr.Slider(32, 720, value=480, label="Height", step=32),
            gr.Slider(9, 257, value=25, label="Number of frames", step=8),
            gr.Slider(10, 50, value=30, label="Number of inference steps", step=1),
            gr.Slider(0, 1000000, value=42, label="Seed", step=1),
        ],
        outputs=gr.Video(label="Result"),
        examples=[
            [
                "A woman with light skin, wearing a blue jacket and a black hat with a veil, looks down and to her right, then back up as she speaks; she has brown hair styled in an updo, light brown eyebrows, and is wearing a white collared shirt under her jacket; the camera remains stationary on her face as she speaks; the background is out of focus, but shows trees and people in period clothing; the scene is captured in real-life footage.",
                "worst quality, inconsistent motion, blurry, jittery, distorted",
                704,
                480,
                25,
                30,
                42,
            ],
            [
                """The camera pans over a snow-covered mountain range, revealing a vast expanse of snow-capped peaks and valleys.The mountains are covered in a thick layer of snow, with some areas appearing almost white while others have a slightly darker, almost grayish hue. The peaks are jagged and irregular, with some rising sharply into the sky while others are more rounded. The valleys are deep and narrow, with steep slopes that are also covered in snow. The trees in the foreground are mostly bare, with only a few leaves remaining on their branches. The sky is overcast, with thick clouds obscuring the sun. The overall impression is one of peace and tranquility, with the snow-covered mountains standing as a testament to the power and beauty of nature.""",
                "worst quality, inconsistent motion, blurry, jittery, distorted",
                704,
                480,
                25,
                30,
                42,
            ],
            [
                """A man in a dimly lit room talks on a vintage telephone, hangs up, and looks down with a sad expression. He holds the black rotary phone to his right ear with his right hand, his left hand holding a rocks glass with amber liquid. He wears a brown suit jacket over a white shirt, and a gold ring on his left ring finger. His short hair is neatly combed, and he has light skin with visible wrinkles around his eyes. The camera remains stationary, focused on his face and upper body. The room is dark, lit only by a warm light source off-screen to the left, casting shadows on the wall behind him. The scene appears to be from a movie.""",
                "worst quality, inconsistent motion, blurry, jittery, distorted",
                704,
                480,
                25,
                30,
                42,
            ],
            [
                """A woman with long brown hair and light skin smiles at another woman with long blonde hair. The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek. The camera angle is a close-up, focused on the woman with brown hair's face. The lighting is warm and natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage.""",
                "worst quality, inconsistent motion, blurry, jittery, distorted",
                704,
                480,
                25,
                30,
                42,
            ],
            [
                """A woman walks away from a white Jeep parked on a city street at night, then ascends a staircase and knocks on a door. The woman, wearing a dark jacket and jeans, walks away from the Jeep parked on the left side of the street, her back to the camera; she walks at a steady pace, her arms swinging slightly by her sides; the street is dimly lit, with streetlights casting pools of light on the wet pavement; a man in a dark jacket and jeans walks past the Jeep in the opposite direction; the camera follows the woman from behind as she walks up a set of stairs towards a building with a green door; she reaches the top of the stairs and turns left, continuing to walk towards the building; she reaches the door and knocks on it with her right hand; the camera remains stationary, focused on the doorway; the scene is captured in real-life footage.""",
                "worst quality, inconsistent motion, blurry, jittery, distorted",
                704,
                480,
                25,
                30,
                42,
            ],
            [
                """A clear, turquoise river flows through a rocky canyon, cascading over a small waterfall and forming a pool of water at the bottom.The river is the main focus of the scene, with its clear water reflecting the surrounding trees and rocks. The canyon walls are steep and rocky, with some vegetation growing on them. The trees are mostly pine trees, with their green needles contrasting with the brown and gray rocks. The overall tone of the scene is one of peace and tranquility.""",
                "worst quality, inconsistent motion, blurry, jittery, distorted",
                704,
                480,
                25,
                30,
                42,
            ],
        ],
        allow_flagging="never",
    )
    return demo
